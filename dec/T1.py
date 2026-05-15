import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from models.layers import WeightedGNNConv


def run():
    def generate_single_random_split(data, num_classes, train_ratio=0.6, val_ratio=0.2):
        num_nodes = data.num_nodes
        y = data.y

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        for c in range(num_classes):
            class_idx = (y == c).nonzero(as_tuple=False).view(-1)

            shuffled_idx = class_idx[torch.randperm(class_idx.size(0))]

            num_class_nodes = shuffled_idx.size(0)
            train_end = int(num_class_nodes * train_ratio)
            val_end = train_end + int(num_class_nodes * val_ratio)

            train_mask[shuffled_idx[:train_end]] = True
            val_mask[shuffled_idx[train_end:val_end]] = True
            test_mask[shuffled_idx[val_end:]] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    # --- 1. 环境与数据加载 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = Planetoid(root='../data', name='Cora')
    data = dataset[0].to(device)
    # data = generate_single_random_split(data, dataset.num_classes)
    # --- 2. 超参数设置 ---
    dropout_rate = 0.2
    weight_decay = 5e-4
    learning_rate = 5e-3

    # Loss 权重
    reg_weight = 0.05
    cluster_weight = 0.1

    # 训练阶段控制
    epochs = 500
    warmup_epochs = 50  # 前50个epoch只训练分类和图平滑，不加入聚类Loss

    # --- 3. 核心组件定义 ---

    def target_distribution(q):
        """计算 DEC 目标分布 P"""
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def get_clustering_loss(z, cluster_centers):
        """计算 KL 散度聚类损失 (DEC Loss)"""
        z_expand = z.unsqueeze(1)
        c_expand = cluster_centers.unsqueeze(0)
        dist_sq = torch.sum((z_expand - c_expand) ** 2, dim=2)

        q = 1.0 / (1.0 + dist_sq)
        q = q / torch.sum(q, dim=1, keepdim=True)
        p = target_distribution(q)

        log_q = torch.log(q + 1e-6)
        loss_cluster = F.kl_div(log_q, p, reduction='batchmean')
        return loss_cluster

    class CoGNNLayer(nn.Module):
        """
        结合了动作网络(Action Network)和环境网络(Environment Network)的单层 Co-GNN
        """

        def __init__(self, in_channels, out_channels):
            super(CoGNNLayer, self).__init__()
            # 动作网络：根据节点特征输出4个动作的 logits [S, L, B, I]
            self.action_net = nn.Sequential(nn.Linear(in_channels, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 4))
            self.env_net = GCNConv(in_channels, out_channels)
            self.env_net1 = WeightedGNNConv(in_channels, 128, aggr='mean', bias=True)
            self.env_net2 = WeightedGNNConv(128, out_channels, aggr='mean', bias=True)
            self.env_net3 = WeightedGNNConv(in_channels, out_channels, aggr='mean', bias=True)

        def forward(self, x, edge_index, tau=0.1):
            # 1. 动作网络预测 (生成动作对数概率)
            action_logits = self.action_net(x)

            # 2. Gumbel-Softmax 采样动作 (hard=True 保证输出为 One-hot，但保留梯度)
            # 形状: [N, 4]
            if self.training:
                action_probs = F.gumbel_softmax(action_logits, tau=tau, hard=True)
            else:
                # 推理阶段直接取 argmax
                action_probs = F.one_hot(action_logits.argmax(dim=1), num_classes=4).float()

            # 3. 解析动作，生成边权重 (Mask)
            # edge_index[0] 是 source (u), edge_index[1] 是 target (v)
            row, col = edge_index

            # 节点 u 广播的概率：当它选择了 S(0) 或 B(2) 时
            broadcast_prob = action_probs[:, 0] + action_probs[:, 2]

            # 节点 v 监听的概率：当它选择了 S(0) 或 L(1) 时
            listen_prob = action_probs[:, 0] + action_probs[:, 1]

            # 这条边有效的概率 = u 广播 * v 监听
            edge_weight = broadcast_prob[row] * listen_prob[col]

            # 4. 环境网络执行基于权重的消息传递
            # x = self.env_net1(x, edge_index=edge_index, edge_weight=edge_weight)
            # out = self.env_net2(x, edge_index=edge_index, edge_weight=edge_weight)
            # out = self.env_net(x, edge_index, edge_weight)
            out = self.env_net3(x, edge_index=edge_index, edge_weight=edge_weight)  # best
            return out, edge_weight

    class CoGNN_DEC_Model(nn.Module):
        def __init__(self, num_features, hidden_dim, num_classes):
            super(CoGNN_DEC_Model, self).__init__()

            # 两层 Co-GNN
            self.layer1 = CoGNNLayer(num_features, hidden_dim)
            self.layer2 = CoGNNLayer(hidden_dim, num_classes)

            # DEC 聚类中心 (在隐空间 z 中)
            self.cluster_centers = Parameter(torch.Tensor(num_classes, num_classes))
            nn.init.xavier_uniform_(self.cluster_centers)

            self.dropout = dropout_rate

        def forward(self, x, edge_index, tau=0.1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, edge_weight1 = self.layer1(x, edge_index, tau)
            x = F.relu(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            # Z 为聚类隐空间表示
            z, edge_weight2 = self.layer2(x, edge_index, tau)

            return z, edge_weight2

    # --- 4. 训练与评估逻辑 ---
    model = CoGNN_DEC_Model(dataset.num_features, 128, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_test_acc = 0.0
    tau = 0.1  # Gumbel-Softmax 温度

    print(f'\n=== 开始训练 (前 {warmup_epochs} Epochs 为 Warm-up 阶段) ===')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z, dynamic_edge_weights = model(data.x, data.edge_index, tau=tau)

        # 1. 监督分类 Loss
        out = F.log_softmax(z, dim=1)
        loss_cls = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        # 2. 动态图平滑正则化 (Dynamic Dirichlet Energy)
        # 仅平滑那些没有被动作网络切断的边
        row, col = data.edge_index
        z_src, z_dst = z[row], z[col]
        # ||z_u - z_v||^2 结合 edge_weight
        reg_loss = (dynamic_edge_weights * torch.norm(z_src - z_dst, dim=1) ** 2).mean()

        # 3. 联合 Loss
        loss = loss_cls + reg_weight * reg_loss

        # 4. DEC 聚类启动逻辑 (Two-Stage Strategy)
        if epoch == warmup_epochs:
            print("\n>>> 预热结束，启动 KMeans 初始化聚类中心，激活 DEC 损失 <<<\n")
            model.eval()
            with torch.no_grad():
                z_init, _ = model(data.x, data.edge_index)

            kmeans = KMeans(n_clusters=dataset.num_classes, n_init=20)
            kmeans.fit(z_init.detach().cpu().numpy())
            model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_).to(device)
            model.train()

        # 如果过了 Warmup 阶段，加上聚类 Loss
        if epoch >= warmup_epochs:
            loss_cluster = get_clustering_loss(z, model.cluster_centers)
            loss = loss + cluster_weight * loss_cluster

        loss.backward()
        optimizer.step()

        # --- 评估阶段 ---
        model.eval()
        with torch.no_grad():
            z_eval, _ = model(data.x, data.edge_index)
            pred = z_eval.argmax(dim=1)

            train_acc = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum()) / int(data.train_mask.sum())
            val_acc = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum()) / int(data.val_mask.sum())
            test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum()) / int(data.test_mask.sum())

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

        if epoch % 10 == 0:
            stage = "Warm-up" if epoch < warmup_epochs else "Joint-Train"
            print(
                f'Epoch: {epoch:03d} [{stage}] | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')

    print(f'\n=== 训练完成 ===')
    print(f'最佳验证集精度对应的测试集精度: {best_test_acc:.4f}')
    return best_test_acc


metric_list = []

for i in range(1):
    test_acc = run()
    metric_list.append(test_acc)

print(np.average(metric_list), np.std(metric_list))
