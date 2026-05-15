import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn
from torch.nn import Parameter
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.datasets import Planetoid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cora/Citeseer 参数
dropout_rate = 0.5
weight_decay = 5e-4
learning_rate = 0.01
reg_weight = 0.05
cluster_weight = 0.1

# # Photo 参数
# learning_rate = 0.03
# weight_decay = 1e-05
# reg_weight = 0.01
# cluster_weight = 0.02
# dropout_rate = 0.4

# # Computers 参数
# learning_rate = 0.05
# weight_decay = 5e-06
# reg_weight = 0.01
# cluster_weight = 0.05
# dropout_rate = 0.4

dataset = 'Cora'
# dataset = 'Citeseer'
# dataset = 'Photo'
# dataset = 'Computers'
path = '../data'

dataset = Planetoid(root=path, name='Cora')
data = dataset[0]
A = np.eye(len(data.x), len(data.x))

# print(data.edge_index)
for i in range(len(data.edge_index[0])):
    a, b = data.edge_index[0][i], data.edge_index[1][i]
    a, b = int(a), int(b)

    A[a][b] = 1
    A[b][a] = 1

D = np.sum(A, axis=1)
# --- 计算 Spectral Regularization 用的拉普拉斯矩阵 ---
# 计算 D^-1/2
d_inv_sqrt = np.power(D, -0.5)
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
D_inv_sqrt = np.diag(d_inv_sqrt)
# 归一化拉普拉斯矩阵 L_reg = I - D^-1/2 * A * D^-1/2
L_reg = np.eye(len(data.x)) - np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)
L_reg = torch.FloatTensor(L_reg)
# -----------------------------------------------

D = np.diag(D)
L = np.dot(A, np.linalg.inv(D))  # =L=AD^-1
# print(L)
alpha = .1
M = alpha * np.linalg.inv((np.eye(len(data.x), len(data.x)) - (1 - alpha) * L))
M = torch.FloatTensor(M)
L = torch.FloatTensor(L)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 128
        # ... (之前的定义保持不变) ...
        self.feature_gate = Parameter(torch.ones(data.num_features))
        self.nn1 = nn.Linear(data.num_features, n)
        self.nn3 = nn.Linear(n, dataset.num_classes)
        self.nn2 = nn.Linear(data.num_features, dataset.num_classes)

        # [新增] 聚类中心：假设我们在隐空间做聚类，或者直接在输出空间做
        # 这里为了配合 z 的维度 (dataset.num_classes)，我们设为 num_classes
        # 如果 z 是 hidden 维度，这里就是 n。
        # 你代码里的 z = x + self.nn2(x_gated)，维度是 num_classes
        self.cluster_centers = Parameter(torch.Tensor(dataset.num_classes, dataset.num_classes))
        self.dropout_rate = dropout_rate
        self.reset_parameters()

    def reset_parameters(self):
        # ... (之前的初始化) ...
        nn.init.xavier_uniform_(self.nn1.weight)
        nn.init.xavier_uniform_(self.nn2.weight)
        nn.init.xavier_uniform_(self.nn3.weight)
        # [新增] 初始化聚类中心
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, data):
        # ... (完全保持不变) ...
        x, edge_index = data.x, data.edge_index
        gate_strength = torch.sigmoid(self.feature_gate)
        x_gated = x * gate_strength

        x = F.dropout(x_gated, p=self.dropout_rate, training=self.training)
        x = F.relu(self.nn1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.nn3(x)

        z = x + self.nn2(x_gated)  # z 的维度是 [num_nodes, num_classes]
        x_out = M.mm(z)

        # 依然只返回这两个值，符合你的约束
        return F.log_softmax(x_out, dim=1), z


def target_distribution(q):
    """
    计算目标分布 P，目的是让 Q 分布更尖锐（Sharpening），即更确信属于某类。
    """
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def get_clustering_loss(z, cluster_centers):
    """
    计算 z 和聚类中心的软分配概率 Q
    使用 Student's t-distribution kernel (dof=1)
    """
    # 1. 计算距离 ||z_i - center_j||^2
    # z: [N, D], centers: [K, D] -> dist: [N, K]
    z_expand = z.unsqueeze(1)  # [N, 1, D]
    c_expand = cluster_centers.unsqueeze(0)  # [1, K, D]
    dist_sq = torch.sum((z_expand - c_expand) ** 2, dim=2)

    # 2. 转换为 t-分布概率 q_ij
    q = 1.0 / (1.0 + dist_sq)
    q = q / torch.sum(q, dim=1, keepdim=True)

    # 3. 计算目标分布 p_ij
    p = target_distribution(q)

    # 4. 计算 KL 散度 Loss
    # KL(P || Q) = sum(p * log(p/q))
    log_q = torch.log(q + 1e-6)  # 加 epsilon 防止 nan
    # 注意：F.kl_div 默认输入 log_probability
    loss_cluster = F.kl_div(log_q, p, reduction='batchmean')

    return loss_cluster


data = data.to(device)
L_reg = L_reg.to(device)
M = M.to(device)

test_accs = []
num_runs = 20

for run in range(num_runs):
    print(f'\n--- Run {run + 1}/{num_runs} ---')

    # 每次循环重新初始化模型和优化器
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 在训练前最好用 K-Means 初始化中心，收敛会快很多（可选，但推荐）
    # 注意：KMeans 运行在 CPU 上，所以需要来回转换设备
    kmeans = KMeans(n_clusters=dataset.num_classes, n_init=2 * dataset.num_classes)
    k = model(data)
    b = k[1]
    _z_init = model(data)[1].detach().cpu().numpy()
    kmeans.fit(_z_init)
    model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_).to(device)

    # 变量用于记录本轮 run 的最佳状态
    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(200):
        # --- 训练阶段 ---
        model.train()
        optimizer.zero_grad()
        out, z = model(data)

        # 1. 分类 Loss
        loss_cls = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        # 2. 谱正则化 (Local Smoothness)
        reg_loss = (1.0 / len(data.x)) * torch.trace(torch.mm(z.t(), torch.mm(L_reg, z)))

        # 3. [新增] 聚类 Loss (Global Compactness)
        # 强迫 z 围绕在几个中心周围
        loss_cluster = get_clustering_loss(z, model.cluster_centers)

        # 融合 Loss
        loss = loss_cls + reg_weight * reg_loss + cluster_weight * loss_cluster

        loss.backward()
        optimizer.step()

        # --- 评估与模型选择 (Validation) ---
        model.eval()
        with torch.no_grad():
            out_eval, _ = model(data)
            _, pred = out_eval.max(dim=1)

            # 计算验证集精度
            correct_val = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            val_acc = correct_val / int(data.val_mask.sum())

            # 计算测试集精度 (用于记录 best_val 对应的 test_acc)
            correct_test = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            test_acc = correct_test / int(data.test_mask.sum())

            # 计算训练集精度 (仅用于显示)
            correct_train = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
            train_acc = correct_train / int(data.train_mask.sum())

            # 核心逻辑：如果当前 val_acc 优于历史最佳，则更新记录
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

        # 打印日志
        if epoch % 10 == 0:
            print(
                f'  Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    # 循环结束后，输出本轮的最佳结果
    print(f'  Run {run + 1} Best Val Acc: {best_val_acc:.4f}, Corresponding Test Acc: {best_test_acc:.4f}')
    test_accs.append(best_test_acc)

# 计算统计结果
test_accs = np.array(test_accs)
print(f'\n================================')
print(f'Runs: {num_runs}')
print(f'Mean Accuracy: {np.mean(test_accs):.4f}')
print(f'Std Deviation: {np.std(test_accs):.4f}')
print(f'================================')
