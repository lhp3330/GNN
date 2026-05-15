import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from sklearn.cluster import KMeans
from scipy import sparse
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.nn import GATConv  # 【修改】新增 GATConv
from torch_geometric.utils import to_undirected

from torch_geometric.nn import LabelPropagation

import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==========================================
# 1. 全局超参数（最后冲刺配置）
# ==========================================
# Stage 1: GAT
GAT_HIDDEN = 64
GAT_LR = 0.005
GAT_WEIGHT_DECAY = 5e-4
GAT_DROPOUT = 0.6
GAT_EPOCHS = 200
GAT_HEADS = 8

TOP_K = 20  # 【核心】从 10 翻倍到 20，疯狂加边
CONFIDENCE_THRESHOLD = 0.60  # 【核心】从 0.8 大幅降到 0.6，覆盖更多节点
SIM_THRESHOLD = 0.0

# Stage 2: PPNP
PPNP_HIDDEN = 128
PPNP_LR = 0.01
PPNP_WEIGHT_DECAY = 5e-4
PPNP_DROPOUT = 0.5
PPNP_EPOCHS = 600
PPNP_ALPHA = 0.10  # 【核心】降到 0.1，让传播更远
POWER_ITER_K = 20  # 配合 alpha，增加传播步数

REG_WEIGHT = 0.05
CLUSTER_WEIGHT = 0.20

PRUNE_EPOCH = 100
JS_PRUNE_THRESHOLD = 0.40  # 【核心】大幅提高到 0.4，几乎不剪枝，只剪掉最垃圾的边

NUM_RUNS = 20
EVAL_START_EPOCH = 50

# Label Propagation
LP_NUM_LAYERS = 30  # 稍微增加层数
LP_ALPHA = 0.30  # 提高 alpha，更相信图结构


# ==========================================
# 2. 数据处理与辅助函数 (保持不变)
# ==========================================
def load_dataset(name):
    if name in ['Cora', 'Citeseer']:
        return Planetoid(root='./data', name=name)
    elif name in ['Computers', 'Photo']:
        return Amazon(root='./data', name=name)
    raise ValueError(f"Unknown dataset: {name}")


def split_data(data, dataset_name, num_train_per_class=20, num_val_per_class=30):
    num_classes = int(data.y.max().item()) + 1
    num_nodes = data.y.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.size(0))]
        train_mask[perm[:num_train_per_class]] = True
        val_mask[perm[num_train_per_class:num_train_per_class + num_val_per_class]] = True
        test_mask[perm[num_train_per_class + num_val_per_class:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


# ==========================================
# 3. Stage 1: 【全新 GAT 模型】替换旧 GCN
# ==========================================
class GAT(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, dropout, heads):
        super().__init__()
        # 第一层：多头注意力
        # 注意：GATConv 的第二个参数是 per-head 的维度
        self.conv1 = GATConv(num_features, hidden_dim // heads, heads=heads, dropout=dropout)
        # 第二层：单头注意力 (用于分类)，concat=False 表示不拼接，直接平均
        self.conv2 = GATConv(hidden_dim, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 输入 Dropout
        h = F.dropout(x, p=self.dropout, training=self.training)

        # 第一层 GAT + ELU
        h = self.conv1(h, edge_index)
        h = F.elu(h)  # GAT 原文使用 ELU
        z = h  # 保存这一层的输出用于加边 (Embedding)

        # 第二层 Dropout
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)

        return F.log_softmax(h, dim=1), z


def train_gat(data, num_classes):  # 【重命名】函数名以示区分
    # 初始化 GAT 模型
    model = GAT(data.num_features, num_classes, GAT_HIDDEN, GAT_DROPOUT, GAT_HEADS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GAT_LR, weight_decay=GAT_WEIGHT_DECAY)
    data_dev = data.to(device)
    best_val_acc = 0
    best_state = None

    for epoch in range(1, GAT_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data_dev.x, data_dev.edge_index)
        loss = F.nll_loss(out[data_dev.train_mask], data_dev.y[data_dev.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out, _ = model(data_dev.x, data_dev.edge_index)
                pred = out.argmax(dim=1)
                val_acc = (pred[data_dev.val_mask] == data_dev.y[data_dev.val_mask]).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out, z = model(data_dev.x, data_dev.edge_index)
        pred_labels = out.argmax(dim=1)
        pred_probs = out.exp()
        test_acc = (pred_labels[data_dev.test_mask] == data_dev.y[data_dev.test_mask]).float().mean().item()

    print(f"GAT best val_acc={best_val_acc:.4f}, test_acc={test_acc:.4f}")
    return pred_labels.cpu(), z.cpu(), pred_probs.cpu()


# (后面的 add_edges_by_prediction_confident 函数完全不需要改，直接保留即可)
def add_edges_by_prediction_confident(edge_index, pred_labels, pred_probs,
                                      embeddings, num_nodes, top_k, confidence_threshold, sim_threshold=0.0):
    max_probs = pred_probs.max(dim=1).values
    confident_mask = max_probs >= confidence_threshold
    emb_norm = F.normalize(embeddings.float(), p=2, dim=1)
    pred_np = pred_labels.numpy()
    confident_np = confident_mask.numpy()

    new_edges_src, new_edges_dst = [], []
    orig_edges = set(zip(edge_index[0].cpu().tolist(), edge_index[1].cpu().tolist()))
    num_classes = int(pred_np.max()) + 1

    for c in range(num_classes):
        class_nodes = np.where((pred_np == c) & confident_np)[0]
        if len(class_nodes) < 2: continue
        class_emb = emb_norm[class_nodes]
        sim_matrix = (class_emb @ class_emb.T).numpy()
        np.fill_diagonal(sim_matrix, -1)

        for local_i, global_i in enumerate(class_nodes):
            sims = sim_matrix[local_i]
            top_indices = np.argsort(sims)[::-1][:top_k]
            for local_j in top_indices:
                if sims[local_j] <= sim_threshold: continue
                global_j = int(class_nodes[local_j])
                if (global_i, global_j) not in orig_edges:
                    new_edges_src.append(global_i)
                    new_edges_dst.append(global_j)
                    orig_edges.update([(global_i, global_j), (global_j, global_i)])

    confident_count = confident_mask.sum().item()
    print(f"Confident nodes: {confident_count}/{num_nodes} ({confident_count / num_nodes * 100:.1f}%)")

    if len(new_edges_src) == 0:
        return edge_index.cpu(), 0

    new_edges = torch.tensor([new_edges_src + new_edges_dst, new_edges_dst + new_edges_src], dtype=torch.long)
    aug_edge_index = torch.cat([edge_index.cpu(), new_edges], dim=1)
    print(f"Original edges: {edge_index.shape[1]} | Added: {new_edges.shape[1]} | Total: {aug_edge_index.shape[1]}")
    return aug_edge_index, new_edges.shape[1]


# ==========================================
# 4. Stage 2: PPNP 相关运算与剪枝
# ==========================================
def build_ppnp_matrix_power_iter(edge_index, num_nodes, alpha, K=10, edge_weights=None):
    row = edge_index[0].cpu().tolist()
    col = edge_index[1].cpu().tolist()
    data_vals = edge_weights.cpu().tolist() if edge_weights is not None else [1.0] * len(row)

    self_loops = list(range(num_nodes))
    row, col = row + self_loops, col + self_loops
    data_vals = data_vals + [1.0] * num_nodes

    A_sp = sparse.csr_matrix((data_vals, (row, col)), shape=(num_nodes, num_nodes))
    A_sp = (A_sp + A_sp.T) / 2.0

    d_reg = np.array(A_sp.sum(axis=1)).flatten()
    D_inv_sqrt = sparse.diags(np.power(np.maximum(d_reg, 1e-12), -0.5))
    L_reg_sp = sparse.eye(num_nodes) - D_inv_sqrt @ A_sp @ D_inv_sqrt

    d = np.array(A_sp.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / np.maximum(d, 1e-12))
    T = A_sp @ D_inv

    M_sp = sparse.eye(num_nodes, dtype=np.float64) * alpha
    T_power, coeff = sparse.eye(num_nodes, dtype=np.float64), alpha
    for k in range(1, K + 1):
        T_power = T_power @ T
        coeff *= (1 - alpha)
        M_sp = M_sp + coeff * T_power

    return torch.FloatTensor(M_sp.toarray()), torch.FloatTensor(L_reg_sp.toarray())


def compute_q(z, centers):
    dist = torch.cdist(z, centers, p=2).pow(2)
    q = 1.0 / (1.0 + dist)
    return q / q.sum(dim=1, keepdim=True)


def compute_target_p(q):
    p = (q ** 2) / q.sum(dim=0, keepdim=True)
    return p / p.sum(dim=1, keepdim=True)


def cluster_kl_loss(q, p):
    return (p * torch.log(p / (q + 1e-10) + 1e-10)).sum(dim=1).mean()


def init_cluster_centers(model, data_x, num_classes):
    model.eval()
    with torch.no_grad():
        _, _, z_cluster = model(data_x)
    kmeans = KMeans(n_clusters=num_classes, n_init=20, random_state=42)
    kmeans.fit(z_cluster.cpu().numpy())
    model.cluster_centers.data.copy_(torch.FloatTensor(kmeans.cluster_centers_).to(data_x.device))
    print(f"  [Cluster] KMeans inertia: {kmeans.inertia_:.2f}")


def one_shot_prune(aug_edge_index, orig_num_edges, log_probs, z_cluster, cluster_centers, threshold):
    probs = log_probs.exp().detach().cpu().numpy() + 1e-9
    q = compute_q(z_cluster, cluster_centers).detach().cpu().numpy() + 1e-9

    new_src = aug_edge_index[0, orig_num_edges:].tolist()
    new_dst = aug_edge_index[1, orig_num_edges:].tolist()

    if len(new_src) == 0:
        return aug_edge_index

    p_cls, r_cls = probs[new_src], probs[new_dst]
    m_cls = 0.5 * (p_cls + r_cls)
    js_cls = 0.5 * np.sum(p_cls * np.log(p_cls / m_cls), axis=1) + 0.5 * np.sum(r_cls * np.log(r_cls / m_cls), axis=1)

    p_clu, r_clu = q[new_src], q[new_dst]
    m_clu = 0.5 * (p_clu + r_clu)
    js_clu = 0.5 * np.sum(p_clu * np.log(p_clu / m_clu), axis=1) + 0.5 * np.sum(r_clu * np.log(r_clu / m_clu), axis=1)

    js_combined = np.maximum(js_cls, js_clu)
    keep_aug_mask = js_combined <= threshold

    kept_aug_indices = np.where(keep_aug_mask)[0]
    orig_edges = aug_edge_index[:, :orig_num_edges]
    new_edge_index = torch.cat([orig_edges, aug_edge_index[:, orig_num_edges:][:, kept_aug_indices]], dim=1) if len(
        kept_aug_indices) > 0 else orig_edges

    print(
        f"  [OneShot Prune] removed={(~keep_aug_mask).sum()} | kept={keep_aug_mask.sum()} | Final edges={new_edge_index.shape[1]}")
    return new_edge_index


# ==========================================
# 5. Stage 2: PPNP 模型与训练循环
# ==========================================
class PPNP_ClusterPrune(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, dropout, M):
        super().__init__()
        self.M = M
        self.dropout_rate = dropout
        self.feature_gate = Parameter(torch.ones(num_features))
        self.nn1 = nn.Linear(num_features, hidden_dim)
        self.nn3 = nn.Linear(hidden_dim, hidden_dim)
        self.nn2 = nn.Linear(num_features, hidden_dim)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        # 【统一修改】ReLU → ELU，与 Citeseer 版本一致
        self.cluster_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cluster_centers = Parameter(torch.Tensor(num_classes, hidden_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        for m in [self.nn1, self.nn2, self.nn3, self.cls_head]:
            nn.init.xavier_uniform_(m.weight)
        for m in self.cluster_proj:
            if hasattr(m, 'weight'): nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, x):
        x_gated = x * torch.sigmoid(self.feature_gate)

        h = F.dropout(x_gated, p=self.dropout_rate, training=self.training)
        h = self.nn1(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.relu(h)
        h = self.nn3(h)

        z_shared = h + self.nn2(x_gated)
        log_prob = F.log_softmax(self.M.to(x.device) @ self.cls_head(z_shared), dim=1)

        # 【统一修改】残差连接 + L2 球面投影，与 Citeseer 版本一致
        z_cluster_raw = self.cluster_proj(z_shared)
        z_cluster = F.normalize(z_shared + z_cluster_raw, p=2, dim=1)

        return log_prob, z_shared, z_cluster


def train_stage2(data, aug_edge_index, orig_num_edges, num_classes, run_id):
    num_nodes = data.x.size(0)
    M, L_reg = build_ppnp_matrix_power_iter(aug_edge_index, num_nodes, PPNP_ALPHA, POWER_ITER_K)
    model = PPNP_ClusterPrune(data.num_features, num_classes, PPNP_HIDDEN, PPNP_DROPOUT, M.to(device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PPNP_LR, weight_decay=PPNP_WEIGHT_DECAY)
    data_dev = data.to(device)

    # 初始化标签传播模型（全局只初始化一次）
    lp_model = LabelPropagation(num_layers=LP_NUM_LAYERS, alpha=LP_ALPHA).to(device)

    init_cluster_centers(model, data_dev.x, num_classes)
    best_val_acc, best_test_acc, best_epoch, pruned = 0, 0, 0, False
    current_edge_index = aug_edge_index.to(device)  # 跟踪当前使用的边索引

    for epoch in range(1, PPNP_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        log_prob, z_shared, z_cluster = model(data_dev.x)

        loss_cls = F.nll_loss(log_prob[data_dev.train_mask], data_dev.y[data_dev.train_mask])
        loss_reg = torch.trace(z_shared.T @ L_reg.to(device) @ z_shared) / num_nodes
        q = compute_q(z_cluster, model.cluster_centers)
        loss_cluster = cluster_kl_loss(q, compute_target_p(q).detach())

        (loss_cls + REG_WEIGHT * loss_reg + CLUSTER_WEIGHT * loss_cluster).backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                log_prob, _, _ = model(data_dev.x)
                # ====================== 核心：标签传播后处理 ======================
                # 1. 把log_softmax转成原始概率
                probs = log_prob.exp()
                # 2. 用训练集的真实标签替换模型预测，保证传播起点正确
                probs[data_dev.train_mask] = F.one_hot(data_dev.y[data_dev.train_mask], num_classes).float()
                # 3. 用当前最新的边索引进行传播
                smoothed_probs = lp_model(probs, current_edge_index)
                # 4. 用平滑后的概率计算准确率
                pred = smoothed_probs.argmax(dim=1)
                # ================================================================

                val_acc = (pred[data_dev.val_mask] == data_dev.y[data_dev.val_mask]).float().mean().item()
                test_acc = (pred[data_dev.test_mask] == data_dev.y[data_dev.test_mask]).float().mean().item()
                if epoch >= EVAL_START_EPOCH and val_acc > best_val_acc:
                    best_val_acc, best_test_acc, best_epoch = val_acc, test_acc, epoch

        if epoch == PRUNE_EPOCH and not pruned:
            print(f"\n  === One-Shot Pruning at epoch {epoch} ===")
            init_cluster_centers(model, data_dev.x, num_classes)
            model.eval()
            with torch.no_grad():
                log_prob_prune, _, z_cluster_prune = model(data_dev.x)

            new_edge_index = one_shot_prune(aug_edge_index, orig_num_edges, log_prob_prune, z_cluster_prune,
                                            model.cluster_centers, JS_PRUNE_THRESHOLD)
            print("  Rebuilding M matrix...")
            M_new, L_reg_new = build_ppnp_matrix_power_iter(new_edge_index, num_nodes, PPNP_ALPHA, POWER_ITER_K)
            model.M = M_new.to(device)
            L_reg = L_reg_new.to(device)

            # 更新当前边索引，后续LP会用这个剪枝后的边
            current_edge_index = new_edge_index.to(device)

            init_cluster_centers(model, data_dev.x, num_classes)
            for param_group in optimizer.param_groups: param_group['lr'] = PPNP_LR * 0.5
            print(f"  LR reduced to {PPNP_LR * 0.5:.4f}\n")
            pruned = True

    print(f"  Run {run_id}: best_val={best_val_acc:.4f} best_test={best_test_acc:.4f} @ epoch {best_epoch}")
    return best_test_acc


# ==========================================
# 6. 主执行入口（新增集成学习逻辑）
# ==========================================
def train_stage2_save_probs(data, aug_edge_index, orig_num_edges, num_classes, run_id):
    """
    加入 Cosine Annealing 学习率调度器
    """
    num_nodes = data.x.size(0)
    M, L_reg = build_ppnp_matrix_power_iter(aug_edge_index, num_nodes, PPNP_ALPHA, POWER_ITER_K)
    model = PPNP_ClusterPrune(data.num_features, num_classes, PPNP_HIDDEN, PPNP_DROPOUT, M.to(device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PPNP_LR, weight_decay=PPNP_WEIGHT_DECAY)

    # ====================== 新增 1/3：学习率调度器 ======================
    # T_max: 总周期数，eta_min: 最小学习率 (初始 LR 的 1%)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PPNP_EPOCHS, eta_min=PPNP_LR * 0.01)
    # =====================================================================

    data_dev = data.to(device)

    lp_model = LabelPropagation(num_layers=LP_NUM_LAYERS, alpha=LP_ALPHA).to(device)

    init_cluster_centers(model, data_dev.x, num_classes)
    best_val_acc, best_test_acc, best_epoch, pruned = 0, 0, 0, False
    current_edge_index = aug_edge_index.to(device)
    best_model_state = None

    for epoch in range(1, PPNP_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        log_prob, z_shared, z_cluster = model(data_dev.x)

        loss_cls = F.nll_loss(log_prob[data_dev.train_mask], data_dev.y[data_dev.train_mask])
        loss_reg = torch.trace(z_shared.T @ L_reg.to(device) @ z_shared) / num_nodes
        q = compute_q(z_cluster, model.cluster_centers)
        loss_cluster = cluster_kl_loss(q, compute_target_p(q).detach())

        (loss_cls + REG_WEIGHT * loss_reg + CLUSTER_WEIGHT * loss_cluster).backward()
        optimizer.step()

        # ====================== 新增 2/3：调度器步进 ======================
        # 只在剪枝前使用调度器，剪枝后我们手动降低了 LR
        if not pruned:
            scheduler.step()
        # ==================================================================

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                log_prob, _, _ = model(data_dev.x)
                pred_raw = log_prob.argmax(dim=1)
                val_acc = (pred_raw[data_dev.val_mask] == data_dev.y[data_dev.val_mask]).float().mean().item()
                test_acc = (pred_raw[data_dev.test_mask] == data_dev.y[data_dev.test_mask]).float().mean().item()

                if epoch >= EVAL_START_EPOCH and val_acc > best_val_acc:
                    best_val_acc, best_test_acc, best_epoch = val_acc, test_acc, epoch
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == PRUNE_EPOCH and not pruned:
            print(f"\n  === One-Shot Pruning at epoch {epoch} ===")
            init_cluster_centers(model, data_dev.x, num_classes)
            model.eval()
            with torch.no_grad():
                log_prob_prune, _, z_cluster_prune = model(data_dev.x)

            new_edge_index = one_shot_prune(aug_edge_index, orig_num_edges, log_prob_prune, z_cluster_prune,
                                            model.cluster_centers, JS_PRUNE_THRESHOLD)
            print("  Rebuilding M matrix...")
            M_new, L_reg_new = build_ppnp_matrix_power_iter(new_edge_index, num_nodes, PPNP_ALPHA, POWER_ITER_K)
            model.M = M_new.to(device)
            L_reg = L_reg_new.to(device)
            current_edge_index = new_edge_index.to(device)

            init_cluster_centers(model, data_dev.x, num_classes)

            # ====================== 新增 3/3：重置优化器 ======================
            # 剪枝后手动降低 LR，并创建一个新的优化器，抛弃之前的动量
            for param_group in optimizer.param_groups:
                param_group['lr'] = PPNP_LR * 0.5
            # 可选：如果想让剪枝后的学习率也进行退火，可以在这里重新初始化 scheduler
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PPNP_EPOCHS - PRUNE_EPOCH, eta_min=PPNP_LR * 0.5 * 0.01)
            # ==================================================================

            print(f"  LR reduced to {PPNP_LR * 0.5:.4f}\n")
            pruned = True

    # 加载最佳模型并进行 LP 后处理
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        log_prob, _, _ = model(data_dev.x)
        probs = log_prob.exp()
        probs[data_dev.train_mask] = F.one_hot(data_dev.y[data_dev.train_mask], num_classes).float()
        best_smoothed_probs = lp_model(probs, current_edge_index)
        final_pred = best_smoothed_probs.argmax(dim=1)
        final_test_acc = (final_pred[data_dev.test_mask] == data_dev.y[data_dev.test_mask]).float().mean().item()

    print(
        f"  Run {run_id}: best_val={best_val_acc:.4f} (raw) | final_test={final_test_acc:.4f} (w/ LP) @ epoch {best_epoch}")
    return final_test_acc, best_smoothed_probs, best_val_acc


def run_experiment(dataset_name):
    dataset = load_dataset(dataset_name)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)

    torch.manual_seed(42)
    np.random.seed(42)

    # 【关键修改】使用 Planetoid 官方固定 split，不再调用自定义随机 split
    # data = split_data(data, dataset_name)
    num_classes = int(data.y.max().item()) + 1

    print(f" 使用官方 Planetoid 固定 split")
    print(f"   Train nodes: {data.train_mask.sum().item():4d} "
          f"({data.train_mask.sum().item() / data.x.size(0) * 100:.2f}%)")
    print(f"   Val   nodes: {data.val_mask.sum().item():4d} "
          f"({data.val_mask.sum().item() / data.x.size(0) * 100:.2f}%)")
    print(f"   Test  nodes: {data.test_mask.sum().item():4d} "
          f"({data.test_mask.sum().item() / data.x.size(0) * 100:.2f}%)")
    print(f"   总节点数   : {data.x.size(0)}")

    print("\n--- Stage 1: GAT Training ---")
    pred_labels, embeddings, pred_probs = train_gat(data, num_classes)

    print("\n--- Confident Edge Augmentation ---")
    aug_edge_index, _ = add_edges_by_prediction_confident(
        data.edge_index, pred_labels, pred_probs, embeddings,
        data.x.size(0), TOP_K, CONFIDENCE_THRESHOLD, SIM_THRESHOLD)
    orig_num_edges = data.edge_index.shape[1]

    print(f"\n--- Stage 2: PPNP + ClusterPrune ({NUM_RUNS} runs) ---")
    final_test_accs = []

    for run in range(1, NUM_RUNS + 1):
        print(f"\n>>> Run {run}/{NUM_RUNS} <<<")
        torch.manual_seed(run * 42)
        np.random.seed(run * 42)

        test_acc, _, _ = train_stage2_save_probs(data, aug_edge_index.clone(),
                                                 orig_num_edges, num_classes, run)
        final_test_accs.append(test_acc)

    print(f"\n{'=' * 60}")
    print(f"  All {NUM_RUNS} Results: {np.round(final_test_accs, 4)}")

    mean_acc = np.mean(final_test_accs)
    std_acc = np.std(final_test_accs)

    print(f"\n  Final Report: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"{'=' * 60}\n")


run_experiment('Cora')
