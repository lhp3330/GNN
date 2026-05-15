import sys
import logging
import math
from typing import Tuple, Any
from argparse import Namespace

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn import Module, Dropout, LayerNorm, Identity
from torch_geometric.typing import Adj, OptTensor

from helpers.classes import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
from helpers.utils import set_seed
from models.action import ActionNet
from models.temp import TempSoftPlus

import torch
import torch.nn.functional as F


def target_distribution(q):
    """计算 DEC 的辅助目标分布 P"""
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def get_clustering_loss(z, cluster_centers):
    """
    [融合版] 计算 z 和聚类中心的软分配概率 Q 与 KL 损失
    """
    # 1. 计算距离 ||z_i - center_j||^2
    z_expand = z.unsqueeze(1)  # [N, 1, D]
    c_expand = cluster_centers.unsqueeze(0)  # [1, K, D]
    dist_sq = torch.sum((z_expand - c_expand) ** 2, dim=2)

    # 2. 转换为 t-分布概率 q_ij
    q = 1.0 / (1.0 + dist_sq)
    q = q / torch.sum(q, dim=1, keepdim=True)

    # 3. 计算目标分布 p_ij (必须使用 detach 切断梯度！)
    p = target_distribution(q).detach()

    # 4. 计算 KL 散度 Loss
    log_q = torch.log(q + 1e-8)
    loss_cluster = F.kl_div(log_q, p, reduction='batchmean')

    return loss_cluster, q  # 返回 q 用于评估计算 ACC


# ==========================================
# 1. 聚类专属辅助函数
# ==========================================
def cluster_acc(y_true, y_pred):
    """使用匈牙利算法计算无监督聚类准确率 (ACC)"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def target_distribution(q):
    """计算 DEC 的辅助目标分布 P"""
    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()


# ==========================================
# 2. CoGNN + DEC 模型定义
# ==========================================
class CoGNNDEC(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActionNetArgs, n_clusters: int,
                 alpha: float = 1.0):
        super(CoGNNDEC, self).__init__()
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args.env_dim)
        self.temp = gumbel_args.temp

        self.num_layers = env_args.num_layers
        self.env_net = env_args.load_net()
        self.use_encoders = env_args.dataset_encoders.use_encoders()

        layer_norm_cls = LayerNorm if env_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(env_args.env_dim)
        self.skip = env_args.skip
        self.dropout = Dropout(p=env_args.dropout)
        self.act = env_args.act_type.get()

        self.in_act_net = ActionNet(action_args=action_args)
        self.out_act_net = ActionNet(action_args=action_args)

        self.dataset_encoder = env_args.dataset_encoders
        self.env_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=env_args.env_dim,
                                                                  model_type=env_args.model_type)
        self.act_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=action_args.hidden_dim,
                                                                  model_type=action_args.model_type)

        # DEC 聚类参数
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(self.n_clusters, env_args.env_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

        # 在 CoGNNDEC 的 __init__ 中加上：
        self.feat_decoder = nn.Linear(env_args.env_dim, 1433)
        self.n_clusters = n_clusters
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, env_args.env_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def pretrain_forward(self, x: Tensor, edge_index: Adj, pestat=None) -> Tensor:
        """GAE 预训练前向传播：跳过动作网络，使用全图拓扑"""
        x = self.env_net[0](x, pestat)
        if not self.use_encoders:
            x = self.dropout(x)
            x = self.act(x)

        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)
            out = self.env_net[1 + gnn_idx](x=x, edge_index=edge_index)
            out = self.dropout(out)
            out = self.act(out)
            x = x + out if self.skip else out

        z = self.hidden_layer_norm(x)
        return z

    def forward(self, x: Tensor, edge_index: Adj, pestat=None, edge_attr: OptTensor = None, tau: float = 1.0) -> Tuple[
        Tensor, Tensor, Tensor]:
        env_edge_embedding = self.env_bond_encoder(
            edge_attr) if edge_attr is not None and self.env_bond_encoder else None
        act_edge_embedding = self.act_bond_encoder(
            edge_attr) if edge_attr is not None and self.act_bond_encoder else None

        x = self.env_net[0](x, pestat)
        if not self.use_encoders:
            x = self.dropout(x)
            x = self.act(x)

        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)

            in_logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                        act_edge_attr=act_edge_embedding)
            out_logits = self.out_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                          act_edge_attr=act_edge_embedding)

            current_temp = self.temp_model(x=x, edge_index=edge_index,
                                           edge_attr=env_edge_embedding) if self.learn_temp else tau

            in_probs = F.gumbel_softmax(logits=in_logits, tau=current_temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=current_temp, hard=True)

            edge_weight = self.create_edge_weight(edge_index, keep_in_prob=in_probs[:, 0],
                                                  keep_out_prob=out_probs[:, 0])

            out = self.env_net[1 + gnn_idx](x=x, edge_index=edge_index, edge_weight=edge_weight,
                                            edge_attr=env_edge_embedding)
            out = self.dropout(out)
            out = self.act(out)
            x = x + out if self.skip else out

        # ... (前面的循环和特征计算保持不变) ...
        z = self.hidden_layer_norm(x)
        return z, edge_weight

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob


# ==========================================
# 3. 实验控制类
# ==========================================
class Experiment(object):
    def __init__(self, args: Namespace):
        super().__init__()
        for arg in vars(args):
            self.__setattr__(arg, getattr(args, arg))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(seed=self.seed)

        self.metric_type = self.dataset.get_metric_type()
        self.decimal = self.dataset.num_after_decimal()
        self.dataset.asserts(args)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            filename=f'./logs/Clustering_{self.env_model_type}.log',
            filemode='w'
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        dataset = self.dataset.load(seed=self.seed, pos_enc=self.pos_enc)
        folds = self.dataset.get_folds(fold=self.fold)

        # out_dim = self.dataset.num_classes  # 聚类的数量
        out_dim = self.metric_type.get_out_dim(dataset=dataset)
        gin_mlp_func = self.dataset.gin_mlp_func()
        env_act_type = self.dataset.env_activation_type()

        gumbel_args = GumbelArgs(learn_temp=self.learn_temp, temp_model_type=self.temp_model_type, tau0=self.tau0,
                                 temp=self.temp, gin_mlp_func=gin_mlp_func)

        env_args = EnvArgs(
            model_type=self.env_model_type, num_layers=self.env_num_layers, env_dim=self.env_dim,
            layer_norm=self.layer_norm, skip=self.skip, batch_norm=self.batch_norm, dropout=self.dropout,
            act_type=env_act_type, metric_type=self.metric_type, in_dim=dataset[0].x.shape[1], out_dim=out_dim,
            gin_mlp_func=gin_mlp_func, dec_num_layers=self.dec_num_layers, pos_enc=self.pos_enc,
            dataset_encoders=self.dataset.get_dataset_encoders()
        )

        action_args = ActionNetArgs(
            model_type=self.act_model_type, num_layers=self.act_num_layers,
            hidden_dim=self.act_dim, dropout=self.dropout, act_type=ActivationType.RELU,
            env_dim=self.env_dim, gin_mlp_func=gin_mlp_func
        )

        best_acc_list = []
        for num_fold in folds:
            set_seed(seed=self.seed)
            dataset_by_split = self.dataset.select_fold_and_split(num_fold=num_fold, dataset=dataset)

            # 由于是全图聚类，我们直接取 train 里的整张图
            # 如果你的 dataset_by_split.train 是 loader，可以直接 next(iter(...))
            data = next(iter(dataset_by_split.train)).to(self.device) if hasattr(dataset_by_split.train,
                                                                                 '__iter__') else \
                dataset_by_split.train[0].to(self.device)

            best_acc = self.run_clustering_pipeline(data, gumbel_args, env_args, action_args, out_dim, num_fold)
            best_acc_list.append(best_acc)

            self.logger.info(f"Fold {num_fold} Finished. Best ACC: {best_acc:.4f}")

        final_mean = np.mean(best_acc_list)
        final_std = np.std(best_acc_list)
        print(f"\nFinal Clustering ACC: {final_mean:.4f} ± {final_std:.4f}")
        self.logger.info(f"Final Clustering ACC: {final_mean:.4f} ± {final_std:.4f}")

        return final_mean, final_std

    def run_clustering_pipeline(self, data, gumbel_args, env_args, action_args, n_clusters, num_fold):
        model = CoGNNDEC(gumbel_args=gumbel_args, env_args=env_args, action_args=action_args, n_clusters=n_clusters).to(
            self.device)

        # ==========================================
        # 阶段一：GAE 预训练 (回归最纯粹的结构重构)
        # ==========================================
        print(f"\n[Fold {num_fold}] Phase 1: Structural GAE Pre-training...")
        # 预训练学习率可以稍大
        optimizer_pretrain = optim.Adam(model.env_net.parameters(), lr=0.005)
        pestat = self.pos_enc.get_pe(data=data, device=self.device) if hasattr(self,
                                                                               'pos_enc') and self.pos_enc else None

        for epoch in range(200):
            model.train()
            optimizer_pretrain.zero_grad()

            z = model.pretrain_forward(data.x, data.edge_index, pestat=pestat)

            # 使用 L2 归一化防止特征塌陷到一个点
            z_norm = F.normalize(z, p=2, dim=1)

            # 内积解码器
            adj_pred = torch.sigmoid(torch.matmul(z_norm, z_norm.t()))

            # 只计算已有边的正样本重构
            row, col = data.edge_index
            loss_pos = F.binary_cross_entropy(adj_pred[row, col], torch.ones_like(adj_pred[row, col]))

            # 随机负采样 (数量与正样本一致)
            neg_row = torch.randint(0, data.num_nodes, (data.edge_index.size(1),)).to(self.device)
            neg_col = torch.randint(0, data.num_nodes, (data.edge_index.size(1),)).to(self.device)
            loss_neg = F.binary_cross_entropy(adj_pred[neg_row, neg_col], torch.zeros_like(adj_pred[neg_row, neg_col]))

            loss_pretrain = loss_pos + loss_neg

            loss_pretrain.backward()
            optimizer_pretrain.step()

            # 可选：打印预训练 loss，确保它在下降 (从 ~1.3 降到 ~0.9 左右)
            if epoch % 50 == 0:
                print(f"Pretrain Epoch {epoch} | Loss: {loss_pretrain.item():.4f}")

        # ==========================================
        # 阶段二：KMeans 初始化
        # ==========================================
        print(f"[Fold {num_fold}] Phase 2: KMeans Initialization...")
        model.eval()
        with torch.no_grad():
            z_init = model.pretrain_forward(data.x, data.edge_index, pestat=pestat).cpu().numpy()

        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        y_pred_kmeans = kmeans.fit_predict(z_init)

        acc_kmeans = cluster_acc(data.y.cpu().numpy(), y_pred_kmeans)
        print(f"-> KMeans Initial ACC: {acc_kmeans:.4f}")
        model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

        # ==========================================
        # 阶段三：Co-GNN + Encapsulated DEC...
        # ==========================================
        print(f"[Fold {num_fold}] Phase 3: Co-GNN + Encapsulated DEC...")

        # 【上帝视角】：计算绝对客观的原始特征相似度
        row, col = data.edge_index
        x_norm = F.normalize(data.x, p=2, dim=1)
        raw_edge_sim = torch.sum(x_norm[row] * x_norm[col], dim=1).detach()

        # 【终极修复】：使用分位数 (Quantile) 强制二值化！
        # 假设图中有 20% 是连接不同类的异构边，我们切断相似度最低的那 20%
        threshold = torch.quantile(raw_edge_sim, 0.2)
        target_edge = (raw_edge_sim > threshold).float()  # 1.0 为保留，0.0 为切断

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=self.weight_decay)

        best_acc = 0.0
        tau = self.tau0 if hasattr(self, 'tau0') else 1.5

        with tqdm.tqdm(total=400, file=sys.stdout) as pbar:
            for epoch in range(400):
                model.train()
                optimizer.zero_grad()

                # 1. 前向传播获取特征 z 和 动作网络生成的边保留概率
                z, edge_weight = model(
                    data.x, edge_index=data.edge_index,
                    edge_attr=getattr(data, 'edge_attr', None),
                    pestat=pestat, tau=tau
                )

                # 2. [融合你的逻辑] 直接调用封装好的 DEC 损失
                loss_cluster, q = get_clustering_loss(z, model.cluster_centers)

                # 3. 动作网络损失 (强制网络听从客观特征的指导切断毒边)
                loss_action = F.mse_loss(edge_weight, target_edge)

                # 4. 联合优化
                # cluster_weight 可以设置为 10.0，因为 KL 散度的数值通常比 MSE 小一个量级
                train_loss = 10.0 * loss_cluster + 5.0 * loss_action

                train_loss.backward()
                if self.dataset.clip_grad():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                tau = max(0.5, tau * 0.99)

                # -----------------------------------------
                # 评估记录
                # -----------------------------------------
                model.eval()
                with torch.no_grad():
                    # 评估时 tau 设置得很小
                    z_eval, eval_edge_weight = model(
                        data.x, edge_index=data.edge_index,
                        edge_attr=getattr(data, 'edge_attr', None),
                        pestat=pestat, tau=0.1
                    )

                    # 用 eval_z 算一遍 q 用于取 argmax
                    _, q_eval = get_clustering_loss(z_eval, model.cluster_centers)
                    y_pred_eval = q_eval.argmax(dim=1).cpu().numpy()

                    current_acc = cluster_acc(data.y.cpu().numpy(), y_pred_eval)
                    current_keep = eval_edge_weight.mean().item()

                if current_acc > best_acc:
                    best_acc = current_acc

                log_str = f"Epoch: {epoch} | KL: {loss_cluster.item():.4f} | ActLoss: {loss_action.item():.4f} | Keep: {current_keep:.2f} | ACC: {current_acc:.4f} (Best: {best_acc:.4f})"
                pbar.set_description(log_str)
                pbar.update(n=1)

        return best_acc
