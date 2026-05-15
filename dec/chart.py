import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================
# 1. 模拟数据准备 (GNN 刚输出的隐空间特征 Z)
# ==========================================
np.random.seed(42)
# 假设有 3 个簇，但特征比较分散（模拟还没聚类前）
n_samples = 150
z1 = np.random.randn(n_samples//3, 2) * 1.5 + [2, 2]
z2 = np.random.randn(n_samples//3, 2) * 1.2 + [6, 2]
z3 = np.random.randn(n_samples//3, 2) * 1.5 + [4, 6]
Z = np.vstack([z1, z2, z3])

# ==========================================
# DEC 核心数学函数
# ==========================================
def get_q(Z, centers):
    """计算软分配 Q (Student's t-distribution)"""
    dist_sq = np.sum((Z[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
    q = 1.0 / (1.0 + dist_sq)
    q = q / np.sum(q, axis=1, keepdims=True)
    return q

def get_p(q):
    """计算目标分布 P (Sharpening)"""
    weight = q ** 2 / np.sum(q, axis=0)
    p = (weight.T / np.sum(weight, axis=1)).T
    return p

# ==========================================
# 绘图逻辑 (模拟刚才的 4 个步骤)
# ==========================================
plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['SimHei'] # 允许图表显示中文
plt.rcParams['axes.unicode_minus'] = False

# --- Step 1: 初始状态 ---
plt.subplot(2, 2, 1)
plt.scatter(Z[:, 0], Z[:, 1], c='gray', alpha=0.6, edgecolors='w', s=50)
plt.title("Step 1: 初始隐空间 Z (GNN 输出的特征，较混乱)")
plt.grid(True, linestyle='--', alpha=0.5)

# --- Step 2: KMeans 初始化中心 ---
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Z)
centers = kmeans.cluster_centers_

plt.subplot(2, 2, 2)
# 根据距离简单染色
labels = kmeans.labels_
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
c_map = [colors[l] for l in labels]
plt.scatter(Z[:, 0], Z[:, 1], c=c_map, alpha=0.5, edgecolors='w', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=300, edgecolor='black', label='聚类中心 (Centers)')
plt.title("Step 2: K-Means 寻找初始中心 (确立引力源)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# --- Step 3: 软分配 Q 分布 (犹豫不决) ---
q = get_q(Z, centers)
plt.subplot(2, 2, 3)
plt.scatter(Z[:, 0], Z[:, 1], c=c_map, alpha=0.5, edgecolors='w', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=300, edgecolor='black')

# 选取几个边界点画线，展示它们受到多个中心的拉扯
boundary_points = [20, 60, 110, 85]
for idx in boundary_points:
    for c_idx in range(3):
        # 线的粗细代表概率 q 的大小
        linewidth = q[idx, c_idx] * 5
        plt.plot([Z[idx, 0], centers[c_idx, 0]], [Z[idx, 1], centers[c_idx, 1]],
                 'k-', alpha=0.4, linewidth=linewidth)
plt.title("Step 3: Q 分布 (软分配：边界节点受到多个中心的拉扯)")
plt.grid(True, linestyle='--', alpha=0.5)

# --- Step 4: 目标 P 分布牵引 (产生 KL 散度梯度) ---
p = get_p(q)
# 模拟梯度下降的拉扯效果 (简化的梯度力)
pull_force = np.zeros_like(Z)
for i in range(len(Z)):
    for j in range(3):
        # P 分布比 Q 分布更极端，差异产生拉力
        pull_force[i] += (p[i, j] - q[i, j]) * (centers[j] - Z[i])

Z_new = Z + pull_force * 2.0 # 放大拉力以便观察

plt.subplot(2, 2, 4)
plt.scatter(Z[:, 0], Z[:, 1], c='gray', alpha=0.2, s=50) # 原位置
plt.scatter(Z_new[:, 0], Z_new[:, 1], c=c_map, alpha=0.8, edgecolors='w', s=50) # 新位置
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=300, edgecolor='black')

# 画箭头表示移动方向
for i in range(0, len(Z), 3): # 每隔几个点画一个箭头避免太密
    plt.arrow(Z[i, 0], Z[i, 1], Z_new[i, 0]-Z[i, 0], Z_new[i, 1]-Z[i, 1],
              head_width=0.1, head_length=0.15, fc='black', ec='black', alpha=0.3)

plt.title("Step 4: P 分布牵引 (KL散度迫使特征向中心收敛)")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
