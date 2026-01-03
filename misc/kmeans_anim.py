import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs

# データ生成
np.random.seed(0)
n_samples = 300
n_clusters = 4

data, true_labels = make_blobs(
    n_samples=n_samples, centers=n_clusters, cluster_std=1.0, random_state=0
)

# 初期クラスタ中心をランダムに設定
centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)]

# 色を定義
colors = ["blue", "orange", "green", "purple"]

# 各反復のデータを保存
history = []


def kmeans_step(data, centroids):
    """K-means の 1 ステップを実行"""
    # 距離計算
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    # 各点を最も近いクラスタに割り当て
    labels = np.argmin(distances, axis=1)
    # 各クラスタの新しい中心を計算
    new_centroids = np.array(
        [data[labels == i].mean(axis=0) for i in range(n_clusters)]
    )
    return labels, new_centroids


# アルゴリズムの各ステップを記録
for _ in range(20):  # 最大20反復まで実行
    labels, new_centroids = kmeans_step(data, centroids)
    history.append((data.copy(), centroids.copy(), labels.copy()))
    if np.all(centroids == new_centroids):  # 収束チェック
        break
    centroids = new_centroids

# アニメーションのプロット
fig, ax = plt.subplots(figsize=(6, 6))


def update(frame):
    ax.clear()
    data, centroids, labels = history[frame]
    for cluster in range(n_clusters):
        cluster_points = data[labels == cluster]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=30,
            color=colors[cluster],
            label=f"Cluster {cluster}",
        )
    ax.scatter(
        centroids[:, 0], centroids[:, 1], s=200, c="red", marker="x", label="Centroids"
    )
    ax.set_title(f"K-means Clustering: Iteration {frame + 1}")
    ax.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
    ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
    ax.legend()


ani = FuncAnimation(fig, update, frames=len(history), repeat=True)
# plt.show()

ani.save("kmeans_anim.gif", writer="pillow", fps=1)
