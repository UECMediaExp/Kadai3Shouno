"""
【最終版】高次元データの次元削減：PCA, t-SNE, PCA+t-SNE の比較実験

==========================================
教育目的
==========================================
1. 各手法の特性と限界を理解する
2. 「PCA前処理がt-SNEを改善する」ケースを確認する
3. データの特性に応じた手法選択の重要性を学ぶ

==========================================
重要な発見（実験から得られた知見）
==========================================
- 「PCA+t-SNEが常に最良」というのは神話
- しかし、高次元ノイズがある場合、PCA前処理はt-SNEを大幅に改善する
- 最適な手法はデータの特性による

==========================================
参考文献
==========================================
- van der Maaten & Hinton (2008) "Visualizing Data using t-SNE", JMLR
  https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
  → 原論文でPCA前処理（50次元程度）を推奨

- Kobak & Berens (2019) "The art of using t-SNE for single-cell transcriptomics"
  https://www.nature.com/articles/s41467-019-13056-x
  → PCA前処理の効果を詳細に分析
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons, make_blobs
import warnings
warnings.filterwarnings('ignore')


def create_experiment_data(data_type='moons', n_samples=500, total_dim=100, 
                           noise_strength=0.4, signal_strength=1.5):
    """
    様々なタイプの教育用データを生成
    
    Parameters
    ----------
    data_type : str
        'moons': 三日月形
        'blobs': ガウシアンクラスター  
        'circles': 同心円
        'spiral': スパイラル
    """
    np.random.seed(42)
    
    if data_type == 'moons':
        X_low, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
        intrinsic_dim = 2
        
    elif data_type == 'blobs':
        X_low, y = make_blobs(n_samples=n_samples, n_features=3, centers=4, 
                              cluster_std=0.6, random_state=42)
        intrinsic_dim = 3
        
    elif data_type == 'circles':
        n_per = n_samples // 3
        actual = n_per * 3
        X_low = []
        y = []
        for i, r in enumerate([1, 2.5, 4]):
            theta = np.random.rand(n_per) * 2 * np.pi
            X_low.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
            y.extend([i] * n_per)
        X_low = np.vstack(X_low) + np.random.randn(actual, 2) * 0.15
        y = np.array(y)
        n_samples = actual
        intrinsic_dim = 2
        
    elif data_type == 'spiral':
        n_per = n_samples // 2
        actual = n_per * 2
        t = np.linspace(0, 4*np.pi, n_per)
        X1 = np.column_stack([t * np.cos(t), t * np.sin(t)]) + np.random.randn(n_per, 2) * 0.3
        X2 = np.column_stack([-t * np.cos(t), -t * np.sin(t)]) + np.random.randn(n_per, 2) * 0.3
        X_low = np.vstack([X1, X2])
        y = np.array([0] * n_per + [1] * n_per)
        n_samples = actual
        intrinsic_dim = 2
    
    # 高次元への埋め込み
    signal_dim = min(intrinsic_dim * 3, 10)
    expansion = np.random.randn(X_low.shape[1], signal_dim) * signal_strength
    X_signal = X_low @ expansion
    
    X_high = np.zeros((len(X_low), total_dim))
    X_high[:, :signal_dim] = X_signal
    X_high += np.random.randn(len(X_low), total_dim) * noise_strength
    
    return X_high, y, X_low


def compare_methods(X, y, title, pca_dim=15, ax_row=None, show_scores=True):
    """
    3つの手法を比較
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # t-SNE直接（高次元から）
    tsne1 = TSNE(n_components=2, perplexity=30, random_state=42, 
                 max_iter=1000, init='random')
    X_tsne = tsne1.fit_transform(X_scaled)
    
    # PCA + t-SNE
    pca_pre = PCA(n_components=pca_dim)
    X_pca_pre = pca_pre.fit_transform(X_scaled)
    tsne2 = TSNE(n_components=2, perplexity=30, random_state=42, 
                 max_iter=1000, init='pca')
    X_pca_tsne = tsne2.fit_transform(X_pca_pre)
    
    # スコア計算
    s1 = silhouette_score(X_pca, y)
    s2 = silhouette_score(X_tsne, y)
    s3 = silhouette_score(X_pca_tsne, y)
    
    results = {
        'X_pca': X_pca, 'X_tsne': X_tsne, 'X_pca_tsne': X_pca_tsne,
        'scores': (s1, s2, s3)
    }
    
    if ax_row is not None:
        n_classes = len(np.unique(y))
        cmap = 'Set1' if n_classes <= 9 else 'tab10'
        
        for ax, X_plot, name, score in zip(
            ax_row[:3], 
            [X_pca, X_tsne, X_pca_tsne],
            ['PCA', 't-SNE direct', 'PCA ->t-SNE'],
            [s1, s2, s3]
        ):
            ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=cmap, 
                      alpha=0.7, s=20, edgecolors='white', linewidths=0.3)
            title_str = f'{name}\n(Sil: {score:.3f})' if show_scores else name
            ax.set_title(title_str, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    
    return results


def main():
    print("\n" + "=" * 70)
    print("高次元データの次元削減：教育用比較実験")
    print("=" * 70)
    
    # ===================================
    # 実験1: 基本的な比較
    # ===================================
    print("\n【実験1】4種類のデータでの手法比較")
    print("-" * 50)
    
    fig1, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig1.suptitle('4 types data x 3 methods (embedding in 100D)', fontsize=14, fontweight='bold')
    
    data_types = [
        ('moons', 'moon type'),
        ('blobs', 'cluster type'),
        ('circles', 'circles type'),
        ('spiral', 'spiral type')
    ]
    
    all_results = {}
    
    for i, (dtype, name) in enumerate(data_types):
        X, y, X_low = create_experiment_data(dtype, n_samples=500, total_dim=100, 
                                              noise_strength=0.4)
        results = compare_methods(X, y, name, pca_dim=15, ax_row=axes[i])
        axes[i, 0].set_ylabel(name, fontsize=11, fontweight='bold')
        
        # スコアバー
        methods = ['PCA', 't-SNE', 'PCA+t-SNE']
        scores = results['scores']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = axes[i, 3].bar(range(3), scores, color=colors)
        axes[i, 3].set_xticks(range(3))
        axes[i, 3].set_xticklabels(['P', 'T', 'P+T'], fontsize=9)
        axes[i, 3].set_ylim([min(0, min(scores)-0.1), max(scores)*1.3])
        axes[i, 3].axhline(0, color='gray', linestyle='--', linewidth=0.5)
        
        best_idx = np.argmax(scores)
        for j, (bar, s) in enumerate(zip(bars, scores)):
            marker = '★' if j == best_idx else ''
            axes[i, 3].text(bar.get_x() + bar.get_width()/2, max(s+0.02, 0.05),
                           f'{s:.2f}{marker}', ha='center', fontsize=8)
        
        all_results[dtype] = {'scores': scores, 'best': methods[best_idx]}
        print(f"  {name}: PCA={scores[0]:.3f}, t-SNE={scores[1]:.3f}, PCA+t-SNE={scores[2]:.3f} → {methods[best_idx]}")
    
    axes[0, 0].set_title('PCA (2D)', fontsize=10)
    axes[0, 1].set_title('t-SNE direct', fontsize=10)
    axes[0, 2].set_title('PCA->t-SNE', fontsize=10)
    axes[0, 3].set_title('score comparison', fontsize=10)
    
    plt.tight_layout()
    fig1.savefig('comparison_4types.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("\n  → comparison_4types.png に保存")
    
    # ===================================
    # 実験2: ノイズレベルの影響
    # ===================================
    print("\n【実験2】ノイズレベルと各手法の性能")
    print("-" * 50)
    
    noise_levels = [0.1, 0.3, 0.5, 0.8, 1.2]
    
    fig2, axes2 = plt.subplots(len(noise_levels), 4, figsize=(14, 3*len(noise_levels)))
    fig2.suptitle('Noise level and methods(moon type)', fontsize=14, fontweight='bold')
    
    pca_scores = []
    tsne_scores = []
    pca_tsne_scores = []
    
    for i, noise in enumerate(noise_levels):
        X, y, _ = create_experiment_data('moons', n_samples=400, total_dim=100,
                                          noise_strength=noise)
        results = compare_methods(X, y, f'noise={noise}', pca_dim=10, ax_row=axes2[i])
        axes2[i, 0].set_ylabel(f'noise={noise}', fontsize=10)
        
        scores = results['scores']
        pca_scores.append(scores[0])
        tsne_scores.append(scores[1])
        pca_tsne_scores.append(scores[2])
        
        # スコアバー
        methods = ['PCA', 't-SNE', 'P+T']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = axes2[i, 3].bar(range(3), scores, color=colors)
        axes2[i, 3].set_xticks(range(3))
        axes2[i, 3].set_xticklabels(methods, fontsize=8)
        axes2[i, 3].set_ylim([min(0, min(scores)-0.1), max(max(scores)*1.3, 0.5)])
    
    plt.tight_layout()
    fig2.savefig('noise_sensitivity.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("  → noise_sensitivity.png に保存")
    
    # ノイズvs性能のプロット
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(noise_levels, pca_scores, 'o-', color='#3498db', label='PCA', linewidth=2, markersize=8)
    ax3.plot(noise_levels, tsne_scores, 's-', color='#e74c3c', label='t-SNE直接', linewidth=2, markersize=8)
    ax3.plot(noise_levels, pca_tsne_scores, '^-', color='#2ecc71', label='PCA→t-SNE', linewidth=2, markersize=8)
    ax3.set_xlabel('noise strength', fontsize=12)
    ax3.set_ylabel('Silhouette Score', fontsize=12)
    ax3.set_title('noise level and methods performance', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    
    fig3.savefig('noise_vs_performance.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("  → noise_vs_performance.png に保存")
    
    # ===================================
    # 実験3: 次元数の影響
    # ===================================
    print("\n【実験3】埋め込み次元数の影響")
    print("-" * 50)
    
    dims = [20, 50, 100, 200, 500]
    
    pca_by_dim = []
    tsne_by_dim = []
    pca_tsne_by_dim = []
    
    for dim in dims:
        X, y, _ = create_experiment_data('moons', n_samples=400, total_dim=dim,
                                          noise_strength=0.4)
        results = compare_methods(X, y, f'dim={dim}', pca_dim=min(15, dim-1))
        pca_by_dim.append(results['scores'][0])
        tsne_by_dim.append(results['scores'][1])
        pca_tsne_by_dim.append(results['scores'][2])
        print(f"  {dim}次元: PCA={results['scores'][0]:.3f}, t-SNE={results['scores'][1]:.3f}, PCA+t-SNE={results['scores'][2]:.3f}")
    
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(dims, pca_by_dim, 'o-', color='#3498db', label='PCA', linewidth=2, markersize=8)
    ax4.plot(dims, tsne_by_dim, 's-', color='#e74c3c', label='t-SNE direct', linewidth=2, markersize=8)
    ax4.plot(dims, pca_tsne_by_dim, '^-', color='#2ecc71', label='PCA->t-SNE', linewidth=2, markersize=8)
    ax4.set_xlabel('eembedded dimensions', fontsize=12)
    ax4.set_ylabel('Silhouette Score', fontsize=12)
    ax4.set_title('embedded dimensions and each method performance', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    fig4.savefig('dim_vs_performance.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("  → dim_vs_performance.png に保存")
    
    # ===================================
    # まとめ
    # ===================================
    print("\n" + "=" * 70)
    print("実験結果のまとめ")
    print("=" * 70)
    print("""
【主要な発見】

1. 手法の特性
   - PCA: 線形手法。クラスター分離には有効だが、非線形構造には限界
   - t-SNE直接: 高次元ノイズに敏感。init='random'で特に不安定
   - PCA→t-SNE: PCAでノイズ除去後にt-SNEを適用

2. PCA前処理の効果
   - ノイズが強いほどPCA前処理の効果が大きい
   - 高次元（>100）ではPCA前処理がほぼ必須
   - van der Maaten原論文でも50次元へのPCA推奨

3. データ依存性
   - 「常に最良の手法」は存在しない
   - 線形分離可能なデータ → PCAで十分
   - 非線形構造 + 高次元ノイズ → PCA→t-SNEが有効

【実験パラメータの調整ポイント】

- noise_strength を上げると t-SNE単体の結果が悪化
- total_dim を上げると t-SNE単体が不安定に
- pca_dim は累積寄与率80-95%程度を目安
- perplexity は n_samples/3 程度（5-50の範囲）

【学生への説明のヒント】

「なぜPCA前処理が有効か？」
→ 次元の呪い（curse of dimensionality）
→ 高次元では全点間距離が似通う → t-SNEの確率計算が不安定
→ PCAでノイズ次元を除去 → 信号が際立つ → t-SNEが安定動作
    """)
    print("=" * 70)
    
    plt.show()
    
    return all_results


if __name__ == "__main__":
    main()