import numpy as np
import pandas as pd
from collections import Counter
import os

# Import cuML and CuPy for GPU acceleration
try:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors
    CUDNN_AVAILABLE = True
    print("cuML and CuPy found. Running on GPU.")
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    CUDNN_AVAILABLE = False
    print("cuML or CuPy not found. Falling back to scikit-learn on CPU.")

def save_to_csv(X, y, filename):
    """Saves the sampled data to a CSV file."""
    if hasattr(X, 'get'):
        X = X.get()
    if hasattr(y, 'get'):
        y = y.get()
    X = pd.DataFrame(X)
    y = pd.DataFrame(y, columns=['label'])
    data = pd.concat([X, y], axis=1)
    data.to_csv(filename, index=False)
    print(f"Saved sampled data to: {filename} (Samples: {len(y)})")

# --- 可视化辅助函数 (unchanged) ---
def plot_data(X, y, title, ax, highlight_indices=None, highlight_label=''):
    if hasattr(X, 'get'):
        X = X.get()
    if hasattr(y, 'get'):
        y = y.get()
    count = Counter(y)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='royalblue', marker='o', s=30, alpha=0.7,
               label=f'Majority (n={count.get(0, 0)})')
    if 1 in count:
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='darkorange', marker='^', s=40, alpha=0.9,
                   label=f'Minority (n={count.get(1, 0)})')
    if highlight_indices is not None and len(highlight_indices) > 0:
        ax.scatter(X[highlight_indices, 0], X[highlight_indices, 1],
                   s=150, facecolors='none', edgecolors='red', linewidths=2,
                   label=highlight_label)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

# --- Class Implementation of the Hybrid Sampler ---
class HybridSampler:
    def __init__(self, k_neighbors=5, undersampling_ratio=0.5, random_state=None):
        self.k_neighbors = k_neighbors
        self.undersampling_ratio = undersampling_ratio
        if CUDNN_AVAILABLE:
            self.random_state_ = cp.random.RandomState(seed=random_state)
        else:
            self.random_state_ = np.random.RandomState(seed=random_state)
        self.noise_indices_ = None
        self.X_clean_ = None
        self.y_clean_ = None
        self.synthetic_samples_ = None
        self.xp = cp if CUDNN_AVAILABLE else np

    def fit_resample(self, X, y):
        print(f"\n[DEBUG] 输入数据形状: X={X.shape}, y={y.shape}")
        print(f"[DEBUG] 输入类别分布: {Counter(y)} (少数类 y=1 数量: {np.sum(y==1)})")

        X_gpu = self.xp.asarray(X)
        y_gpu = self.xp.asarray(y)

        self._remove_noise(X_gpu, y_gpu)

        if 1 not in Counter(self.y_clean_.get() if CUDNN_AVAILABLE else self.y_clean_):
            print("\nWarning: No minority samples left after noise removal. Returning cleaned data.")
            return (self.X_clean_.get(), self.y_clean_.get()) if CUDNN_AVAILABLE else (self.X_clean_, self.y_clean_)

        X_resampled_gpu, y_resampled_gpu = self._perform_weighted_sampling()

        print(f"\n[DEBUG] 最终采样后类别分布: {Counter(y_resampled_gpu.get() if CUDNN_AVAILABLE else y_resampled_gpu)}")
        print(f"[DEBUG] 采样后数据形状: X={X_resampled_gpu.shape}, y={y_resampled_gpu.shape}")

        if CUDNN_AVAILABLE:
            return X_resampled_gpu.get(), y_resampled_gpu.get()
        return X_resampled_gpu, y_resampled_gpu

    def _remove_noise(self, X, y):
        print("Step 1: Identifying and removing noise...")
        minority_indices = self.xp.where(y == 1)[0]
        print(f"[DEBUG] 少数类样本数量: {len(minority_indices)}")

        knn_global = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(X)
        _, neighbor_indices = knn_global.kneighbors(X[minority_indices])
        neighbor_indices = neighbor_indices[:, 1:]  # Exclude self

        neighbor_labels = y[neighbor_indices]
        n_majority_neighbors = self.xp.sum(neighbor_labels == 0, axis=1)
        noise_mask = (n_majority_neighbors == self.k_neighbors)

        self.noise_indices_ = minority_indices[noise_mask]
        print(f"Identified and removed {len(self.noise_indices_)} minority outliers.")
        if len(self.noise_indices_) > 0:
            print(f"[DEBUG] 噪声样本索引示例（前5个）: {self.noise_indices_[:5]}")

        all_indices = self.xp.arange(len(X))
        indices_to_keep = self.xp.setdiff1d(all_indices, self.noise_indices_)

        self.X_clean_ = X[indices_to_keep]
        self.y_clean_ = y[indices_to_keep]

        print(f"[DEBUG] 清洗后数据形状: X={self.X_clean_.shape}, y={self.y_clean_.shape}")
        print(f"Cleaned dataset: {Counter(self.y_clean_.get() if CUDNN_AVAILABLE else self.y_clean_)}")

    def _perform_weighted_sampling(self):
        print("\nStep 2: Performing weighted sampling...")
        minority_indices_clean = self.xp.where(self.y_clean_ == 1)[0]
        majority_indices_clean = self.xp.where(self.y_clean_ == 0)[0]

        print(f"[DEBUG] 清洗后少数类数量: {len(minority_indices_clean)}")
        print(f"[DEBUG] 清洗后多数类数量: {len(majority_indices_clean)}")

        X_minority_clean = self.X_clean_[minority_indices_clean]
        X_majority_clean = self.X_clean_[majority_indices_clean]

        # Weighted Undersampling of Majority Class
        knn_minority = NearestNeighbors(n_neighbors=1).fit(X_minority_clean)
        distances, _ = knn_minority.kneighbors(X_majority_clean)
        safety_scores = distances.ravel()

        print(f"[DEBUG] Safety scores (到最近少数类的距离) - min: {safety_scores.min():.4f}, "
              f"max: {safety_scores.max():.4f}, mean: {safety_scores.mean():.4f}")

        n_majority_to_remove = int(len(majority_indices_clean) * self.undersampling_ratio)
        print(f"[DEBUG] 将移除多数类样本数量: {n_majority_to_remove} "
              f"({self.undersampling_ratio*100:.1f}% of {len(majority_indices_clean)})")

        sorted_majority_indices = majority_indices_clean[self.xp.argsort(-safety_scores)]
        majority_indices_to_remove = sorted_majority_indices[:n_majority_to_remove]
        majority_indices_kept = self.xp.setdiff1d(majority_indices_clean, majority_indices_to_remove)

        X_majority_resampled = self.X_clean_[majority_indices_kept]
        y_majority_resampled = self.y_clean_[majority_indices_kept]

        print(f"[DEBUG] 欠采样后多数类剩余数量: {len(X_majority_resampled)}")

        # Weighted Oversampling of Minority Class (SMOTE-like)
        knn_clean = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(self.X_clean_)
        _, clean_neighbors_indices = knn_clean.kneighbors(X_minority_clean)
        clean_neighbors_indices = clean_neighbors_indices[:, 1:]

        clean_neighbor_labels = self.y_clean_[clean_neighbors_indices]
        danger_scores = self.xp.sum(clean_neighbor_labels == 0, axis=1)

        print(f"[DEBUG] Danger scores (邻居中多数类数量) - min: {danger_scores.min()}, "
              f"max: {danger_scores.max()}, mean: {danger_scores.mean():.4f}, "
              f"sum: {danger_scores.sum()}")

        n_minority_target = len(X_majority_resampled)
        n_samples_to_generate = int(len(minority_indices_clean) * 1)
        print(f"[DEBUG] 目标少数类数量（匹配欠采样后多数类）: {n_minority_target}")
        print(f"[DEBUG] 将生成合成少数类样本数量: {n_samples_to_generate} "
              f"({0.1*len(minority_indices_clean):.0f} 的 10%)")

        synthetic_samples_list = []
        if n_samples_to_generate > 0 and danger_scores.sum() > 0:
            selection_probabilities = danger_scores / danger_scores.sum()
            smote_k = min(self.k_neighbors + 1, len(X_minority_clean))
            knn_smote = NearestNeighbors(n_neighbors=smote_k).fit(X_minority_clean)

            for i in range(n_samples_to_generate):
                base_idx_arr = self.random_state_.choice(len(minority_indices_clean), size=1, p=selection_probabilities)
                base_idx = base_idx_arr.item()  # 修复警告

                base_sample = X_minority_clean[base_idx]

                _, smote_neighbor_indices = knn_smote.kneighbors(base_sample.reshape(1, -1))
                smote_neighbor_indices = smote_neighbor_indices[0, 1:]

                if len(smote_neighbor_indices) == 0:
                    continue

                neighbor_idx_arr = self.random_state_.choice(smote_neighbor_indices, size=1)
                neighbor_idx = neighbor_idx_arr.item()  # 修复警告

                neighbor_sample = X_minority_clean[neighbor_idx]

                diff = neighbor_sample - base_sample
                synthetic_sample = base_sample + self.random_state_.rand() * diff
                synthetic_samples_list.append(synthetic_sample)

                if i % 50 == 0 and i > 0:
                    print(f"[DEBUG] 已生成 {i} / {n_samples_to_generate} 个合成样本...")

        print(f"[DEBUG] 实际生成合成样本数量: {len(synthetic_samples_list)}")

        self.synthetic_samples_ = self.xp.array(synthetic_samples_list).reshape(-1, X_majority_resampled.shape[1])
        X_resampled = self.xp.vstack([X_majority_resampled, X_minority_clean, self.synthetic_samples_])
        y_resampled = self.xp.hstack([
            y_majority_resampled,
            self.xp.ones(len(X_minority_clean) + len(self.synthetic_samples_))
        ])

        return X_resampled, y_resampled

def deal_data(input_file='dataset/train.npy', output_file='dataset/sam_train.npy'):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件 {input_file} 不存在")

    print(f"[DEBUG] 加载原始训练数据: {input_file}")
    train = np.load(input_file)
    X = train[:, :-1]
    y = train[:, -1]
    y = np.where(y == 0, 0, 1).astype(int)

    print(f"[DEBUG] 原始 X 形状: {X.shape}, y 形状: {y.shape}")
    print(f"[DEBUG] 原始 y 分布: {Counter(y)} (少数类 y=1: {np.sum(y==1)})")

    sampler = HybridSampler(k_neighbors=5, undersampling_ratio=0.3, random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    X_df = pd.DataFrame(X_resampled.get() if hasattr(X_resampled, 'get') else X_resampled)
    y_df = pd.DataFrame(y_resampled.get() if hasattr(y_resampled, 'get') else y_resampled, columns=['label'])
    data = pd.concat([X_df, y_df], axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, data.values)

    print(f"\n数据处理完成，结果已保存到 {output_file}")
    print(f"[DEBUG] 输出文件形状: {data.shape}")
    print(f"[DEBUG] 输出 y 分布: {Counter(data['label'])}")

# --- Main Logic ---
if __name__ == '__main__':
    deal_data()