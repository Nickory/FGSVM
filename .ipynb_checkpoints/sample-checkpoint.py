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
    # Ensure data is on the CPU (as NumPy arrays) for pandas
    if hasattr(X, 'get'):
        X = X.get()
    if hasattr(y, 'get'):
        y = y.get()

    X = pd.DataFrame(X)
    y = pd.DataFrame(y, columns=['label'])
    data = pd.concat([X, y], axis=1)
    data.to_csv(filename, index=False)
    print(f"Saved sampled data to: {filename} (Samples: {len(y)})")


# --- 可视化辅助函数 ( unchanged ) ---
def plot_data(X, y, title, ax, highlight_indices=None, highlight_label=''):
    """Helper function to plot the 2D dataset."""
    # Ensure data is on the CPU for plotting
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
    """
    A three-step hybrid sampling method for imbalanced datasets, optimized with cuML.

    The process involves:
    1. Noise Removal: Identifies and removes minority class samples that are likely noise.
    2. Weighted Undersampling: Removes majority class samples that are "safe" (furthest from minority samples).
    3. Weighted Oversampling (SMOTE-like): Generates synthetic minority samples in "danger" zones.

    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest neighbors for all algorithm steps.
    undersampling_ratio : float, default=0.5
        Proportion of majority class samples to remove.
    random_state : int, default=None
        Seed for the random number generator.
    """

    def __init__(self, k_neighbors=5, undersampling_ratio=0.5, random_state=None):
        self.k_neighbors = k_neighbors
        self.undersampling_ratio = undersampling_ratio
        # Use CuPy's random state if on GPU, otherwise NumPy's
        if CUDNN_AVAILABLE:
            self.random_state_ = cp.random.RandomState(seed=random_state)
        else:
            self.random_state_ = np.random.RandomState(seed=random_state)

        # Attributes for intermediate results
        self.noise_indices_ = None
        self.X_clean_ = None
        self.y_clean_ = None
        self.synthetic_samples_ = None
        # Use cupy or numpy based on availability
        self.xp = cp if CUDNN_AVAILABLE else np

    def fit_resample(self, X, y):
        """
        Resamples the dataset X and y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix.
        y : array-like of shape (n_samples,)
            The target vector.

        Returns
        -------
        X_resampled, y_resampled : ndarray
            The resampled data and labels as NumPy arrays.
        """
        # Convert data to CuPy arrays for GPU processing if available
        X_gpu = self.xp.asarray(X)
        y_gpu = self.xp.asarray(y)

        # Step 1: Identify and Remove Outlier Minority Points (Noise)
        self._remove_noise(X_gpu, y_gpu)

        if 1 not in Counter(self.y_clean_.get() if CUDNN_AVAILABLE else self.y_clean_):
            print("\nWarning: No minority samples left after noise removal. Returning cleaned data.")
            return (self.X_clean_.get(), self.y_clean_.get()) if CUDNN_AVAILABLE else (self.X_clean_, self.y_clean_)

        # Step 2: Perform Weighted Undersampling and Oversampling
        X_resampled_gpu, y_resampled_gpu = self._perform_weighted_sampling()

        print(f"\nFinal balanced dataset: {Counter(y_resampled_gpu.get() if CUDNN_AVAILABLE else y_resampled_gpu)}")

        # Return NumPy arrays for compatibility with other libraries
        if CUDNN_AVAILABLE:
            return X_resampled_gpu.get(), y_resampled_gpu.get()
        return X_resampled_gpu, y_resampled_gpu

    def _remove_noise(self, X, y):
        """Identifies and removes noisy minority samples using the assigned backend (CPU/GPU)."""
        print("Step 1: Identifying and removing noise...")

        minority_indices = self.xp.where(y == 1)[0]

        # Fit KNN on the entire dataset to find neighbors
        knn_global = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(X)
        _, neighbor_indices = knn_global.kneighbors(X[minority_indices])
        neighbor_indices = neighbor_indices[:, 1:]  # Exclude self
        neighbor_labels = y[neighbor_indices]

        # A minority sample is noise if all K neighbors are from the majority class
        n_majority_neighbors = self.xp.sum(neighbor_labels == 0, axis=1)
        noise_mask = (n_majority_neighbors == self.k_neighbors)
        self.noise_indices_ = minority_indices[noise_mask]

        # Create the "clean" dataset by removing identified noise
        all_indices = self.xp.arange(len(X))
        indices_to_keep = self.xp.setdiff1d(all_indices, self.noise_indices_)
        self.X_clean_ = X[indices_to_keep]
        self.y_clean_ = y[indices_to_keep]

        print(f"Identified and removed {len(self.noise_indices_)} minority outliers.")
        print(f"Cleaned dataset: {Counter(self.y_clean_.get() if CUDNN_AVAILABLE else self.y_clean_)}")

    def _perform_weighted_sampling(self):
        """Executes weighted undersampling and oversampling on the clean dataset."""
        print("\nStep 2: Performing weighted sampling...")

        minority_indices_clean = self.xp.where(self.y_clean_ == 1)[0]
        majority_indices_clean = self.xp.where(self.y_clean_ == 0)[0]
        X_minority_clean = self.X_clean_[minority_indices_clean]
        X_majority_clean = self.X_clean_[majority_indices_clean]

        # --- Weighted Undersampling of Majority Class ---
        # Fit k-NN once on minority data to get distances for majority points
        knn_minority = NearestNeighbors(n_neighbors=1).fit(X_minority_clean)
        distances, _ = knn_minority.kneighbors(X_majority_clean)
        safety_scores = distances.ravel()

        n_majority_to_remove = int(len(majority_indices_clean) * self.undersampling_ratio)
        sorted_majority_indices = majority_indices_clean[self.xp.argsort(-safety_scores)]
        majority_indices_to_remove = sorted_majority_indices[:n_majority_to_remove]

        majority_indices_kept = self.xp.setdiff1d(majority_indices_clean, majority_indices_to_remove)
        X_majority_resampled = self.X_clean_[majority_indices_kept]
        y_majority_resampled = self.y_clean_[majority_indices_kept]

        # --- Weighted Oversampling of Minority Class (SMOTE-like) ---
        # Calculate "danger" scores for minority points
        knn_clean = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(self.X_clean_)
        _, clean_neighbors_indices = knn_clean.kneighbors(X_minority_clean)
        clean_neighbors_indices = clean_neighbors_indices[:, 1:]  # Exclude self
        clean_neighbor_labels = self.y_clean_[clean_neighbors_indices]
        danger_scores = self.xp.sum(clean_neighbor_labels == 0, axis=1)

        n_minority_target = len(X_majority_resampled)
        n_samples_to_generate = int(len(minority_indices_clean)*0.1)

        synthetic_samples_list = []  # Use a list to accumulate
        if n_samples_to_generate > 0 and danger_scores.sum() > 0:
            selection_probabilities = danger_scores / danger_scores.sum()

            # Re-use knn_minority model fit earlier, but with more neighbors for SMOTE
            smote_k = min(self.k_neighbors + 1, len(X_minority_clean))
            knn_smote = NearestNeighbors(n_neighbors=smote_k).fit(X_minority_clean)

            for _ in range(n_samples_to_generate):
                base_idx_arr = self.random_state_.choice(len(minority_indices_clean),size = 1, p=selection_probabilities)
                base_idx = int(base_idx_arr)  # Ensure index is an integer
                base_sample = X_minority_clean[base_idx]

                _, smote_neighbor_indices = knn_smote.kneighbors(base_sample.reshape(1, -1))
                smote_neighbor_indices = smote_neighbor_indices[0, 1:]  # Exclude self

                if len(smote_neighbor_indices) == 0: continue  # Skip if no neighbors

                neighbor_idx = self.random_state_.choice(smote_neighbor_indices,size = 1)
                neighbor_sample = X_minority_clean[int(neighbor_idx)]

                # Generate synthetic sample
                diff = neighbor_sample - base_sample
                synthetic_sample = base_sample + self.random_state_.rand() * diff
                synthetic_samples_list.append(synthetic_sample)

        # Convert list to array and combine final dataset
        self.synthetic_samples_ = self.xp.array(synthetic_samples_list).reshape(-1, X_majority_resampled.shape[1])

        X_resampled = self.xp.vstack([X_majority_resampled, X_minority_clean, self.synthetic_samples_])
        y_resampled = self.xp.hstack([
            y_majority_resampled,
            self.xp.ones(len(X_minority_clean) + len(self.synthetic_samples_))
        ])

        return X_resampled, y_resampled


def deal_data(input_file='dataset/train.npy', output_file='dataset/sam_train.npy'):
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件 {input_file} 不存在")

    # 检查输出文件是否已存在
    if os.path.exists(output_file):
        print(f"输出文件 {output_file} 已存在，跳过处理")
        return

    # 加载并处理原始数据
    train = np.load(input_file)
    X = train[:, :-1]
    y = train[:, -1]

    y = np.where(y == 0, 0, 1).astype(int)

    sampler = HybridSampler(k_neighbors=5, undersampling_ratio=0.3, random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    X = pd.DataFrame(X_resampled)
    y = pd.DataFrame(y_resampled)
    data = pd.concat([X, y], axis=1).astype(np.float32)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存处理后的数据
    np.save(output_file, data.values)
    print(f"数据处理完成，结果已保存到 {output_file}")

# --- Main Logic ---
if __name__ == '__main__':
    # From CSV file (assuming the last column is the label)
    # Note: Ensure the CSV does not contain a header row, or adjust pd.read_csv accordingly
    deal_data()
