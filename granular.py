import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

# Import cuML and CuPy for GPU acceleration
try:
    from cuml.svm import SVC
    print("cuML found. Running on GPU.")
except ImportError:
    from smo import SVC
    print("cuML not found. Running on CPU.")

class FGSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', tol=1e-3, max_iter=1000, sigma=0.5, beta=1.0):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.sigma = sigma
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.estimators_ = []
        self.reference_points_ = None

    def _granulate_func(self, refer, change, sigma=0.5, beta=1.0):
        array = 1 - np.abs(change - refer)
        if beta == 0:
            beta = 1e-6
        a = sigma - beta / 2.0
        b = sigma + beta / 2.0
        a = max(0, a)
        b = min(1, b)
        if a >= b:
            return np.where(array > a, 1.0, 0.0).astype(np.float32)
        conditions = [array <= a, (array > a) & (array < b)]
        choices = [0, (array - a) / (b - a)]
        result = np.select(conditions, choices, default=1.0)
        return result.astype(np.float32)

    def fit(self, X, y, reference_n):
        """
        Granulate the data using reference samples.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training vectors.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        reference_n : int
            The number of reference samples to take from X for granulating the dataset.
        """
        self.reference_points_ = X[0:reference_n]
        
        print("Granulating the training data...")
        granulated_X_3d = np.array([
            self._granulate_func(ref_point, X, self.sigma, self.beta)
            for ref_point in self.reference_points_
        ])
        print("Granulation finished, starting training.")
        
        # 初始化 estimators 列表
        self.estimators_ = []
        num_estimators = granulated_X_3d.shape[0]
        
        # 添加 tqdm 进度条：显示训练每个基分类器的进度
        print(f"开始训练 {num_estimators} 个基 SVM 模型（每个模型独立训练）...")
        for i in tqdm(range(num_estimators),
                      desc="Training base SVMs",
                      unit="model",
                      ncols=100,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
            
            # 选取第 i 个粒化视图
            X_train_granulated = granulated_X_3d[i, :, :]
            
            estimator = SVC(
                C=self.C,
                kernel=self.kernel,
                degree=self.degree,
                gamma=self.gamma
            )
            
            # 训练当前基分类器
            estimator.fit(X_train_granulated, y)
            
            # 保存
            self.estimators_.append(estimator)
        
        print("所有基 SVM 训练完成！")
        return self

    def predict(self, X):
        """
        Granulate and classify the input samples.
        """
        if not hasattr(self, 'estimators_') or not hasattr(self, 'reference_points_'):
            raise RuntimeError("The model has not been trained yet. Please call the 'fit' method first.")
        
        # Granulate the entire test set
        granulated_X_test_3d = np.array([
            self._granulate_func(ref_point, X, self.sigma, self.beta)
            for ref_point in self.reference_points_
        ])
        
        # Prediction
        decision_granules = []
        for i, estimator in enumerate(self.estimators_):
            X_test_granulated = granulated_X_test_3d[i, :, :]
            decision_granules.append(estimator.predict(X_test_granulated))
        
        decision_values = np.zeros(X.shape[0])
        for decision_value in decision_granules:
            decision_values += decision_value
        
        # Final classification based on the sign of the accumulated values
        y_pred = np.where(decision_values > 0, 1, -1)
        return y_pred