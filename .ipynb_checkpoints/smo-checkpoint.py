import numpy as np

class SVC:
    """
    使用 SMO (Sequential Minimal Optimization) 算法实现的 SVM 分类器。

    参数:
    ----------
    C : float, default=1.0
        正则化参数。

    kernel : {'linear', 'poly', 'rbf'}, default='rbf'
        指定要使用的内核类型。

    degree : int, default=3
        当 kernel='poly' 时的多项式次数。

    gamma : float, default=1.0
        当 kernel='rbf' 时的核系数。

    tol : float, default=1e-3
        KKT 条件的容忍度。

    max_iter : int, default=100
        算法的最大迭代次数。
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=1.0, tol=1e-3, max_iter=100):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None
        self.K = None
        self.E_cache = None  # 误差缓存

    # --- 内核函数 ---
    def _linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def _poly_kernel(self, x1, x2):
        return (np.dot(x1, x2) + 1) ** self.degree

    def _rbf_kernel(self, x1, x2):
        # RBF 核函数, gamma = 1 / (2 * sigma^2)
        return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))

    def _get_kernel_function(self):
        """根据 self.kernel 返回对应的核函数"""
        if self.kernel == 'linear':
            return self._linear_kernel
        elif self.kernel == 'poly':
            return self._poly_kernel
        elif self.kernel == 'rbf':
            return self._rbf_kernel
        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel}")

    def _get_kernel_matrix(self, X):
        """预计算核矩阵 K"""
        n_samples, _ = X.shape
        K = np.zeros((n_samples, n_samples))
        kernel_func = self._get_kernel_function()
        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = kernel_func(X[i], X[j])
                K[j, i] = K[i, j]
        return K

    # --- SMO 核心辅助函数 ---
    def _g(self, i):
        """计算决策函数 g(x_i) 的值"""
        # (sum_{j=1 to N} alpha_j * y_j * K(x_i, x_j)) + b
        return np.sum(self.alpha * self.y * self.K[:, i]) + self.b

    def _E(self, i):
        """计算误差 E_i = g(x_i) - y_i"""
        return self._g(i) - self.y[i]

    def _check_KKT(self, i):
        """
        检查样本 i 是否违反 KKT 条件。
        KKT 条件:
        1. alpha_i = 0  ==>  y_i * g(x_i) >= 1
        2. 0 < alpha_i < C ==> y_i * g(x_i) = 1
        3. alpha_i = C  ==>  y_i * g(x_i) <= 1

        违反 KKT 条件的情况 (在 tol 容忍度下):
        - y_i * E_i < -tol  and alpha_i < C  (对应 y_i*g(x_i) < 1)
        - y_i * E_i > tol   and alpha_i > 0  (对应 y_i*g(x_i) > 1)
        """
        y_g = self.y[i] * self._g(i)

        # 对应条件 1 和 3, y_i*g(x_i) 需要 >= 1，如果小于 1-tol 则违反
        if self.alpha[i] < self.C and y_g < 1 - self.tol:
            return True
        # 对应条件 2 和 3, y_i*g(x_i) 需要 <= 1，如果大于 1+tol 则违反
        elif self.alpha[i] > 0 and y_g > 1 + self.tol:
            return True

        return False

    def _select_j(self, i, E_i):
        """启发式选择第二个变量 j，使其 |E_i - E_j| 最大"""
        max_delta_E = 0
        j_selected = -1
        E_j_selected = 0

        # 将所有非边界 alpha 对应的 E 放入缓存的有效列表
        valid_E_cache_list = np.nonzero((self.E_cache != 0))[0]

        if len(valid_E_cache_list) > 1:
            # 优先从缓存中寻找
            for j in valid_E_cache_list:
                if i == j: continue
                E_j = self._E(j)
                delta_E = abs(E_i - E_j)
                if delta_E > max_delta_E:
                    max_delta_E = delta_E
                    j_selected = j
                    E_j_selected = E_j
            return j_selected, E_j_selected
        else:
            # 如果缓存中没有，则随机选择一个 j
            j = i
            while j == i:
                j = np.random.randint(0, len(self.alpha))
            E_j = self._E(j)
            return j, E_j

    def update(self, i, j):
        """优化 alpha_i 和 alpha_j"""
        if i == j:
            return False

        # 计算 E_i 和 E_j
        E_i = self._E(i)
        E_j = self._E(j)

        # 保存旧值
        alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

        # 计算 alpha_j 的边界 L 和 H
        if self.y[i] != self.y[j]:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_j_old + alpha_i_old - self.C)
            H = min(self.C, alpha_j_old + alpha_i_old)

        if L == H:
            return False

        # 计算 eta = K_ii + K_jj - 2*K_ij
        eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
        if eta <= 0:
            return False

        # 计算未经剪辑的 alpha_j_new
        alpha_j_new_unc = alpha_j_old + self.y[j] * (E_i - E_j) / eta

        # 剪辑 alpha_j_new
        alpha_j_new = np.clip(alpha_j_new_unc, L, H)

        # 如果变化太小，则忽略
        if abs(alpha_j_new - alpha_j_old) < 1e-5:
            return False

        # 计算 alpha_i_new
        alpha_i_new = alpha_i_old + self.y[i] * self.y[j] * (alpha_j_old - alpha_j_new)

        # 更新 b
        b1 = self.b - E_i - self.y[i] * (alpha_i_new - alpha_i_old) * self.K[i, i] - \
             self.y[j] * (alpha_j_new - alpha_j_old) * self.K[i, j]
        b2 = self.b - E_j - self.y[i] * (alpha_i_new - alpha_i_old) * self.K[i, j] - \
             self.y[j] * (alpha_j_new - alpha_j_old) * self.K[j, j]

        if 0 < alpha_i_new < self.C:
            self.b = b1
        elif 0 < alpha_j_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        # 更新 alpha 值
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new

        # 更新误差缓存
        self.E_cache[i] = self._E(i)
        self.E_cache[j] = self._E(j)

        return True

    # --- 主要训练和预测函数 ---
    def fit(self, X, y):
        """训练 SVM 模型"""
        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # 预计算核矩阵
        self.K = self._get_kernel_matrix(X)
        self.E_cache = np.array([self._E(i) for i in range(n_samples)])

        iter_num = 0
        entire_set = True
        alpha_changed_count = 0

        while iter_num < self.max_iter and (alpha_changed_count > 0 or entire_set):
            alpha_changed_count = 0

            if entire_set:
                # 遍历整个数据集
                for i in range(n_samples):
                    if self._check_KKT(i):
                        E_i = self.E_cache[i]
                        j, E_j = self._select_j(i, E_i)
                        if self.update(i, j):
                            alpha_changed_count += 1
            else:
                # 遍历所有非边界样本 (0 < alpha < C)
                non_bound_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
                for i in non_bound_indices:
                    if self._check_KKT(i):
                        E_i = self.E_cache[i]
                        j, E_j = self._select_j(i, E_i)
                        if self.update(i, j):
                            alpha_changed_count += 1

            iter_num += 1

            # 交替在整个数据集和非边界样本集上进行优化
            if entire_set:
                entire_set = False
            elif alpha_changed_count == 0:
                entire_set = True  # 如果非边界样本没有更新，则尝试整个数据集

    def predict(self, X_test):
        """对新数据进行预测"""
        if self.alpha is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        kernel_func = self._get_kernel_function()
        preds = []
        for x in X_test:
            # 计算 g(x) = sum(alpha_i * y_i * K(x_i, x)) + b
            pred = np.sum([self.alpha[i] * self.y[i] * kernel_func(self.X[i], x) for i in range(len(self.X))]) + self.b
            preds.append(np.sign(pred))
        return np.array(preds)

    def score(self, X, y):
        """计算模型在给定数据上的准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)