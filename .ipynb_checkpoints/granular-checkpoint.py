import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

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
        # # --- Initial checks and setup ---
        # self.reference_points_ = np.asarray(reference_points)
        # if self.reference_points_.ndim == 1:
        #     self.reference_points_ = self.reference_points_.reshape(1, -1)
        # if self.reference_points_.shape[1] != X.shape[1]:
        #     raise ValueError("The feature dimensions of reference points and training data must match.")

        # Granulate the entire training set
        # Efficiently build a 3D array using list comprehension and np.array()
        print("Granulating the training data...")
        granulated_X_3d = np.array([
            self._granulate_func(ref_point, X, self.sigma, self.beta)
            for ref_point in self.reference_points_
        ])
        print("Granulation finished, starting training.")

        # Train
        self.estimators_ = []
        num_estimators = granulated_X_3d.shape[0]
        for i in range(num_estimators):
            # Select the i-th granulated training set (a 2D "slice") from the 3D array
            X_train_granulated = granulated_X_3d[i, :, :]
            estimator = SVC(
                C=self.C, kernel=self.kernel, degree=self.degree,
                gamma=self.gamma
            )
            estimator.fit(X_train_granulated, y)
            self.estimators_.append(estimator)
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
            # decision_granules
            decision_granules.append(estimator.predict(X_test_granulated))

        decision_values = np.zeros(X.shape[0])
        for decision_value in decision_granules:
            decision_values += decision_value

        # Final classification based on the sign of the accumulated values
        y_pred = np.where(decision_values > 0, 1, -1)
        return y_pred