from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from granular import FGSVM

def evaluation_para(y_true, y_pred):
    """
    Calculates and returns evaluation metrics for a machine learning model.

    Args:
        y_true (list): A list of ground truth labels.
        y_pred (list): A list of predicted labels from the model.

    Returns:
        list: A list containing the evaluation metrics in the following order:
              [accuracy, precision, recall, fpr, f1].
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision
    # 'zero_division=0' handles cases where there are no positive predictions.
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)

    # Calculate recall
    recall = recall_score(y_true, y_pred, average='binary')

    # Calculate False Positive Rate (FPR)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average='binary')

    # Store the evaluation metrics in a list and return it.
    metrics = [
        accuracy,
        precision,
        recall,
        fpr,
        f1
    ]
    return metrics

def getdata():
    train = np.load('dataset/train.npy')
    test = np.load('dataset/test.npy')

    X_train = train[:100, :-1]
    y_train = train[:100, -1]
    X_test = test[:100, :-1]
    y_test = test[:100, -1]

    y_train = np.where(y_train == 0, -1, 1).astype(int)
    y_test = np.where(y_test == 0, -1, 1).astype(int)

    print(set(y_test))
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = getdata()
    model = FGSVM(C=6.0, kernel='rbf', degree=3, gamma=4.9, beta=0.829, tol=1e-3, max_iter=100)
    model.fit(X_train, y_train, 5)
    pred = model.predict(X_test)
    result = evaluation_para(y_test, pred)
    print(result)
