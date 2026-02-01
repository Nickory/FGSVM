from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from collections import Counter
from granular import FGSVM  # 假设你的 FGSVM 类在 granular.py 中

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
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average='binary')
    metrics = [accuracy, precision, recall, fpr, f1]
    return metrics

def getdata():
    train = np.load('dataset/train.npy')
    test = np.load('dataset/test.npy')
    
    # # 建议使用全量数据（去掉限制），或根据内存设置上限
    # # 如果内存不足，可保留 [:10000] 或更小
    X_train = train[:, :-1]           # 全量推荐
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    
    # 如果内存吃紧，可临时用部分数据测试
    # X_train = train[:100, :-1]
    # y_train = train[:100, -1]
    # X_test = test[:100, :-1]
    # y_test = test[:100, -1]
     # 在 getdata() 里面替换原来的切片方式
    # np.random.seed(42)           # 可复现
    # idx = np.random.permutation(len(train))[:20000]
    # X_train = train[idx, :-1]
    # y_train = train[idx, -1]

    # # 测试集也可以随机一点（可选）
    # idx_test = np.random.permutation(len(test))[:1000]
    # X_test  = test[idx_test, :-1]
    # y_test  = test[idx_test, -1]
    
    y_train = np.where(y_train == 0, -1, 1).astype(int)
    y_test = np.where(y_test == 0, -1, 1).astype(int)
    
    print(f"[DEBUG] 训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"[DEBUG] 训练集 y 分布: {Counter(y_train)}")
    print(f"[DEBUG] 测试集 y 分布: {Counter(y_test)}")
    print(set(y_test))
    
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = getdata()
    
    # 建议调参：gamma 调小，reference_n 增大，C 适当增大
    model = FGSVM(
        C=6.0,          # 增大惩罚
        kernel='rbf',
        degree=3,
        gamma=4.9,       # 强烈建议调小！原 4.9 太大了
        beta=0.829,
        tol=1e-3,
        max_iter=100     # 增加迭代次数
    )
    
    print("开始训练 FGSVM...")
    model.fit(X_train, y_train, reference_n=5)  # 建议增大参考点数
    
    pred = model.predict(X_test)
    
    # Debug 打印
    print("\n[DEBUG] 测试集预测值分布:", Counter(pred))
    print("[DEBUG] 正类预测数量 (1):", np.sum(pred == 1))
    print("[DEBUG] 负类预测数量 (-1):", np.sum(pred == -1))
    
    cm = confusion_matrix(y_test, pred)
    print("\n混淆矩阵:\n", cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    result = evaluation_para(y_test, pred)
    print("\n最终评估结果:", result)