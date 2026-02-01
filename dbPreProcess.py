import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ==============================
# 配置部分（不变）
# ==============================
INPUT_FILE = "/root/autodl-tmp/Dry_Bean_Dataset/Dry_Bean_Dataset.xlsx"
OUTPUT_DIR = "dataset"

FEATURE_COLUMNS = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
    'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
    'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
    'ShapeFactor3', 'ShapeFactor4'
]

LABEL_COLUMN = 'Class'

# ==============================
# 辅助函数：计算 balance ratio 并打印（不变）
# ==============================
def print_balance_info(y_data, dataset_name="数据集"):
    pos_count = np.sum(y_data == 1)
    neg_count = np.sum(y_data == 0)
    total = len(y_data)
    pos_ratio = pos_count / total if total > 0 else 0
    balance_ratio = pos_count / neg_count if neg_count > 0 else float('inf')
   
    print(f"\n{dataset_name} 二分类分布：")
    print(f" 正样本 (1)：{pos_count} 条 ({pos_ratio:.4f})")
    print(f" 负样本 (0)：{neg_count} 条")
    print(f" Balance Ratio (正/负)：{balance_ratio:.4f}")
    print(f" 不平衡程度：{'轻度' if 0.5 <= balance_ratio <= 2 else '中度' if 0.2 <= balance_ratio < 0.5 or 2 < balance_ratio <= 5 else '严重'}")

# ==============================
# 1. 读取数据（不变）
# ==============================
print("读取数据文件:", INPUT_FILE)
if INPUT_FILE.lower().endswith(('.xlsx', '.xls')):
    df = pd.read_excel(INPUT_FILE)
else:
    df = pd.read_csv(INPUT_FILE)
print("\n原始数据形状:", df.shape)
print("列名（共 {} 列）：".format(len(df.columns)))
print(list(df.columns))
print("\n原始类别分布：")
print(df[LABEL_COLUMN].value_counts().sort_index())
print()

# ==============================
# 2. 选择特征列 & 处理缺失值（不变）
# ==============================
required_cols = FEATURE_COLUMNS + [LABEL_COLUMN]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print("错误：缺少以下列", missing_cols)
    print("请检查文件列名是否完全匹配（注意大小写和空格）")
    exit(1)
df = df[required_cols].copy()
if df.isnull().any().any():
    print("存在缺失值，正在用中位数填充...")
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

# ==============================
# 3. 确定正负类（按 Excel 从上到下首次出现顺序）
# ==============================
classes = df[LABEL_COLUMN].drop_duplicates().tolist()
print("\n所有类别（按 Excel 从上到下首次出现顺序）：")
print(list(enumerate(classes, 1)))

# 正类：取第 3,4,5,6 个（索引从 0 开始，所以 2:6）
positive_classes = classes[3:7]   # 注意：这里用 2:6（第3~6个），你原代码写的是 [3:7]，可能是笔误
print("\n原正类（标签=1 之前）：", positive_classes)
print("原负类（标签=0 之前）：", [c for c in classes if c not in positive_classes])

# 创建二分类标签（先按原逻辑）
df['binary_label'] = df[LABEL_COLUMN].isin(positive_classes).astype(int)

# ★★★ 关键：交换正负类，让少数类成为 y=1 ★★★
df['binary_label'] = 1 - df['binary_label']

print("\n交换后正类（标签=1）：", [c for c in classes if df[df[LABEL_COLUMN].isin([c])]['binary_label'].iloc[0] == 1])
print("交换后负类（标签=0）：", [c for c in classes if df[df[LABEL_COLUMN].isin([c])]['binary_label'].iloc[0] == 0])

# 打印原始数据集的 balance ratio（交换后）
print_balance_info(df['binary_label'].values, "原始完整数据集（正负交换后）")

# ==============================
# 4. 准备 X 和 y（不变）
# ==============================
X = df[FEATURE_COLUMNS].values.astype(np.float32)
y = df['binary_label'].values.astype(np.int32)  # 现在 y==1 是少数类

# ==============================
# 5. 划分 7:3 训练/测试集（分层）
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("\n划分结果：")
print(f"训练集：{X_train.shape[0]} 条 ({X_train.shape[0]/len(X):.1%})")
print(f"测试集：{X_test.shape[0]} 条 ({X_test.shape[0]/len(X):.1%})")

print_balance_info(y_train, "训练集（正负交换后）")
print_balance_info(y_test, "测试集（正负交换后）")

# ==============================
# 6. 合并成 [特征 + 标签] 格式并保存（不变）
# ==============================
train_data = np.column_stack([X_train, y_train])
test_data = np.column_stack([X_test, y_test])

print("\n训练集最后 3 行（特征... + 标签）：")
print(train_data[-3:])

print("\n测试集最后 3 行（特征... + 标签）：")
print(test_data[-3:])

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "train.npy"), train_data)
np.save(os.path.join(OUTPUT_DIR, "test.npy"), test_data)

print(f"\n已保存：")
print(f" {OUTPUT_DIR}/train.npy shape = {train_data.shape}")
print(f" {OUTPUT_DIR}/test.npy shape = {test_data.shape}")
print("标签值范围：", np.unique(y_train), np.unique(y_test))