```markdown
# FGSVM 项目代码使用说明

基于论文复现的 **FGSVM** 算法，主要针对不平衡及多分类转二分类场景的 SVM 改进。

## 主要文件

| 文件          | 说明                                      |
|---------------|-------------------------------------------|
| `granular.py` | FGSVM 核心结构（细粒度/颗粒化逻辑）       |
| `sain.py`     | 主程序入口（训练、预测、评估）            |
| `sample.py`   | 不平衡数据采样处理                        |
| `smo.py`      | SMO 算法实现的 SVM（支持 cuML GPU 加速）  |

## 推荐使用流程（以 Dry Bean 数据集为例）

1. 准备原始数据：`Dry_Bean_Dataset.xlsx`
2. 运行预处理脚本（例如 `dbPreProcess.py`）：
   - 读取 Excel
   - 多分类 → 二分类（根据论文指定 positive 类，标签建议为 ±1）
   - 按类别首次出现顺序处理
   - 划分 train/test（建议 7:3 或 8:2，也可 5 折）
   - 保存为 `train.npy` / `test.npy`（最后一列为 label）
3. （可选）严重不平衡时，使用 `sample.py` 生成采样后的 `sam_train.npy`
4. 运行主程序：
   ```bash
   python main.py
   ```
   - 读取 `train.npy` / `test.npy`（或 `sam_train.npy`）
   - 特征标准化
   - 训练 FGSVM
   - 测试评估

## 重要注意

- **GPU 加速**：需安装 CUDA + cuML，否则自动使用纯 Python SMO（较慢）
- **超参数**：主要在 `main.py` 中修改（C、gamma、颗粒阈值等）
- **常见问题**：
  - 标签非 ±1 → 检查二分类转换
  - 数据 shape 不匹配 → 确认 npy 格式
  - 正负样本比例异常 → 核对 positive 类定义顺序

完成预处理 → 直接运行 `main.py` 即可开始实验。