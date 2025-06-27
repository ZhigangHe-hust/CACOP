# 密度图自适应归一化

本项目基于 [PlantCount_CACViT](https://github.com/Meng-Shen/CACOP/tree/main/PlantCount_CACViT) 进行了模型改进，专注于密度图的自适应归一化，提升植物计数任务的准确性和泛化能力。

---

## 1. 数据集及预训练模型下载

- **预训练模型文件**：[点击下载](https://pan.quark.cn/s/e8b163f9bb74)
- **参考原始项目**：[PlantCount_CACViT](https://github.com/Meng-Shen/CACOP/tree/main/PlantCount_CACViT)

---

## 2. 环境配置

请参考 [PlantCount_CACViT](https://github.com/Meng-Shen/CACOP/tree/main/PlantCount_CACViT) 的环境配置文档进行环境搭建。

---

## 3. 数据集准备

本项目的主要改进集中在 loss 计算部分。  
如需使用本项目的改进版 loss，直接替换原有的 `train_val.py` 文件即可。

---

## 4. 训练方法

- 根据 batch size 设置随机 mask 保留概率 `p`，如 batch size=12 时，`p=0.85`
- 改进模块可调参数：
  - 前景与背景加权：`w_fg`、`w_bg`
  - 分段误差阈值：`tau`
  - 大误差放大倍数：`lambda_high`
  - Count Loss 权重：`alpha`

**训练命令示例：**
```bash
python train_val.py --resume $PATH_TO_BEST_MODEL_PTH --batch_size 12
```
- 训练过程中最佳模型将自动保存为 `checkpoint777.pth`

---

## 5. 测试方法

请根据实际路径调整模型参数及测试参数，运行以下命令进行测试：

```bash
python test.py
```

---

## 6. 结果与结论

- 默认配置下，训练 `epochs=53`，最佳模型权重如下：
- [最优模型权重及训练过程](https://pan.quark.cn/s/9b1c5d363750)

**在本项目植物计数数据集测试集上的最终结果：**

|   指标名称   |   数值    |
|:------------:|:---------:|
|     MAE      |   5.35    |
|    RMSE      |   9.85    |
|    rMAE      |   0.37    |
|   rRMSE      |   0.59    |
|     R²       |   0.91    |
|   FLOPs      | 449.86 G  |

---

## 7. 致谢

感谢 [PlantCount_CACViT](https://github.com/Meng-Shen/CACOP/tree/main/PlantCount_CACViT) 项目的开源贡献，为本项目的改进与创新提供了坚实的基础。

---

