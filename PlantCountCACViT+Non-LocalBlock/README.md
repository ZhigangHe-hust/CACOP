<p align="center">
    <h1 align="center">PlantCountCACViT + Non-Local Block</h1>
</p>

# 1.简介

我们在视觉Transformer的编码器间插入非局部模块，加强对场景中长距离依赖与全局空间关系的建模能力，用于应对植株簇生、叶片重叠时局部特征不完整的情况。实验结果表明，改进后的模型在一定程度上降低了平均绝对误差（MAE），但在平均平方误差（MSE）上略有退步，我们认为是可能是由于全局建模引入一定噪声所致。

# 2.环境配置及模型运行等

请参考：<a href="https://github.com/ZhigangHe-hust/CACOP/blob/main/PlantCount_CACViT/readme.md" title="环境配置及模型运行">环境配置及模型运行</a>

# 3.实验结果

<a href="https://pan.quark.cn/s/659ce7d2ccae" title="最优模型权重">最优模型权重</a>，提取码：**Si8x**

引入Non-Local模块前后的 CACViT 模型在 PlantCount 数据集测试集上的性能评估结果如下：

![image text](https://github.com/ZhigangHe-hust/CACOP/blob/main/PlantCountCACViT%2BNon-LocalBlock/figs/fig1.png)

