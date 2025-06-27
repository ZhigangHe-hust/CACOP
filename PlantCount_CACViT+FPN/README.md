# PlantCountCACViT+FPN

### 一、数据集及预训练模型文件下载

FSC147数据集下载：参考<a href="https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master" title="Learning To Count Everything">Learning To Count Everything</a>

本项目的植物计数数据集下载：<a href="https://pan.quark.cn/s/76cec041ff98"
title="PlantCountDataset">PlantCountDataset</a>，提取码：8Gnp

本项目在CACViT的最优模型的基础上进行预训练，预训练模型文件的下载可以参考<a href="https://github.com/Xu3XiWang/CACViT-AAAI24" title="CACViT">CACViT</a>

### 二、环境配置

**建议Python版本和显卡型号：**

Python 3.8.18，NVIDIA GeForce RTX 3090

**安装依赖：**

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/
whl/torch_stable.html
```

```
pip install -r requirements.txt
```

### 三、任务实现过程

#### 1.模型训练

```bash
python train_val.py
```

**断点续训：**

根据实际情况确定你恢复训练的模型权重文件路径和start_epoch

```
python train_val.py --resume ./output_dir/checkpoint-30.pth --start_epoch 30
```

#### 2.模型测试

[21:26:53.734467] Averaged stats: 
[21:26:53.734531] MAE: 5.13,RMSE: 8.32,rMAE: 0.37,rRMSE: 0.51,R2: 0.95
[21:26:53.734548] 
FLOPs Summary:
[21:26:53.734572] Total FLOPs for all samples: 46.2823 TFLOPs
[21:26:53.734593] Average FLOPs per sample: 306.5056 GFLOPs
[21:26:53.734653] Min sample FLOPs: 120.9996 GFLOPs
[21:26:53.734679] Max sample FLOPs: 1028.4965 GFLOPs
[21:26:53.735247] Median sample FLOPs: 302.4990 GFLOPs
[21:26:53.735278] Testing time 0:00:38

### 四、结论

<a href="https://pan.quark.cn/s/2ff6e9ef361d" title="最优模型权重">最优模型权重</a>，提取码：YCev

最优模型权重在本项目的植物计数数据集上的测试集上的测试结果为：

|         指标名称         |    指标数值     |
| :----------------------: | :-------------: |
|           MAE            |      5.13       |
|           RMSE           |      8.32       |
|           rMAE           |      0.37       |
|          rRMSE           |      0.51       |
|            R2            |      0.95       |
| Average FLOPs per sample | 306.5056 GFLOPs |
|       Testing time       |       38s       |
