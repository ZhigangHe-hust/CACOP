# PlantCountCACViT

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

> GFLOPs (Giga-FLOPs) = 10^9 FLOPs, TFLOPs (Tera-FLOPs) = 10^12 FLOPs

（1）记录预训练模型在PlantCount测试集上的测试结果

```bash
python test.py
```

[19:30:36.828639] Averaged stats: 
[19:30:36.828678] **MAE:11.81,RMSE:21.70,rMAE: 1.02,rRMSE: 1.31,R2: 0.59**
[19:30:36.828688] 
FLOPs Summary:
[19:30:36.828703] Total FLOPs for all samples: 67.9290 TFLOPs
[19:30:36.828715] **Average FLOPs per sample:** **449.8607 GFLOPs**
[19:30:36.828739] Min sample FLOPs: 177.5921 GFLOPs
[19:30:36.828750] Max sample FLOPs: 1509.5326 GFLOPs
[19:30:36.829045] Median sample FLOPs: 443.9802 GFLOPs
[19:30:36.829067] **Testing time 0:00:58**

（2）记录训练过程中MAE最小的模型在PlantCount测试集上的测试结果

```
python test.py --resume ./output_dir/checkpoint-666.pth
```

[19:34:04.632144] Averaged stats: 
[19:34:04.632184] **MAE: 5.52,RMSE: 8.62,rMAE: 0.43,rRMSE: 0.52,R2: 0.93**
[19:34:04.632195] 
FLOPs Summary:
[19:34:04.632209] Total FLOPs for all samples: 67.9290 TFLOPs
[19:34:04.632222] **Average FLOPs per sample:** **449.8607 GFLOPs**
[19:34:04.632246] Min sample FLOPs: 177.5921 GFLOPs
[19:34:04.632259] Max sample FLOPs: 1509.5326 GFLOPs
[19:34:04.632540] Median sample FLOPs: 443.9802 GFLOPs
[19:34:04.632564] **Testing time 0:00:57**

（3）记录训练30个epoch后的模型在PlantCount测试集上的测试结果

```bash
python test.py --resume ./output_dir/checkpoint-30.pth
```

[19:37:57.936347] **MAE: 5.33,RMSE: 8.26,rMAE: 0.41,rRMSE: 0.50,R2: 0.94**
[19:37:57.936360] 
FLOPs Summary:
[19:37:57.936369] Total FLOPs for all samples: 67.9290 TFLOPs
[19:37:57.936382] **Average FLOPs per sample: 449.8607 GFLOPs**
[19:37:57.936399] Min sample FLOPs: 177.5921 GFLOPs
[19:37:57.936412] Max sample FLOPs: 1509.5326 GFLOPs
[19:37:57.936680] Median sample FLOPs: 443.9802 GFLOPs
[19:37:57.936705] **Testing time 0:00:56**

（4）记录训练49个epoch后的模型在PlantCount测试集上的测试结果

```bash
python test.py --resume ./output_dir/checkpoint-49.pth
```

[19:51:08.438072] Averaged stats: 
[19:51:08.438108] **MAE: 5.52,RMSE: 8.62,rMAE: 0.43,rRMSE: 0.52,R2: 0.93**
[19:51:08.438117] 
FLOPs Summary:
[19:51:08.438134] Total FLOPs for all samples: 67.9290 TFLOPs
[19:51:08.438145] **Average FLOPs per sample:** **449.8607 GFLOPs**
[19:51:08.438170] Min sample FLOPs: 177.5921 GFLOPs
[19:51:08.438182] Max sample FLOPs: 1509.5326 GFLOPs
[19:51:08.438450] Median sample FLOPs: 443.9802 GFLOPs
[19:51:08.438472] **Testing time 0:00:58**

> 注意：为了防止输出被覆盖，这里需要将output_dir和log_dir重新命名为output_dir_0608_1和log_dir_0608_1，然后再进行3

#### **3.根据测试结果找到最优模型**

目前最优的模型为checkpoint-30，而且epoch=30到epoch=56之间都没有比checkpoint-30性能更优良的模型，那么就在checkpoint-30附件找最优模型

在train_val.py中进行修改：

```
if args.output_dir and epoch >= 20 and epoch % 1 == 0 # 从第20轮开始，每轮都进行验证
if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):# 每1个epoch保存一次,训练结束前的最后一步保存
```

模型训练：

```bash
python train_val.py --resume ./output_dir_0608_1/checkpoint-20.pth --start_epoch 20 --epochs 40
```

模型测试：

```
python test.py --resume ./output_dir/checkpoint-666.pth
```

[23:14:23.329991] Averaged stats: 
[23:14:23.330031] **MAE: 5.25,RMSE: 7.37,rMAE: 0.42,rRMSE: 0.44,R2: 0.95**
[23:14:23.330043] 
FLOPs Summary:
[23:14:23.330057] Total FLOPs for all samples: 67.9290 TFLOPs
[23:14:23.330069] **Average FLOPs per sample: 449.8607 GFLOPs**
[23:14:23.330089] Min sample FLOPs: 177.5921 GFLOPs
[23:14:23.330102] Max sample FLOPs: 1509.5326 GFLOPs
[23:14:23.330367] Median sample FLOPs: 443.9802 GFLOPs
[23:14:23.330393] **Testing time 0:00:57**

找到性能更优良的模型权重！

> 注意：为了防止输出被覆盖，这里需要将output_dir和log_dir重新命名为output_dir_0608_2和log_dir_0608_2，然后再进行4

**4.验证最优模型在FSC147测试集上的测试结果**

在test.py中修改数据的路径为：

```
data_path = './data/FSC147_384_V2/'
anno_file = './data/FSC147_384_V2/annotation_FSC147_384.json'
data_split_file = './data/FSC147_384_V2/Train_Test_Val_FSC_147.json'
im_dir = './data/FSC147_384_V2/images_384_VarV2'
gt_dir = './data/FSC147_384_V2/gt_density_map_adaptive_384_VarV2'
```

运行代码：

```
python test.py --resume ./output_dir_0608_2/checkpoint-666.pth
```

[11:20:23.582749] Averaged stats: 
[11:20:23.582788] **MAE:23.01,RMSE:131.16,rMAE: 0.24,rRMSE: 1.98,R2: 0.20**
[11:20:23.582798] 
FLOPs Summary:
[11:20:23.582814] Total FLOPs for all samples: 276.3333 TFLOPs
[11:20:23.582842] **Average FLOPs per sample: 232.2128 GFLOPs**
[11:20:23.582949] Min sample FLOPs: 88.7960 GFLOPs
[11:20:23.582982] Max sample FLOPs: 3551.8415 GFLOPs
[11:20:23.583855] Median sample FLOPs: 177.5921 GFLOPs
[11:20:23.583887] **Testing time 0:03:39**

### 四、结论

<a href="https://pan.quark.cn/s/aaa63b751b19" title="最优模型权重">最优模型权重</a>，提取码：QDVx

最优模型权重在本项目的植物计数数据集上的测试集上的测试结果为：

|         指标名称         |    指标数值     |
| :----------------------: | :-------------: |
|           MAE            |      5.25       |
|           RMSE           |      7.37       |
|           rMAE           |      0.42       |
|          rRMSE           |      0.44       |
|            R2            |      0.95       |
| Average FLOPs per sample | 449.8607 GFLOPs |
|       Testing time       |       57s       |

