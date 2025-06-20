# PlantCountBMNNet+

### 一、数据集及预训练模型文件下载

FSC147数据集下载：参考<a href="https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master" title="Learning To Count Everything">Learning To Count Everything</a>

本项目使用的植物计数数据集下载：<a href="https://pan.quark.cn/s/ccfa3de40a56" title="PlantCountDataset">PlantCountDataset</a>，提取码：tuq8

### 二、环境配置

**建议Python、PyTorch、CUDA版本：**

```
Python 3.8.5
PyTorch 1.8.1
CUDA 11.1
```

**依赖安装：**

```bash
pip install -r requirements.txt
```

### 三、数据集准备

将下载的数据集按如下格式放在对应存储数据的文件夹中：
````
$PATH_TO_DATASET/
├──── gt_density_map_ours
│    ├──── 1037 density maps (.npy files)
│    
├──── images_ours
│    ├──── 1037 images (.jpg)
│ 
├────annotation_ours.json (annotation file)
├────train.txt
├────val.txt
├────test.txt

````

### 四、训练

在config文件夹中修改参数文件bmnet+_ours.yaml，调整数据集路径及训练参数，并运行以下命令来训练BMNet+：
```bash
python train.py --cfg 'config/bmnet+_ours.yaml'
```

### 五、测试

在config文件夹中修改参数文件test_bmnet+_ours.yaml，调整数据集路径、模型参数路径及测试参数，并运行以下命令来测试BMNet+：
```bash
python train.py --cfg 'config/test_bmnet+_ours.yaml'
```

### 六、结论

在默认配置下，训练`epochs=600`后，有最优模型。

<a href="https://pan.quark.cn/s/ec72ef56ff90" title="最优模型权重及训练过程">最优模型权重及训练过程</a>，提取码：HxAw

最优模型权重在本项目的植物计数数据集上的训练集上的测试结果为：

|         指标名称         |    指标数值     |
| :----------------------: | :-------------: |
|           MAE            |      8.63       |
|           RMSE           |     12.36       |
|           rMAE           |      0.58       |
|          rRMSE           |      0.95       |
|            R2            |      0.59       |
|          FLOPs           |    8.38e+10     |

最优模型权重在本项目的植物计数数据集上的验证集上的测试结果为：

|         指标名称         |    指标数值     |
| :----------------------: | :-------------: |
|           MAE            |      7.51       |
|           RMSE           |     11.00       |
|           rMAE           |      0.43       |
|          rRMSE           |      0.65       |
|            R2            |      0.57       |
|          FLOPs           |    7.87e+10     |

最优模型权重在本项目的植物计数数据集上的测试集上的测试结果为：

|         指标名称         |    指标数值     |
| :----------------------: | :-------------: |
|           MAE            |      6.47       |
|           RMSE           |     12.00       |
|           rMAE           |      0.50       |
|          rRMSE           |      0.71       |
|            R2            |      0.87       |
|          FLOPs           |    6.84e+10     |