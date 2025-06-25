# PlantCount FamNet

### 一、数据集及预训练模型文件下载

FSC147数据集下载：参考<a href="https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master" title="Learning To Count Everything">Learning To Count Everything</a>

本项目使用的数据集、密度图以及最终训练参数下载：<a href="https://pan.quark.cn/s/0903ab844631" title="FamNet_data">FamNet_data</a>

### 二、环境配置

**建议Python、PyTorch、CUDA版本：**

```
Python 3.9.21
PyTorch 2.0.1
CUDA 11.8
```

**依赖安装：**

```bash   ”“bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

### 三、数据集准备

> **下载完成后，请将解压得到的数据集内容放入项目根目录下的 `data/` 文件夹内，目录结构如下：**
````
$PATH_TO_DATASET/
├──── gt_density_map_ours
│    ├──── 1037 density maps (.npy files)
│    
├──── images_ours
│    ├──── 1037 images (.jpg)
│ 
├────annotation_ours.json (annotation file)
├────ImageClasses_ours.txt
├────Train_Test_Val_ours.json
````

### 四、训练

训练时可根据需要调整下列参数，包括 `data_path`、`test_split`、`epochs`、`learning_rate`、`gpu-id` 等，并运行以下命令来训练FamNet：
```bash   ”“bash
python train.py
```

### 五、测试

测试时可根据需要调整下列参数，包括 `data_path`、`test_split`、`model_path`、`gpu-id` 等，并运行以下命令来测试FamNet：
#### 1. 在验证集（val）上测试，不进行适应
```bash   ”“bash
python test.py --data_path /PATH/TO/YOUR/DATASET/ --test_split val
```

#### 2. 在验证集（val）上测试，并开启测试时适应（adaptation）
```bash   ”“bash
python test.py --data_path /PATH/TO/YOUR/DATASET/ --test_split val --adapt
```

### 六、结论

在默认配置下，训练 `epochs=1500` 后，取最优模型。

最优模型权重在本项目的植物计数数据集上的训练集上的测试结果为：

|         指标名称         |    指标数值     |
| :----------------------: | :-------------: |
|           MAE            |      9.07       |
|           RMSE           |     13.21       |   
|           rMAE           |     0.4213      |
|          rRMSE           |     0.6139      |
|            R2            |     0.5318      |  
|          FLOPs           |   1.33e+11      |   

在本项目的植物计数数据集上的验证集上的测试结果为：

|         指标名称         |    指标数值     |
| :----------------------: | :-------------: |
|           MAE            |     10.02       |
|           RMSE           |     15.03       |  
|           rMAE           |     0.4845      |
|          rRMSE           |     0.7269      |
|            R2            |     0.1955      |  
|          FLOPs           |   1.33e+11      |  

在本项目的植物计数数据集上的测试集上的测试结果为：

|         指标名称         |    指标数值     |
| :----------------------: | :-------------: |
|           MAE            |     13.41       |
|           RMSE           |     33.31       |  
|           rMAE           |     0.8094      |
|          rRMSE           |     2.0101      |
|            R2            |     0.0251      |  
|          FLOPs           |   1.33e+11      |  
