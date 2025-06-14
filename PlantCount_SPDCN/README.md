# PlantCountSPDCN

## Datasets

FSC147数据集下载：参考<a href="https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master" title="Learning To Count Everything">Learning To Count Everything</a>

本项目数据集下载：见[Google Drive](https://drive.google.com/drive/folders/18QmVV1Yxm3cWzVoxtLsw0fQFtk1zcI8j?usp=sharing)

## Requirement

(同[SPDCN-CAC](https://github.com/Elin24/SPDCN-CAC/tree/main))

### 已验证环境：

```
PyTorch 1.9.0
Python 3.8(ubuntu18.04)
CUDA 11.1
```

### 快速部署

```
# 镜像创建
conda create -n SPDCN python=3.8
conda activate SPDCN

# 依赖安装
cd CACOP/PlantCount_SPDCN
pip install -r requirements.txt
```

## Data Preparation

* 已下载数据集放入`\datasets`
* 修改`datasets/gendata384x576.py`中`root`路径以及其他数据集相关文件路径（或直接将数据集相关实际文件名统一为代码中名称，避免后续`train.py`中修改）
* 运行`python datasets/gendata384x576.py`

## Training

* 修改参数于`/config.py`，修改数据集路径于`/run.sh`

* 运行`./run.sh`

## Result 

在默认配置下，训练`epochs=100`后，有最优模型：

* 权重下载[SPDCN](https://drive.google.com/file/d/1QpcYdKCXyH9iLMzig3MpFduJVKHGc1E5/view?usp=sharing)

* 评估指标

```
[2025-06-14 15:00:12 SPDCN](val.py 176): INFO  * MAE 10.505 RMSE 15.547
[2025-06-14 15:00:12 SPDCN](val.py 177): INFO  * rMAE 0.643 rRMSE 0.754
[2025-06-14 15:00:12 SPDCN](val.py 178): INFO  * R2 0.759
[2025-06-14 15:00:12 SPDCN](val.py 180): INFO  * Average FLOPs per sample: 70.45G
[2025-06-14 15:00:12 SPDCN](val.py 181): INFO  * Median sample FLOPs: 563.61G
```

