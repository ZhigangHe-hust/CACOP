# PlantCountSPDCN

## Datasets

FSC147数据集下载：参考<a href="https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master" title="Learning To Count Everything">Learning To Count Everything</a>

本项目数据集下载：见[Google Drive](https://drive.google.com/drive/folders/18QmVV1Yxm3cWzVoxtLsw0fQFtk1zcI8j?usp=sharing)


### 已验证环境：

```
PyTorch  2.0.0
Python  3.8(ubuntu20.04)
CUDA  11.8
GPU RTX4090 24GB(训练时近满显存)
```

### 环境部署

```
# 镜像创建
conda create -n geco python=3.8
conda activate geco

# 依赖安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
pip install tqdm
pip install pycocotools
pip install scipy
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Data Preparation

* 已下载数据集放入`\DATA_folder`，该文件夹下图像单独存放`\images`下，其余放在`\annotations`下
* `python utils/data.py --data_path DATA_folder`重新生成密度图
* `python generate_coco_annotations.py`划分`instances_{train\test\val}.json`
* 下载[sam_hq_vit_h.pth](https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing)于`\MODEL_folder`

## Training

* 修改参数于`./run.sh`

* 运行`bash ./run.sh`

## Result 

* 最优权重下载[GeCo_Plant](https://pan.quark.cn/s/02046fd26c17)
* 相关指标

| 划分集 | MAE    | RMSE   | rMAE   | rRMSE  | R2     | FLOPs        |
|--------|--------|--------|--------|--------|--------|--------------|
| Train  | 10.53  | 16.42  | 0.4893 | 0.7630 | 0.2768 | 5712.09G   |
| Val    | 6.77   | 9.10   | 0.3275 | 0.4403 | 0.7049 | 5712.09G   |
| Test   | 6.74   | 10.56  | 0.4069 | 0.6375 | 0.9019 | 5712.09G   |
