# 类别无关植物计数(`CACOP`)项目进展

---

## 数据集收集

* 一千张植物图片已收集标注完毕，并整理完数据集类别表；

* 生成了数据集相关的注释和划分文件、密度图，项目数据集结构如下：

  ```
  our_data
  ├── Train_Val_Test.json
  ├── annotations.json
  ├── gt_density_maps
  └── images   └──图象   └──图象   └──图象
  ```

* 数据集已上传至[谷歌网盘](https://drive.google.com/drive/folders/18QmVV1Yxm3cWzVoxtLsw0fQFtk1zcI8j?usp=sharing)

---

## 模型复现

已成功复现5个模型：BMNet+、CACViT、FamNet，模型复现结果如下：

* #### BMNet+

  ```
  train：MAE 8.63, rMAE 0.58, RMSE 12.36, rRMSE 0.95, R2 0.59, FLOPS 8.38e+10train:MAE 8.63, rMAE 0.58, RMSE 12.36, rRMSE 0.95, R2 0.59, FLOPS 8.38 e10
  val：MAE 7.51, rMAE 0.43, RMSE 11.00, rRMSE 0.65, R2 0.57, FLOPS 7.87e+10val：MAE 7.51, rMAE 0.43, RMSE 11.00, rRMSE 0.65, R2 0.57, FLOPS 7.87e 10
  test：MAE 6.47, rMAE 0.50, RMSE 12.00, rRMSE 0.71, R2 0.87, FLOPS 6.84e+10检验：MAE 6.47, rMAE 0.50, RMSE 12.00, rRMSE 0.71, R2 0.87, FLOPS 6.84 e10
  ```

* #### CACViT

  ```
  [23:14:23.329991]Averaged stats:
  [23:14:23.330031]MAE:5.25,RMSE:7.37,rMAE:0.42,rRMSE:0.44,R2:0.95
  [23:14:23.330043]
  FLOPs Summary:   失败的总结:
  [23:14:23.330057]Total FLOPs for all samples: 67.9290 TFLOPs[23:14:23.330057]所有样品的总FLOPs: 67.9290 TFLOPs
  [23:14:23.330069]Average FLOPs per sample:449.8607 GFLOPs
  [23:14:23.330089]Min sample FLOPs:177.5921 GFLOPs
  [23:14:23.330102]MaX Sample FLOPs:1509.5326 GFLOPs
  [23:14:23.330367]Median sample FLOPs:443.9802 GFLOPs
  [23:14:23.330393]Testing time 0:00:57
  ```

* #### FamNet
  ```
  lr=1e-5：
  train：MAE:   9.07, rMAE:  0.4213, RMSE:  13.21, rRMSE:  0.6139, R²:  0.5318, FLOPS: 133.4948G
  val：MAE:  10.02, rMAE:  0.4845, RMSE:  15.03, rRMSE:  0.7269, R²:  0.1955, FLOPS: 133.4948G
  test：MAE:  13.41, rMAE:  0.8094, RMSE:  33.31, rRMSE:  2.0101, R²:  0.0251, FLOPS: 133.4948G
  ```

* #### SPDCN
  ```
  [2025-06-14 15:00:12 SPDCN](val.py 176): INFO  * MAE 10.505 RMSE 15.547
  [2025-06-14 15:00:12 SPDCN](val.py 177): INFO  * rMAE 0.643 rRMSE 0.754
  [2025-06-14 15:00:12 SPDCN](val.py 178): INFO  * R2 0.759
  [2025-06-14 15:00:12 SPDCN](val.py 180): INFO  * Average FLOPs per sample: 70.45G
  [2025-06-14 15:00:12 SPDCN](val.py 181): INFO  * Median sample FLOPs: 563.61G
  ```
* #### GeCo
  ```
  Train：MAE: 10.6274, rMAE: 0.3781, RMSE: 12.9121, rRMSE: 0.5253, R2: 0.6742, FLOPS: 2856.0426G
  Val：  MAE: 3.3954, rMAE: 0.3521, RMSE: 5.1146, rRMSE: 0.5143, R2: 0.8118, FLOPS: 2856.0426G
  Test： MAE: 5.8543, rMAE: 0.5832, RMSE: 7.7532,  rRMSE: 0.6075, R2: 0.8132, FLOPS: 2856.0426G
  ```

## Demo
  * 代码见`\Demo`，基于`flask`实现，运行后在浏览器打开生成的网址
  * 基于可视化效果最优的`GeCo`实现
  * 由于GeCo显存占用大，因此还开发了基于`CACViT`的轻便版本（`demo_CACVIT.py`、`predictor_CACVIT.py`、`\templates\demo_CACVIT.html`）
  * 示例`./demo.mp4` <video src="demo.mp4" controls width="640"></video>

尝试过但有问题的模型：

* #### CountGD

  需要提供全部样本的标注框。

* #### SSD

  模型仓库代码不全，缺少一些模型函数文件。

* #### LOCA

  模型代码以Linux系统为基础，不便在windows系统运行。

正在进行的工作：


​       查阅文献撰写报告，以及尝试其他模型。
