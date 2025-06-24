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
  └── images   
  ```

* 数据集已上传至[谷歌网盘](https://drive.google.com/drive/folders/18QmVV1Yxm3cWzVoxtLsw0fQFtk1zcI8j?usp=sharing)

---

## 模型复现

已成功复现5个模型，模型复现结果如下：

* #### BMNet+

  ```
  train：MAE 8.63, rMAE 0.58, RMSE 12.36, rRMSE 0.95, R2 0.59, FLOPS 83.8G
  val：  MAE 7.51, rMAE 0.43, RMSE 11.00, rRMSE 0.65, R2 0.57, FLOPS 78.7G
  test： MAE 6.47, rMAE 0.50, RMSE 12.00, rRMSE 0.71, R2 0.87, FLOPS 68.4G
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
  Train：MAE: 10.53, rMAE: 0.4893, RMSE: 16.42, rRMSE: 0.7630, R2: 0.2768, FLOPS: 5712.09G
  Val：  MAE: 6.77 , rMAE: 0.3275, RMSE: 9.10 , rRMSE: 0.4403, R2: 0.7049, FLOPS: 5712.09G
  Test： MAE: 6.74 , rMAE: 0.4069, RMSE: 10.56, rRMSE: 0.6375, R2: 0.9019, FLOPS: 5712.09G
  ```

## Demo
  * 代码见`\Demo`，基于`flask`实现，运行后在浏览器打开生成的网址
  * 基于可视化效果最优的`GeCo`实现
  * 由于GeCo显存占用大，因此还开发了基于`CACViT`的轻便版本（`demo_CACVIT.py`、`predictor_CACVIT.py`、`\templates\demo_CACVIT.html`）
  * 示例`./demo.mp4`
    

https://github.com/user-attachments/assets/2561c451-8c3c-49e6-ba7d-fd047cc0b76b

