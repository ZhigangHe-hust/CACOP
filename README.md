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

  

---

## 模型复现

已成功复现2个模型：BMNet+、CACViT，模型复现结果如下：

* #### BMNNet+

  ```
  train：MAE 8.63, rMAE 0.58, RMSE 12.36, rRMSE 0.95, R2 0.59, FLOPS 8.38e+10
  val：MAE 7.51, rMAE 0.43, RMSE 11.00, rRMSE 0.65, R2 0.57, FLOPS 7.87e+10
  test：MAE 6.47, rMAE 0.50, RMSE 12.00, rRMSE 0.71, R2 0.87, FLOPS 6.84e+10
  ```

* #### CACViT

  ```
  
  ```

尝试过但有问题的模型：

* #### SSD

  模型仓库代码不全，缺少一些模型函数文件。

* #### LOCA

  模型代码以Linux系统为基础，不便在windows系统运行。

正在进行的工作：


​       查阅文献撰写报告，以及尝试其他模型。
