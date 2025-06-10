import torch
import json

# # 1.检查权重文件是否损坏
# checkpoint = torch.load('E:/PR_ML_Course_Design_No_14/Code/CACViT-AAAI24/pretrain/best_model.pth', map_location='cpu')


# # 2.查看字典的键值对（txt）
# # class_file = 'E:/PR_ML_Course_Design_No_14/Code/CACViT-AAAI24/data/FSC147_384_V2/ImageClasses_FSC147.txt'
# class_file = './data/FSC147_384_V2/ImageClasses_FSC147.txt'
# class_dict = {}
# with open(class_file) as f:
#     for line in f:
#         key = line.split()[0]
#         val = line.split()[1:]
#         class_dict[key] = val

# items = list(class_dict.items())  
# print(items[0][0])  # 第一个键：2.jpg
# print(items[0][1])  # 第一个值：['sea', 'shells']


# 3.查看字典的键值对（json)
anno_file = './data/annotations.json'
with open(anno_file) as f:
    annotations = json.load(f)
items = list(annotations.items())  
print(items[0][0])  # 第一个键：abelmoschus_esculentus_green_fruits_1
print(items[0][1])  
# 第一个值：
# {
# 'box_examples_coordinates': 
# [ [[106, 511], [179, 511], [179, 592], [106, 592]], 
#   [[434, 602], [505, 602], [505, 676], [434, 676]], 
#   [[387, 216], [457, 216], [457, 282], [387, 282]] ], 
# 'points': 
# [ [88, 177], [109, 236], [144, 273], [75, 350], [93, 671], [149, 567],
#   [227, 690], [285, 647], [343, 521], [369, 440], [305, 341], [295, 291], 
#   [263, 221], [301, 172], [375, 222], [409, 254], [509, 230], [485, 512], 
#   [532, 533], [595, 499], [561, 270], [609, 203], [593, 302], [597, 351], 
#   [678, 646], [812, 586], [695, 446], [773, 265], [394, 686], [463, 622], 
#   [517, 632], [547, 642], [591, 640], [728, 581], [809, 660], [853, 584], [918, 560] ] 
# }


# # 4.查看字典的键值对（json)
# data_split_file = 'E:/PR_ML_Course_Design_No_14/Code/CACViT-AAAI24/data/FSC147_384_V2/Train_Test_Val_FSC_147.json'
# data_split_file = './data/FSC147_384_V2/Train_Test_Val_FSC_147.json'
# with open(data_split_file) as f:
#     data_split = json.load(f)
# items = list(data_split.items())  
# print(items[0][0])  # 第1个键：test
# print(items[1][0])  # 第2个键：test_coco
# print(items[2][0])  # 第3个键：train
# print(items[3][0])  # 第4个键：val
# print(items[4][0])  # 第5个键：val_coco
# print("-"*20)
# print(items[0][1])  # 第1个值：['2.jpg', ···, '6901.jpg']
