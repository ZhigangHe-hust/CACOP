"""
Training Code for Learning To Count Everything, CVPR 2021
Authors: Viresh Ranjan,Udbhav, Thu Nguyen, Minh Hoai

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
"""
import torch.nn as nn
from model import  Resnet50FPN,CountRegressor,weights_normal_init
from utils import MAPS, Scales, Transform,TransformTrain,extract_features, visualize_output_and_save
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists,join
import random
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.cuda.amp import autocast, GradScaler
import math


parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the dataset")
parser.add_argument("-o", "--output_dir", type=str,default="./logsSave_ours_lr_4_147", help="/Path/to/output/logs/")
parser.add_argument("-ts", "--test-split", type=str, default='val', choices=["train", "test", "val"], help="what data split to evaluate on on")
parser.add_argument("-ep", "--epochs", type=int,default=1500, help="number of training epochs")
parser.add_argument("-g", "--gpu", type=int,default=0, help="GPU id")
parser.add_argument("-lr", "--learning-rate", type=float,default=1e-5, help="learning rate")
args = parser.parse_args()


data_path = args.data_path
anno_file = data_path + 'annotation_ours.json'
data_split_file = data_path + 'Train_Test_Val_ours.json'
im_dir = data_path + 'images_ours'
gt_dir = data_path + 'gt_density_map_adaptive_ours'

if not exists(args.output_dir):
    os.mkdir(args.output_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

criterion = nn.MSELoss().cuda()

resnet50_conv = Resnet50FPN()
resnet50_conv.cuda()
# resnet50_conv.eval()
resnet50_conv.train()

regressor = CountRegressor(6, pool='mean')
weights_normal_init(regressor, dev=0.001)

# 加载预训练参数
checkpoint_path = './data/pretrainedModels/FamNet_Save0.pth'  # 改为你真实的路径
if os.path.exists(checkpoint_path):
    print("Loading pre-trained weights from", checkpoint_path)
    regressor.load_state_dict(torch.load(checkpoint_path), strict=False)
else:
    print("Pre-trained checkpoint not found, training from scratch.")

regressor.train()
regressor.cuda()
optimizer = optim.Adam(regressor.parameters(), lr = args.learning_rate)

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

    
def train():
    # scaler = GradScaler()

    im_ids = data_split['train']
    random.shuffle(im_ids)
    
    #train_mae = 0
    #train_rmse = 0
    #train_loss = 0
    train_loss = 0
    gt_counts, pred_counts = [], []
    
    pbar = tqdm(im_ids)
    cnt = 0
    for im_id in pbar:
        cnt += 1
        # anno = annotations[im_id]
        # 假如 im_id 是 "Morus_alba_fruit_9"
        if not im_id.endswith('.jpg'):
            im_id = im_id + '.jpg'
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        #rects = list()
        rects = []
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        density_path = gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype('float32')    
        sample = {'image':image,'lines_boxes':rects,'gt_density':density}
        sample = TransformTrain(sample)
        image, boxes,gt_density = sample['image'].cuda(), sample['boxes'].cuda(),sample['gt_density'].cuda()

        # with torch.no_grad():
        #     features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
        features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
        #features.requires_grad = True
        #print("features sum:", features.sum().item())
        optimizer.zero_grad()
        output = regressor(features)

        #if image size isn't divisible by 8, gt size is slightly different from output size
        if output.shape[2] != gt_density.shape[2] or output.shape[3] != gt_density.shape[3]:
            orig_count = gt_density.sum().detach().item()
            gt_density = F.interpolate(gt_density, size=(output.shape[2],output.shape[3]),mode='bilinear')
            new_count = gt_density.sum().detach().item()
            if new_count > 0: gt_density = gt_density * (orig_count / new_count)
        loss = criterion(output, gt_density)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred_cnt = torch.sum(output).item()
        gt_cnt = torch.sum(gt_density).item()
        
        gt_counts.append(gt_cnt)
        pred_counts.append(pred_cnt)
        if math.isnan(pred_cnt) or math.isnan(gt_cnt):
            print(f"[⚠️ NaN] Skipping {im_id}: pred_cnt={pred_cnt}, gt_cnt={gt_cnt}")
            continue


        cnt = len(gt_counts)
        mae = np.mean(np.abs(np.array(gt_counts) - np.array(pred_counts)))
        rmse = np.sqrt(np.mean((np.array(gt_counts) - np.array(pred_counts)) ** 2))
        mean_gt = np.mean(gt_counts)
        rmae = mae / mean_gt if mean_gt != 0 else 0
        rrmse = rmse / mean_gt if mean_gt != 0 else 0
        ss_res = np.sum((np.array(gt_counts) - np.array(pred_counts)) ** 2)
        ss_tot = np.sum((np.array(gt_counts) - np.mean(gt_counts)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        
        pbar.set_description(
            'GT: {:5.1f}, Pred: {:5.1f}, MAE: {:5.2f}, RMSE: {:5.2f}, rMAE: {:5.4f}, rRMSE: {:5.4f}, R2: {:5.4f}'.format(
                gt_cnt, pred_cnt, mae, rmse, rmae, rrmse, r2))
      
        print("")
    train_loss = train_loss / len(im_ids)
    #train_mae = (train_mae / len(im_ids))
    #train_rmse = (train_rmse / len(im_ids))**0.5
    torch.cuda.empty_cache()
    return train_loss, mae, rmse, rmae, rrmse, r2



   
def eval():
    #...
    cnt = 0
    SAE = 0 # sum of absolute errors
    SSE = 0 # sum of square errors
    rSAE = 0  # sum relative abs error
    rSSE = 0  # sum relative squared error
    gt_list = []
    pred_list = []

    
    print("Evaluation on {} data".format(args.test_split))
    im_ids = data_split[args.test_split]
    pbar = tqdm(im_ids)
    for im_id in pbar:
        if not im_id.endswith('.jpg'):
            im_id = im_id + '.jpg'
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        sample = {'image':image,'lines_boxes':rects}
        sample = Transform(sample)
        image, boxes = sample['image'].cuda(), sample['boxes'].cuda()

        with torch.no_grad():
            output = regressor(extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales))

        cnt = cnt + 1
        gt_cnt = dots.shape[0]
        pred_cnt = output.sum().item()
        SAE += abs(gt_cnt - pred_cnt)
        SSE += (gt_cnt - pred_cnt) ** 2
        rSAE += abs(gt_cnt - pred_cnt) / (gt_cnt + 1e-6)
        rSSE += ((gt_cnt - pred_cnt) / (gt_cnt + 1e-6)) ** 2
        gt_list.append(gt_cnt)
        pred_list.append(pred_cnt)
        #...
        

        pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))
        print("")
    
    #...
    MAE = SAE / cnt
    RMSE = (SSE / cnt) ** 0.5
    rMAE = rSAE / cnt
    rRMSE = (rSSE / cnt) ** 0.5

    #from sklearn.metrics import r2_score
    R2 = r2_score(gt_list, pred_list)

    print('On {} data, MAE: {:6.2f}, rMAE: {:.4f}, RMSE: {:6.2f}, rRMSE: {:.4f}, R2: {:.4f}'.format(
        args.test_split, MAE, rMAE, RMSE, rRMSE, R2
    ))
    
    #print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
    return MAE, RMSE, rMAE, rRMSE, R2
    #...


best_mae, best_rmse = 1e7, 1e7
stats = list()
for epoch in range(0,args.epochs):
    #FLOPS
    if epoch == 0:
#        # 随便选一个训练图像，拿到 features 的尺寸
#         sample_im_id = data_split['train'][0]
#         if not sample_im_id.endswith('.jpg'):
#             sample_im_id = sample_im_id + '.jpg'
#         anno = annotations[sample_im_id]
#         bboxes = anno['box_examples_coordinates']
#         rects = []
#         for bbox in bboxes:
#             x1, y1 = bbox[0][0], bbox[0][1]
#             x2, y2 = bbox[2][0], bbox[2][1]
#             rects.append([y1, x1, y2, x2])
#         image = Image.open('{}/{}'.format(im_dir, sample_im_id))
#         image.load()
#         sample = {'image': image, 'lines_boxes': rects}
#         sample = Transform(sample)
#         image, boxes = sample['image'].cuda(), sample['boxes'].cuda()

#         # 打印 feature shape
#         with torch.no_grad():
#             f_debug = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
#             print("[Debug] feature shape for FLOPs:", f_debug.shape)

        # 用打印出来的 feature shape 构造 dummy_input
        from thop import profile
        #dummy_input = torch.randn_like(f_debug)
        dummy_input = torch.randn(1, 3, 6, 90, 125).cuda()# 建议根据实际 features shape 调整
        flops, params = profile(regressor, inputs=(dummy_input,))
        print(f"[模型分析] Params: {params/1e6:.2f} M | FLOPs: {flops/1e9:.2f} GFLOPs")
        
        with open(join(args.output_dir, "flops.txt"), 'w') as f:
            f.write(f"FLOPs(G): {flops/1e9:.4f}\n")
            f.write(f"Params(M): {params/1e6:.4f}\n")

    
    
    regressor.train()
    train_loss,train_mae,train_rmse, train_rmae, train_rrmse, train_r2 = train()
    regressor.eval()
    val_mae, val_rmse, rmae, rrmse, r2 = eval()
    torch.cuda.empty_cache()
    
    stats.append((train_loss, train_mae, train_rmse, train_rmae, train_rrmse, train_r2, val_mae, rmae, val_rmse, rrmse, r2))
    stats_file = join(args.output_dir, "stats" +  ".txt")
    with open(stats_file, 'w') as f:
        for s in stats:
            f.write("%s\n" % ','.join([str(x) for x in s]))    
    if best_mae >= val_mae:
        best_mae = val_mae
        best_rmse = val_rmse
        model_name = args.output_dir + '/' + "FamNet.pth"
        torch.save(regressor.state_dict(), model_name)

    print(
    "Epoch {:3d} | Loss: {:.6f} | "
    "Train → MAE: {:.2f}, rMAE: {:.4f}, RMSE: {:.2f}, rRMSE: {:.4f}, R²: {:.4f} | "
    "Val → MAE: {:.2f}, rMAE: {:.4f}, RMSE: {:.2f}, rRMSE: {:.4f}, R²: {:.4f} | "
    "Best Val MAE: {:.2f}, RMSE: {:.2f}".format(
        epoch+1, train_loss,
        train_mae, train_rmae, train_rmse, train_rrmse, train_r2,
        val_mae, rmae, val_rmse, rrmse, r2,
        best_mae, best_rmse
    )
)





