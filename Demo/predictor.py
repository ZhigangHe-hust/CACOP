import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import sys
sys.path.append('..')
from models.geco_infer import build_model
from utils.arg_parser import get_argparser
from utils.data import resize_and_pad
import argparse
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from torchvision import ops
import torchvision.transforms as T

def get_args_parser():
    parser = argparse.ArgumentParser('GeCo', parents=[get_argparser()])
    return parser

class PlantCounter:
    def __init__(self, model_path):
        # 参数设置
        args = get_args_parser().parse_args()
        
        # set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Init model
        self.model = build_model(args)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.model.to(self.device)
        
        # Load model
        self.model.load_state_dict(
            torch.load(model_path, weights_only=True)['model'], strict=False
        )
        
        # eval mode
        self.model.eval()
        
    def preprocess_image(self, image, boxes):
        
        # Use resize_and_pad function from original demo
        image, boxes, scale = resize_and_pad(image, boxes, full_stretch=False)
        
        # Normalize image
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)
        
        return image, boxes, scale

    def predict(self, image, boxes):     
        # Convert PIL image to tensor if needed
        if isinstance(image, Image.Image):
            image = T.ToTensor()(image)
        
        # Convert boxes to tensor if they aren't already
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = boxes.clone().detach()
        
        # Ensure boxes are in the correct format [y1, x1, y2, x2]
        if boxes.shape[0] > 0:  # Only swap if there are boxes
            boxes = boxes[:, [1, 0, 3, 2]]  # Swap coordinates to match original format
        
        # Preprocess image and boxes
        img, boxes, scale = self.preprocess_image(image, boxes)
            
        # Add batch dimension if not present
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(boxes.shape) == 2:
            boxes = boxes.unsqueeze(0)
            
        # Move to device
        img = img.to(self.device)
        boxes = boxes.to(self.device)
        
        with torch.no_grad():
            # Get prediction
            outputs, _, _, _, masks = self.model(img, boxes)
            
            # Process outputs similar to demo.py
            idx = 0
            thr = 4
            keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr],
                   outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr], 0.5)

            boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr])[keep]
            boxes = torch.clamp(boxes, 0, 1)
            
            # Convert boxes back to original scale
            pred_boxes = boxes.cpu() / torch.tensor([scale, scale, scale, scale]) * img.shape[-1]
            
            # Process masks
            masks_ = masks[idx][(outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr)[0]]
            N_masks = masks_.shape[0]
            indices = torch.randint(1, N_masks + 1, (1, N_masks), device=masks_.device).view(-1, 1, 1)
            masks = (masks_ * indices).sum(dim=0)

            mask_display = (
                transforms.Resize((int(img.shape[2] / scale), int(img.shape[3] / scale)), 
                        interpolation=transforms.InterpolationMode.NEAREST)(
                    masks.cpu().unsqueeze(0))[0])[:image.shape[1], :image.shape[2]]
                       
            # Convert mask to RGBA image using matplotlib colormap
            cmap = plt.cm.tab20  # Use a colormap with distinct colors
            norm = plt.Normalize(vmin=0, vmax=N_masks)
            rgba_image = cmap(norm(mask_display))
            rgba_image[mask_display == 0, -1] = 0  # Set alpha to 0 for background
            mask_image = (rgba_image * 255).astype(np.uint8)
            
            # Clean up
            del masks
            del masks_
            del outputs
            
            return mask_image, len(boxes), pred_boxes
            