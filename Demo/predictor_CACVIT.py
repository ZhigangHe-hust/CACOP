import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import sys
sys.path.append('../PlantCount_CACViT')
import models.CACViT as CntViT
import util.misc as misc
import argparse
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL')
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N')
    parser.add_argument('--data_path', default='/data', type=str)
    parser.add_argument('--output_dir', default='/output_dir')
    parser.add_argument('--log_dir', default='/log_dir')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='../PlantCount_CACViT/checkpoints/best_model.pth')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    return parser

class PlantCounter:
    def __init__(self, model_path):
        # 参数设置
        args = get_args_parser().parse_args()
        args.resume = model_path
        
        # Initialize
        misc.init_distributed_mode(args)
        
        # set device
        self.device = torch.device(args.device)
        
        # random seed
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Init model
        self.model = CntViT.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        self.model.to(self.device)
        
        # Load model
        misc.load_model_FSC(args=args, model_without_ddp=self.model)
        
        # eval mode
        self.model.eval()
        
    def preprocess_image(self, image, boxes):
        # Convert image to tensor
        image = transforms.ToTensor()(image)
        
        # Process boxes
        scale_x = []
        scale_y = []
        processed_boxes = []
        
        for box in boxes:
            y1, x1, y2, x2 = box
            scale_x1 = torch.tensor((x2-x1+1)/384)
            scale_x.append(scale_x1)
            scale_y1 = torch.tensor((y2-y1+1)/384)
            scale_y.append(scale_y1)
            
            # Crop and resize box
            bbox = image[:,y1:y2+1,x1:x2+1]
            bbox = transforms.Resize((64, 64))(bbox)
            processed_boxes.append(bbox.numpy())
            
        scale_xx = torch.stack(scale_x).unsqueeze(-1)  # [N, 1]
        scale_yy = torch.stack(scale_y).unsqueeze(-1)  # [N, 1]
        scale = torch.cat((scale_xx, scale_yy), dim=1)  # [N, 2]
        
        processed_boxes = np.array(processed_boxes)
        processed_boxes = torch.Tensor(processed_boxes)
        # [bs, n, c, h, w]
        processed_boxes = processed_boxes.unsqueeze(0)  # Add batch dimension
        
        scale = scale.unsqueeze(0)  # [1, N, 2]
        
        return image, processed_boxes, scale
        
    def get_cluster_points(self, density_map, eps=20, min_samples=3):
        """
        Convert density map to cluster points using DBSCAN
        Args:
            density_map: numpy array of density values
            eps: DBSCAN parameter for maximum distance between points
            min_samples: DBSCAN parameter for minimum points in a cluster
        Returns:
            cluster_points: list of (x, y) coordinates for cluster centers
            cluster_count: number of clusters
        """
        # Apply Gaussian blur to smooth the density map
        smoothed_map = gaussian_filter(density_map, sigma=1.0)
        
        # Get points where density is above threshold
        threshold = np.max(smoothed_map) * 0.1  # 10% of max density
        points = np.where(smoothed_map > threshold)
        points = np.column_stack((points[1], points[0]))  # Convert to (x, y) format
        values = smoothed_map[points[:, 1], points[:, 0]]
        
        if len(points) == 0:
            return [], 0
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        # Calculate cluster centers using weighted average
        cluster_points = []
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
            mask = labels == label
            cluster_points_coords = points[mask]
            cluster_values = values[mask]
            
            # Calculate weighted center
            weights = cluster_values / np.sum(cluster_values)
            center_x = np.sum(cluster_points_coords[:, 0] * weights)
            center_y = np.sum(cluster_points_coords[:, 1] * weights)
            cluster_points.append((center_x, center_y))
        
        return cluster_points, len(cluster_points)

    def create_cluster_image(self, cluster_points, height, width):
        """
        Create a transparent image with cluster points marked
        """
        # Create transparent image
        cluster_image = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Mark cluster points
        for x, y in cluster_points:
            x, y = int(x), int(y)
            # Draw a small circle at each cluster point
            for i in range(max(0, x-10), min(width, x+10)):
                for j in range(max(0, y-10), min(height, y+10)):
                    dist = (i-x)**2 + (j-y)**2
                    if dist <= 81:  # Circle with radius 20
                        if dist <= 64:  # Inner circle (radius 18)
                            cluster_image[j, i] = [255, 0, 0, 255]  # Solid red
                        else:  # Border (radius 18-20)
                            cluster_image[j, i] = [255, 255, 255, 255]  # White border
        
        return cluster_image

    def predict(self, image, boxes):
        # Preprocess
        image, processed_boxes, scale = self.preprocess_image(image, boxes)
        
        # Move to device
        image = image.to(self.device)
        processed_boxes = processed_boxes.to(self.device)
        scale = scale.to(self.device)
        
        # Get image dimensions
        _, h, w = image.shape
        
        # Initialize density map
        density_map = torch.zeros([h, w], device=self.device)
        
        # Process image in patches
        start = 0
        prev = -1
        
        with torch.no_grad():
            while start + 383 < w:
                # Prepare patch
                patch = image[:,:,start:start+384]
                patch = torch.nn.functional.interpolate(
                    patch.unsqueeze(0), 
                    size=(384, 384), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Prepare input
                input_x = [patch, processed_boxes, scale]
                
                # Get prediction
                output = self.model(input_x)
                output = output.squeeze(0)
                
                # Resize output
                output = output.unsqueeze(0).unsqueeze(0)
                output = torch.nn.functional.interpolate(
                    output, 
                    size=(h, 384), 
                    mode='bilinear', 
                    align_corners=False
                )
                output = output.squeeze(0).squeeze(0)
                
                # Update density map
                b1 = nn.ZeroPad2d(padding=(start, w-prev-1, 0, 0))
                d1 = b1(output[:,0:prev-start+1])
                b2 = nn.ZeroPad2d(padding=(prev+1, w-start-384, 0, 0))
                d2 = b2(output[:,prev-start+1:384])
                
                b3 = nn.ZeroPad2d(padding=(0, w-start, 0, 0))
                density_map_l = b3(density_map[:,0:start])
                density_map_m = b1(density_map[:,start:prev+1])
                b4 = nn.ZeroPad2d(padding=(prev+1, 0, 0, 0))
                density_map_r = b4(density_map[:,prev+1:w])
                
                density_map = density_map_l + density_map_r + density_map_m/2 + d1/2 + d2
                
                prev = start + 383
                start = start + 128
                if start+383 >= w:
                    if start == w - 384 + 128: 
                        break
                    else: 
                        start = w - 384
        
        # Convert density map to numpy
        density_map_np = density_map.cpu().numpy()
        
        # Calculate count before converting density_map_np back to tensor
        count = torch.sum(torch.from_numpy(density_map_np)/60).item()

        # Get cluster points
        cluster_points, cluster_count = self.get_cluster_points(density_map_np)
        
        # Create cluster image
        cluster_image = self.create_cluster_image(cluster_points, h, w)
        
        return cluster_image, cluster_count, density_map_np

def convert_np_image_to_base64(image_np, cmap='viridis', alpha=0.5):
    """
    Converts a numpy array image to a base64 string (for density map).
    Applies colormap and adjusts alpha.
    """
    # Normalize density map for colormap application
    if image_np.max() > 0:
        normalized_map = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    else:
        normalized_map = np.zeros_like(image_np)

    # Apply colormap and convert to RGB
    cmap_func = plt.get_cmap(cmap)
    colored_map = cmap_func(normalized_map)[:, :, :3]  # Remove alpha channel if present from cmap

    # Apply overall alpha to the RGB values
    colored_map = (colored_map * 255 * alpha).astype(np.uint8) # Apply alpha here
    
    # Convert to PIL Image and then to base64
    img_pil = Image.fromarray(colored_map)
    buffer = BytesIO()
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8') 