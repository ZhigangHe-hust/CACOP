import json
import os
from PIL import Image
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_image_info(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height

def convert_box_to_xywh(box_points):
    # box_points是一个包含4个点坐标的列表，每个点是一个[x,y]坐标
    # 转换为[x,y,width,height]格式
    x_coords = [point[0] for point in box_points]
    y_coords = [point[1] for point in box_points]
    
    x = min(x_coords)
    y = min(y_coords)
    width = max(x_coords) - x
    height = max(y_coords) - y
    
    return [x, y, width, height]

def generate_coco_annotations(split_name, image_list, annotation_data, images_dir):
    coco_data = {
        "images": [],
        "annotations": []
    }
    
    # 创建图片ID到文件名的映射
    image_id_map = {}
    for idx, image_name in enumerate(image_list, 1):
        image_path = os.path.join(images_dir, image_name)
        if os.path.exists(image_path):
            width, height = get_image_info(image_path)
            image_info = {
                "id": idx,
                "file_name": image_name,
                "width": width,
                "height": height
            }
            coco_data["images"].append(image_info)
            image_id_map[image_name] = idx
    
    # 添加标注信息
    annotation_id = 1
    for image_name, image_data in annotation_data.items():
        if image_name in image_id_map:
            image_id = image_id_map[image_name]
            # 获取box_examples_coordinates
            boxes = image_data.get("box_examples_coordinates", [])
            for box_points in boxes:
                # 将框的四个点坐标转换为[x,y,width,height]格式
                x, y, w, h = convert_box_to_xywh(box_points)
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # 假设所有对象都属于类别1
                    "segmentation": [],
                    "area": w * h,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
    
    return coco_data

def main():
    # 加载数据集划分
    split_data = load_json('DATA_folder/annotations/Train_Val_Test.json')
    
    # 加载标注数据
    annotation_data = load_json('DATA_folder/annotations/annotation_FSC147_384.json')
    
    # 设置图片目录
    images_dir = 'DATA_folder/images_384_VarV2'
    
    # 为每个划分生成标注文件
    for split_name in ['train', 'val', 'test']:
        image_list = split_data[split_name]
        coco_data = generate_coco_annotations(split_name, image_list, annotation_data, images_dir)
        output_file = f'DATA_folder/annotations/instances_{split_name}.json'
        save_json(coco_data, output_file)
        print(f"Generated {output_file}")

if __name__ == "__main__":
    main() 