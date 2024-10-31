# -*- coding: utf-8 -*-
# @Time : 2024/5/2 10:48 AM
# @Author : Caster Lee
# @Email : li949777411@gmail.com
# @File : atom_fn.py
# @Project : 1paper_figure
import os.path

import torch
from typing import List

import PIL.Image as Image
import torchvision.transforms as pth_transforms
from  pathlib import Path

import yaml

from ..models.GraphConvNet import GraphConvNet

# 喂入模型的图像预处理
img_transform=pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])




##########  Basic function   ##########

def mkdir(path):
    """
    # 判断文件路径是否存在，不存在则创建
    Args:
        path:

    Returns:

    """
    if not Path(path).exists():
        Path.mkdir(Path(path),parents=True,exist_ok=True)
        print(f"\u2b50 Make a new directory:{path}")
    else:
       # print(f"\u26a0 ️Directory {path} has existed!")
            pass
def get_filename(path:str)->str:
    """
    给定文件路径，返回文件名字(不带格式)
    """

    return  path.rsplit('/',1)[-1].split('.')[0]

def get_images_path(images_dir: str = '', extensions:list=[".JPEG", ".png", ".jpg"])->List[str]:
    """
    #  递归获取所有文件夹的图片路径
    Args:
        images_dir:
        extensions:

    Returns:

    """
    images_path_list = []
    images_dir = Path(images_dir)
    if images_dir.is_dir():

        for iter_dir in list(images_dir.iterdir()):
            if iter_dir.is_dir():
               sub_images_path_list = get_images_path(str(iter_dir))
               images_path_list.extend(sub_images_path_list)
            else:
                for extension in extensions:
                    if extension in str(iter_dir):
                        image_path = str(iter_dir)

                        images_path_list.append(image_path)
    else:
        for extension in extensions:
            if extension in str(images_dir):
                image_path = str(images_dir)

                images_path_list.append(image_path)
    return images_path_list

def get_tensor(image_path: str):
    """load image and translate into tensor shape:(1,3,H,W)"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = img_transform(img).unsqueeze(0)
    return img_tensor

def get_target_layers(model:GraphConvNet,target_layers_idx:List):
    # 获取backbone 每一层图聚合后经过的第一个映射层（以此层进行热力图可视化）
    #imagenet 是model.backbone    RSNA是net
    return [model.backbone[idx].grapher.fc2[0] for idx in target_layers_idx]
########################################
