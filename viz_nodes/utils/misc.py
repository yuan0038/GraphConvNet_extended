# -*- coding: utf-8 -*-
# @Time : 2024/4/28 10:22 PM
# @Author : Caster Lee
# @Email : li949777411@gmail.com
# @File : misc.py
# @Project : MGVIG
import os
import pathlib
from pathlib import Path
from typing import List

from .atom_fn import img_transform, get_images_path, mkdir, get_filename
from PIL import Image


########## some utils function ##########
def mk_outputdir(image_path,output_root_dir,image_root=None):
    """
    给定图像路径和输出文件夹，根据图像名称创建图像输出文件夹, 如果给定image_root文件夹，则会根据image_root中的结构在output_root_dir中创建文件夹
    Args:
        image_path:
        output_root_dir:

    Returns:

    """
    if image_root is not None and pathlib.Path(image_root).is_dir() :
        relative_path = image_path.split(image_root)[1].rsplit('.',1)[0]
        if relative_path.startswith('/'):
            relative_path = relative_path.split('/', 1)[1]
        output_dir =os.path.join(output_root_dir,relative_path)

    else:
        image_name = get_filename(image_path)
        output_dir = os.path.join(output_root_dir,image_name)
    # print("output_dir",output_dir)
    # print(image_path.split(image_root)[1].rsplit('.',1)[0])
    # print(output_root_dir)
    mkdir(output_dir)
    return output_dir

def get_images(image_root,output_root_dir,tensor_format=False):
    """
        从指定路径中获取所有的图像pil image或tensor 以及对应的路径,并创建目标图像的输出文件夹
    Args:
        image_root:
        tensor_format: true will return tensor otherwise will return pil image
    Returns:

    """
    if os.path.isdir(image_root):
        images_path_list = sorted(get_images_path(image_root))
        if tensor_format:
            images_list = [img_transform(Image.open(image_path).convert('RGB')).unsqueeze(0) for image_path in images_path_list]
        else:
            images_list = [Image.open(image_path).convert('RGB') for image_path in images_path_list]
#
        # 不能用replace(image_root, output_root_dir),分隔符/稍微没注意，就会让文件夹名字连一块
        output_dir_list =[mk_outputdir(image_path,output_root_dir,image_root) for image_path in images_path_list]

        return images_list, images_path_list, output_dir_list
    else:
        image = Image.open(image_root).convert('RGB')
        if tensor_format:
            image = img_transform(image).unsqueeze(0)
        output_directory = mk_outputdir(image_root,output_root_dir)
        return [image], [image_root], [output_directory]








