# -*- coding: utf-8 -*-
# @Time : 2024/4/28 10:20 PM
# @Author : Caster Lee
# @Email : li949777411@gmail.com
# @File : viz.py
# @Project : MGVIG
import os.path

import PIL
import numpy as np
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

from typing import List
import PIL.Image as Image

from .plot import plot_cam, plot_neighbors
from .utils.atom_fn import get_target_layers
from .operation import get_topk,get_max_cam_neighbors
from .utils.misc import get_images
from .utils.cam.grad_cam import Grad_CAM_DIY
from matplotlib import colormaps, pyplot as plt




class BaseViz:
    # 输入图像根目录 -> 获取所有图像，处理后，保存成更高的分辨率
    def __init__(self,image_root,output_root_dir,hr_ratio):
        self.image_root = image_root
        self.output_root_dir=output_root_dir
        self.hr_ratio = hr_ratio

    def process(self):
        raise Exception("Not Implemented")




class Viz_CAM(BaseViz):

    def __init__(self,image_root,output_root_dir,hr_ratio,model, target_layers_idx,**kwargs):
        super(Viz_CAM, self).__init__(image_root,output_root_dir,hr_ratio)
        self.model = model
        self.target_layers_idx = target_layers_idx
        self.target_layers= get_target_layers(model,target_layers_idx)
        self.cam = Grad_CAM_DIY(model,self.target_layers)

        self.resize_shape = (224*hr_ratio,224*hr_ratio)
        self.patch_size = 16*hr_ratio
        self.label_list = kwargs['label_list'] if 'label_list' in kwargs else None
        #self.patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else None
    def process(self,viz_neighbors=9,save_topk=1):
        '''

        Args:
            viz_neighbors:  >0 will visualize the 'viz_neighbors'Nearest Neighbors of max grad cam value :

            save_topk:  >0 will save topk dict(category:probability)

        Returns:

        '''
        image_tensor_list, image_path_list, output_dir_list=get_images(image_root=self.image_root,output_root_dir=self.output_root_dir,tensor_format=True)
        pil_image_list,_,_=get_images(image_root=self.image_root,output_root_dir=self.output_root_dir,tensor_format=False)
        for idx,image_tensor in enumerate(image_tensor_list):
                image_path = image_path_list[idx]
                if save_topk>0:
                    print(image_path)
                    print("\u2b50"*5,f"top {save_topk}","\u2b50"*5)
                    print((get_topk(image_path_list[idx],self.model,topk=save_topk,output_root_dir=self.output_root_dir,image_root=self.image_root,label_list=self.label_list)))


                cam_list,layer_center_neighbors_list = get_max_cam_neighbors(image_path, self.model, target_layers_idx=
                    self.target_layers_idx, return_neighbors_num=viz_neighbors)
                for layer_idx in range(len(layer_center_neighbors_list)):
                    if viz_neighbors>0:
                        assert self.patch_size is not None,"'patch_size' can not be None if you want to viz neighbors"


                        layer_center_neighbors=layer_center_neighbors_list[layer_idx]
                        center_idx=layer_center_neighbors[1]
                        print(layer_center_neighbors)
                        max_cam_neighbors_output_path = os.path.join(output_dir_list[idx],f"layer_{self.target_layers_idx[layer_idx]+1}_center_{center_idx}_neighbors.png")
                        plot_neighbors( image_path,[layer_center_neighbors],patch_size=self.patch_size,pad=1,resize_shape=self.resize_shape,output_path=max_cam_neighbors_output_path)

                    cam_output_path = os.path.join(output_dir_list[idx],f"layer_{self.target_layers_idx[layer_idx]+1}_cam.png")
                    plot_cam( image_path, cam_list[layer_idx], resize_shape=self.resize_shape,output_path= cam_output_path )










