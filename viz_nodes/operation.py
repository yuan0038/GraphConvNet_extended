# -*- coding: utf-8 -*-
# @Time : 2024/4/30 5:02 PM
# @Author : Caster Lee
# @Email : li949777411@gmail.com
# @File : operation.py
# @Project : 1paper_figure
import os.path
from typing import List

import PIL.Image as Image
import numpy as np
import torch
import yaml
import json

from .models.GraphConvNet import GraphConvNet
from .utils import Grad_CAM_DIY
from .utils.atom_fn import mkdir, get_tensor,get_target_layers
from .utils.misc import mk_outputdir
from .utils.imagenet_label import  IMAGENET_LABEL_DICT

from collections import namedtuple

NEIGHBOR = namedtuple("Neighbor","layer_idx center_idx neighbors")


def img2patches(image_path:str,patch_size:int=16,pad:int=0,resize_shape=(224,224),output_root_dir:str=None,save_patchize_image=False)->List[Image.Image]:
    """
    Change an image into patches
    Args:
        image_path:
        patch_size:
        pad: to pad blank around each patch default:0
        resize_shape: resize image ,default:(224,224), None: not resize
        output_root_dir:if not None,will save all patches in 'output_root_dir',default:None
        save_patchize_image:default False if True,will save patchize image to output_path,pad must>0
    Returns:
            a list of patches(format: Image)
    """
    pil_image = Image.open(image_path).convert('RGB')
    if resize_shape is not None:
        pil_image=pil_image.resize(resize_shape)

    width, height =  pil_image.size

    w_pnum = width // patch_size
    h_pnum = height // patch_size

    patches_list = []
    for i in range(h_pnum):
        for j in range(w_pnum):
            box = (j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size)
            block = pil_image.crop(box)
            if pad>0:
                block=np.pad(block, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(255, 255))
            patches_list.append(block)
            if output_root_dir is not None:
                output_dir=mk_outputdir(image_path,output_root_dir)
                patches_output_dir = os.path.join(output_dir,'patches')
                mkdir(patches_output_dir)
                block.save(str(patches_output_dir) + '/' + f"patch_{i}_{j}.png")


    if output_root_dir is not None and save_patchize_image:
        assert isinstance(pad,int) and pad>0,"if save patchized image,param 'pad' must >0"

        patchize_image = np.hstack(patches_list[0:w_pnum])

        for i in range(1, h_pnum):
            row = np.hstack(patches_list[i * w_pnum: (i + 1) * w_pnum])
            patchize_image = np.vstack([patchize_image, row])

        pil_patchize_image = Image.fromarray(patchize_image)
        output_dir=mk_outputdir(image_path,output_root_dir)

        patchize_image_name = "patchized_image.png"

        pil_patchize_image.save(os.path.join(output_dir,patchize_image_name))

    return patches_list

def get_topk(image_path,model,topk=5,output_root_dir=None,image_root=None,label_list=None):
    """
    返回topk 字典，并将字典保存到指定文件夹
    Args:
        image_tensor:
        model:
        topk:

    Returns:
            the Dict of topk  {category name : probability}
    """
    model.eval()
    image_tensor = get_tensor(image_path)
    logits = model(image_tensor)
    probs = torch.softmax(logits, -1).squeeze(0).detach().cpu().numpy()
    topk_idxes = np.argsort(probs, axis=-1, ).tolist()[::-1][:topk]

    topk_dict = {}
    if label_list is not None:
        label_idxes = label_list
    else:
        print("Not given label list,and we will use ImageNet label")
        label_idxes = IMAGENET_LABEL_DICT

    for topk_idx in  topk_idxes:
        topk_idx = int(topk_idx)
        prob = probs[topk_idx]

        label_name =  label_idxes[topk_idx] + '_' + str(topk_idx)
        topk_dict[label_name] = f'{prob:.5f}'

    if output_root_dir is not None:
       if image_root is not None:
         output_dir=mk_outputdir(image_path,output_root_dir,image_root)

       else:
         output_dir = mk_outputdir(image_path, output_root_dir)


       topk_dict_name = f"top_{topk}.json"
       output_path = os.path.join(output_dir,topk_dict_name)

       with open (output_path,'w') as f:
            json.dump(topk_dict,f,indent=4)
    return  topk_dict

def get_max_cam_neighbors(image_path,model:GraphConvNet,target_layers_idx,return_neighbors_num:int=None):
    """
    return cam map list and  max grad cam  activation value patch's neighbors
    Args:
        image_tensor:
        cam_map:
        model:
        layer_idx:
        return_neighbors_num:

    Returns:
        List((layer_idx,center_idx,neighbors))
    """
    CAM = Grad_CAM_DIY(model,target_layers=get_target_layers(model,target_layers_idx))

    image_tensor = get_tensor(image_path)
    cam_list =  CAM(image_tensor)
    layer_center_neighbors_list=[]
    for layer_idx,cam_map in enumerate(cam_list):
        h, w = cam_map.shape
        cam_max, max_position_idx = torch.max(cam_map.view((1, h * w)), dim=1)

        neighbors=model.get_neighbors(image_tensor,target_layers_idx[layer_idx],max_position_idx)
        if return_neighbors_num is not None:
            neighbors=neighbors[:return_neighbors_num]
        layer_center_neighbors_list.append(NEIGHBOR(target_layers_idx[layer_idx],int(max_position_idx),neighbors))
    return cam_list,layer_center_neighbors_list