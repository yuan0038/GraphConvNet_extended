# -*- coding: utf-8 -*-
# @Time : 2024/5/1 10:41 PM
# @Author : Caster Lee
# @Email : li949777411@gmail.com
# @File : plot.py
# @Project : 1paper_figure
import PIL
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from .utils.misc import get_filename
from .operation import img2patches
import  matplotlib.pyplot as plt
from matplotlib import colormaps

def plot_neighbors(image_path:str,layer_center_neighbors_list,patch_size,pad=1,resize_shape=(224,224),output_path:str=None,):
    """

    Args:
        image_path:
        center_neighbors_list:  list:[(center,[center's neighbors])]
        patch_size:
        pad:
        resize_shape:
        output_path:

    Returns:

    """
    class PLOT_NEIGHBORS_CONFIGS:
        node_color = 'gray'
        center_and_neighbor_colors = ['red', 'lightgreen', 'plum', 'orange', 'skyblue']

        nodes_size = 3
        center_size = 7

    assert pad>0,"when plot neighbors params 'pad' must >0"
    patches_list = img2patches(image_path,patch_size,pad,resize_shape=resize_shape)

    width_p_num= resize_shape[0] //patch_size
    height_p_num =resize_shape[1] //patch_size
    patch_size_w_ = patch_size + 2 * pad
    patch_size_h_ = patch_size + 2 * pad
    patches_num = len(patches_list)

    color_list = [PLOT_NEIGHBORS_CONFIGS.node_color] * patches_num
    x = patch_size_w_ // 2
    y = patch_size_h_ // 2
    point_size_list = [(x - PLOT_NEIGHBORS_CONFIGS.nodes_size, y - PLOT_NEIGHBORS_CONFIGS.nodes_size, x + PLOT_NEIGHBORS_CONFIGS.nodes_size, y + PLOT_NEIGHBORS_CONFIGS.nodes_size)] * patches_num

    for idx in range(patches_num):
       # print(layer_center_neighbors_list)
        for j, layer_center_neighbors in enumerate(layer_center_neighbors_list):

            layer_idx = layer_center_neighbors[0]
            center_idx = layer_center_neighbors[1]
            neighbors = layer_center_neighbors[2]
            if idx == center_idx:

                point_size_list[idx] = (x - PLOT_NEIGHBORS_CONFIGS.center_size, y - PLOT_NEIGHBORS_CONFIGS.center_size, x + PLOT_NEIGHBORS_CONFIGS.center_size, y + PLOT_NEIGHBORS_CONFIGS.center_size)
            if idx == center_idx or idx in neighbors:
                color_list[idx] = PLOT_NEIGHBORS_CONFIGS.center_and_neighbor_colors[j]
                break

    nodes_list=[]
    for i, one_patch in enumerate(patches_list):
        one_patch = Image.fromarray(one_patch)
        draw = ImageDraw.Draw(one_patch)

        draw.ellipse(point_size_list[i], fill=color_list[i])

        # one_patch.show()
        # one_patch.save(f"{out_path}/{i}.png")
        one_patch_np = np.array(one_patch)
        nodes_list.append(one_patch_np)

    output_image = np.hstack(nodes_list[0:width_p_num])
    for i in range(1, height_p_num):
        row = np.hstack(nodes_list[i * width_p_num: (i + 1) * width_p_num])
        output_image = np.vstack([output_image, row])

    output_image = Image.fromarray(output_image)
    output_image.save(output_path)

def plot_cam(image_path:str,cam_image,resize_shape,output_path):
    """

    Args:
        image_path:
        cam_image:  the cam image(numpy)
        resize_shape:
        output_path:

    Returns:

    """
    cmap = colormaps['jet']
    pil_image = Image.open(image_path)

    fig, ax = plt.subplots()
    ax.axis('off')  # removes the axis markers
    ax.imshow(pil_image.resize(resize_shape))

    overlay = to_pil_image((cam_image), mode='F').resize(resize_shape, resample=PIL.Image.BICUBIC)

    overlay = (255 * cmap(np.asarray(overlay))[:, :, :3]).astype(np.uint8)
    ax.imshow(overlay, alpha=0.4, interpolation='nearest', )
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close(fig)



