# -*- coding: utf-8 -*-
# @Time : 2024/4/30 6:11 PM
# @Author : Caster Lee
# @Email : li949777411@gmail.com
# @File : MyImage.py
# @Project : 1paper_figure
import PIL
from .utils.misc import  get_filename
class MyImage(PIL.Image.Image):
    def __init__(self,path,):
       self.path = path

    def get_image_name(self)->str:
        return get_filename(self.path)

def open(image_path:str):
    return MyImage(image_path)

