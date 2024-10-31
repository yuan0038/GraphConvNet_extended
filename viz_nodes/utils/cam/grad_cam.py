# -*- coding: utf-8 -*-
# @Time : 2023/11/28 11:02 AM
# @Author : Caster Lee
# @Email : li949777411@gmail.com
# @File : grad_cam.py
# @Project : MGVIG
import numpy as np
import torch
import torch.nn as nn
from typing import List,Callable
from .activations_and_gradients import ActivationsAndGradients
class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


class Grad_CAM_DIY:
    def __init__(self,model: nn.Module,
                 target_layers:List[nn.Module],
                 reshape_transform:Callable = None,
                 ):
        self.model = model.eval()
        self.target_layers = target_layers

        # 用钩子获取梯度和输出
        self.activations_and_grads = ActivationsAndGradients(model,target_layers,reshape_transform=reshape_transform)


    def __call__(self,
                input_tensor: torch.Tensor,
                targets: List[int]=None,
                eigen_smooth: bool = False) -> np.ndarray:




        outputs = self.activations_and_grads(input_tensor)  # 输出：logits


        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        else:
            target_categories=targets
        targets = [ClassifierOutputTarget(
                category) for category in target_categories]



        self.model.zero_grad()
        loss = sum([target(output)  # target（output）返回的是指定类别的类分数
                    for target, output in zip(targets, outputs)])
        loss.backward(retain_graph=True)

        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]


        cam_list= []
        #print("\u2b50",grad.shape)
        for idx in range(len(self.target_layers)):
            activation = activations_list[idx]
            grad = grads_list[idx]
            alpha = torch.mean(torch.as_tensor(grad), dim=(2, 3), keepdim=True)
            cam = torch.relu(torch.sum(alpha * activation, dim=1, keepdim=False).squeeze()).detach()
            cam /= torch.max(cam)
            cam_list.append(cam)
        return cam_list
