# -*- coding: utf-8 -*-
# @Time : 2024/5/7 7:54 PM
# @Author : Caster Lee
# @Email : li949777411@gmail.com
# @File : GraphConvNet_rsna.py
# @Project : GraphConvNet-main
import timm
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from torch.nn import Sequential as Seq
from .GraphConvNet import GraphConvNet

import torch.nn as nn

from .vig import ViG


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 3, 'input_size': (3, 512, 512), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
    ),
    'gnn_patch16_384': _cfg(
        crop_pct=0.9, input_size=(3, 384, 384),
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
    ),

}

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Conv2d(input_dim // 2 ** l, input_dim // 2 ** (l + 1), kernel_size=1,bias=True) for l in range(L)]
        list_FC_layers.append(nn.Conv2d(input_dim // 2 ** L, output_dim,kernel_size=1, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

@register_model
def graphconvnet_ti_backbone(pretrain=False,**kwargs):
    model = GraphConvNet(
        channels=192,
        n_blocks=12,
        k=9,
        gconv_type='mr',
        use_dilation=True,
        n_classes=0,
        act='gelu',
        norm='batch',
        bias=True,
        epsilon=0.2,
        use_stochastic=False,
        dropout=0,
        drop_path_rate=0.0,
    )
    return model.backbone
@register_model
def graphconvnet_s_backbone(pretrain=False,**kwargs):
        model = GraphConvNet(
            channels=320,
            n_blocks=16,
            k=9,
            gconv_type='mr',
            use_dilation=True,
            n_classes=0,
            act='gelu',
            norm='batch',
            bias=True,
            epsilon=0.2,
            use_stochastic=False,
            dropout=0,
            drop_path_rate=0.0,
        )
        return model
import torch

class GraphConvNet_RSNA(GraphConvNet):
        def __init__(self, freeze_vig_layer_num,load_imagenet_pretrained,channels,n_blocks,k,gconv_type,use_dilation,n_classes,
                 act,norm,bias,epsilon,use_stochastic,dropout,drop_path_rate):

            super().__init__(channels,n_blocks,k,gconv_type,use_dilation,n_classes,
                 act,norm,bias,epsilon,use_stochastic,dropout,drop_path_rate)


            # assert backbone in ['graphconvnet_ti','graphconvnet_s','graphconvnet_b']
            # out_channel={'graphconvnet_ti':192,'graphconvnet_s':320,'graphconvnet_b':640}

            def model_name_to_ckpt_path(backbone_name):
                backbone_dict = {
                    'graphconvnet_ti': './ckpt/GraphConvNet_Ti_77_1.pth.tar',
                    'graphconvnet_s': './ckpt/GraphConvNet_S_82.pth.tar',
                    'graphconvnet_b': './ckpt/GraphConvNet_B_83_2.pth.tar'
                }
                return backbone_dict[backbone_name]
            if load_imagenet_pretrained:
                if channels == 192:
                    backbone_name = 'graphconvnet_ti'
                elif channels == 320:
                    backbone_name = 'graphconvnet_s'
                elif channels == 640:
                    backbone_name = 'graphconvnet_b'
                state_dict = torch.load(model_name_to_ckpt_path(backbone_name),map_location='cpu')['state_dict']
                missing_keys, unexpected_keys = self.load_state_dict(state_dict,strict=False)
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)#5
            self.prediction =  MLPReadout(channels,output_dim=3,L=5)


                # state_dict = torch.load(model_name_to_ckpt_path(backbone), map_location='cpu')['state_dict']
                # missing_keys, unexpected_keys = self.net.load_state_dict(state_dict, strict=False)
                # print("missing_keys: ", missing_keys)
                # print("unexpected_keys: ", unexpected_keys)
            self.init_head(self.prediction)
            self.freeze_first_x_layers(freeze_vig_layer_num=freeze_vig_layer_num)
        def freeze_first_x_layers(self,freeze_vig_layer_num=0):


            for params in self.stem.parameters():
                if params.requires_grad:
                    params.requires_grad=False
            if freeze_vig_layer_num != 0:
                for params in self.backbone[:freeze_vig_layer_num].parameters():
                        if params.requires_grad:
                            params.requires_grad = False
                #for params in self.net.stem.parameters():
        def init_head(self,module:nn.Module):
            for m in module.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)

                    m.weight.requires_grad = True
                    if m.bias is not None:
                        m.bias.data.zero_()
                        m.bias.requires_grad = True

@register_model
def graphconvnet_ti_rsna(pretrained=False,freeze_layer_num=0,load_imagenet_pretrained=True,**kwargs):
    model = GraphConvNet_RSNA(
        freeze_vig_layer_num=freeze_layer_num,
        load_imagenet_pretrained=load_imagenet_pretrained,
        channels=192,
        n_blocks=12,
        k=9,
        gconv_type='mr',
        use_dilation=True,
        n_classes=0,
        act='gelu',
        norm='batch',
        bias=True,
        epsilon=0.2,
        use_stochastic=False,
        dropout=0,
        drop_path_rate=0.0,
    )
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model
@register_model
def graphconvnet_s_rsna(pretrained=False,freeze_layer_num=0,load_imagenet_pretrained=True,**kwargs):
    model = GraphConvNet_RSNA(
        freeze_vig_layer_num=freeze_layer_num,
        load_imagenet_pretrained=load_imagenet_pretrained,
        channels=320,
        n_blocks=16,
        k=9,
        gconv_type='mr',
        use_dilation=True,
        n_classes=1000,
        act='gelu',
        norm='batch',
        bias=True,
        epsilon=0.2,
        use_stochastic=False,
        dropout=0,
        drop_path_rate=0.0,
                              )
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def graphconvnet_ti_rsna_384(pretrained=False,freeze_layer_num=0,**kwargs):
    model = GraphConvNet_RSNA(backbone='graphconvnet_ti',freeze_vig_layer_num=freeze_layer_num)
    model.default_cfg = default_cfgs['gnn_patch16_384']
    return model

class ViG_RSNA(ViG):
        def __init__(self, freeze_vig_layer_num, load_imagenet_pretrained, channels, n_blocks, k, gconv_type,
                     use_dilation, n_classes,act, norm, bias, epsilon, use_stochastic, dropout, drop_path_rate):

            super().__init__(channels, n_blocks, k, gconv_type, use_dilation, n_classes,
                             act, norm, bias, epsilon, use_stochastic, dropout, drop_path_rate)

            # assert backbone in ['graphconvnet_ti','graphconvnet_s','graphconvnet_b']
            # out_channel={'graphconvnet_ti':192,'graphconvnet_s':320,'graphconvnet_b':640}

            def model_name_to_ckpt_path(backbone_name):
                backbone_dict = {
                    'vig_ti': './ckpt/ViG_Ti_75_1.pth.tar',
                    'graphconvnet_s': './ckpt/GraphConvNet_S_82.pth.tar',
                    'graphconvnet_b': './ckpt/GraphConvNet_B_83_2.pth.tar'
                }
                return backbone_dict[backbone_name]

            if load_imagenet_pretrained:
                if channels == 192:
                    backbone_name = 'vig_ti'
                elif channels == 320:
                    backbone_name = 'graphconvnet_s'
                elif channels == 640:
                    backbone_name = 'graphconvnet_b'
                state_dict = torch.load(model_name_to_ckpt_path(backbone_name), map_location='cpu')['state_dict']
                missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)  # 5
            self.prediction = MLPReadout(channels, output_dim=3, L=5)

            # state_dict = torch.load(model_name_to_ckpt_path(backbone), map_location='cpu')['state_dict']
            # missing_keys, unexpected_keys = self.net.load_state_dict(state_dict, strict=False)
            # print("missing_keys: ", missing_keys)
            # print("unexpected_keys: ", unexpected_keys)
            self.init_head(self.prediction)
            self.freeze_first_x_layers(freeze_vig_layer_num=freeze_vig_layer_num)

        def freeze_first_x_layers(self, freeze_vig_layer_num=0):

            for params in self.stem.parameters():
                if params.requires_grad:
                    params.requires_grad = False
            if freeze_vig_layer_num != 0:
                for params in self.backbone[:freeze_vig_layer_num].parameters():
                    if params.requires_grad:
                        params.requires_grad = False
                # for params in self.net.stem.parameters():

        def init_head(self, module: nn.Module):
            for m in module.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)

                    m.weight.requires_grad = True
                    if m.bias is not None:
                        m.bias.data.zero_()
                        m.bias.requires_grad = True

@register_model
def vig_ti_rsna(pretrained=False,freeze_layer_num=0,load_imagenet_pretrained=True,**kwargs):
    model = ViG_RSNA(
        freeze_vig_layer_num=freeze_layer_num,
        load_imagenet_pretrained=load_imagenet_pretrained,
        channels=192,
        n_blocks=12,
        k=9,
        gconv_type='mr',
        use_dilation=True,
        n_classes=0,
        act='gelu',
        norm='batch',
        bias=True,
        epsilon=0.2,
        use_stochastic=False,
        dropout=0,
        drop_path_rate=0.0,
    )
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model