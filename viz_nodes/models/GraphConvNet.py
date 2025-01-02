# -*- coding: utf-8 -*-
# @Time : 2024/4/4 9:12 AM
# @Author : Caster Lee
# @Email : li949777411@gmail.com
# @File : GraphConvNet.py
# @Project : MGVIG
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential as Seq
from models.gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }
# default_cfgs = {
#     'gnn_patch16_224': _cfg(
#         crop_pct=0.9, input_size=(3, 224, 224),
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
#     ),
#
# }
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class Stem_isotropic(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),

            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x
class Stem_pyramid(nn.Module):
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out
class Downsample(nn.Module):
    """ Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x
class Block(nn.Module):
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None,
                 bias=True, stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False,
                 ):
        super(Block, self).__init__()

        self.grapher = Grapher(in_channels, kernel_size, dilation, conv, act, norm, bias, stochastic, epsilon, r, n,
                               drop_path, relative_pos)

        self.FFN = FFN(in_channels, in_channels * 4, act=act, drop_path=drop_path)
        self.res_block = Bottleneck(in_channels, in_channels // 4)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x,return_neighbors=False):
        _tmp = x
        x,neighbors= self.grapher(x)
        x = self.FFN(x)
        x = self.res_block(x)

        x = x + _tmp
        x = self.bn(x)
        if return_neighbors:
            return neighbors
        return x


class GraphConvNet(nn.Module):
    def __init__(self,
                 channels,n_blocks,k,gconv_type,use_dilation,n_classes,
                 act,norm,bias,epsilon,use_stochastic,dropout,drop_path_rate
                 ):
        '''

        :param channels: number of channels of deep features
        :param n_blocks: number of basic blocks in the backbone
        :param k: neighbor num
        :param gconv_type:  graph conv layer {edge, mr}
        :param use_dilation: use dilated knn or not
        :param n_classes: Dimension of head
        :param act: activation layer {relu, prelu, leakyrelu, gelu, hswish}
        :param norm:   batch or instance normalization {batch, instance}
        :param bias: bias of conv layer True or False
        :param epsilon: stochastic epsilon for gcn
        :param use_stochastic: stochastic for gcn, True or False
        :param dropout: dropout rate
        :param drop_path_rate:
        '''
        super().__init__()

        self.n_blocks = n_blocks
        self.stem = Stem_isotropic(out_dim=channels, act=act)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.n_blocks)]  # stochastic depth decay rule
        print('dpr', dpr)
        self.num_knn = [int(x.item()) for x in torch.linspace(k, 2 * k, self.n_blocks)]  # number of knn's k
        print('num_knn', self.num_knn)


        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        if use_dilation:
            self.max_dilation = 196 // max(self.num_knn)
            self.dilations = [min(i // 4 + 1, self.max_dilation) for i in range(self.n_blocks)]


            print("dilation", self.dilations)
            self.backbone = Seq(*[Block(channels, self.num_knn[i], self.dilations[i], gconv_type, act, norm,
                                            bias, use_stochastic, epsilon, 1, drop_path=dpr[i])
                                  for i in range(self.n_blocks)])
        else:
            print("dilation", [1] * self.n_blocks)
            self.backbone = Seq(*[Block(channels, self.num_knn[i], 1, gconv_type, act, norm,
                                            bias, use_stochastic, epsilon, 1, drop_path=dpr[i]
                                            ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(dropout),
                              nn.Conv2d(1024, n_classes, 1, bias=True)) if n_classes>0 else nn.Identity()
        self.model_init()

    def interpolate_pos_encoding(self, x):
        w, h = x.shape[2], x.shape[3]
        p_w, p_h = self.pos_embed.shape[2], self.pos_embed.shape[3]

        if w * h == p_w * p_h and w == h:
            return self.pos_embed

        w0 = w
        h0 = h
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            self.pos_embed,
            scale_factor=(w0 / p_w, h0 / p_h),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        return patch_pos_embed

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        B, C, H, W = inputs.shape

        x = self.stem(inputs)

        x = x + self.interpolate_pos_encoding(x)

        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

    def get_neighbors(self, x, layer_idx, center_idx):

        x = self.stem(x) + self.pos_embed
        for i in range(len(self.backbone)):
            if i == layer_idx:
                #raw_dilation = self.dilations[layer_idx]
                #self.backbone[layer_idx].grapher.graph_conv.dilated_knn_graph.dilation = 1
                all_neighbors = self.backbone[i](x, return_neighbors=True)

                c_neighbors = all_neighbors[0, 0, center_idx, :]
                neighbors = c_neighbors.numpy().tolist()[0]
                #self.backbone[layer_idx].grapher.graph_conv.dilated_knn_graph.dilation=raw_dilation
                return neighbors
            else:
                x = self.backbone[i](x)

@register_model
def graphconvnet_ti(pretrained=False,**kwargs):
    model = GraphConvNet(
        channels=192,
        n_blocks=12,
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

    model.default_cfg = _cfg()

    return model

@register_model
def graphconvnet_s(pretrained=False,**kwargs):
    model = GraphConvNet(
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
    model.default_cfg = _cfg()
    return model




class GraphConvNet_p(torch.nn.Module):
    def __init__(self, k,gconv,channels,blocks,n_classes,act,norm,bias,epsilon,use_stochastic,dropout,drop_path):
        super().__init__()

        self.n_blocks = sum(blocks)
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        print(num_knn)
        max_dilation = 49 // max(num_knn)

        self.stem = Stem_pyramid(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))
        HW = 224 // 4 * 224 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(
                        *[Block(channels[i],num_knn[idx], min(idx // 4 + 1, max_dilation), gconv, act, norm,
                                bias, use_stochastic, epsilon, reduce_ratios[i],n=HW, drop_path=dpr[idx],
                                relative_pos=True)])
                    ]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(dropout),
                              nn.Conv2d(1024, n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def interpolate_pos_encoding(self, x):
        w, h = x.shape[2], x.shape[3]
        p_w, p_h = self.pos_embed.shape[2], self.pos_embed.shape[3]

        if w * h == p_w * p_h and w == h:
            return self.pos_embed

        w0 = w
        h0 = h
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            self.pos_embed,
            scale_factor=(w0 / p_w, h0 / p_h),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        return patch_pos_embed

    def forward(self, inputs):
        B, C, H, W = inputs.shape

        x = self.stem(inputs)

        x = x + self.interpolate_pos_encoding(x)

        for i in range(len(self.backbone)):

            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

@register_model
def graphconvnetp_ti(pretrained=False,**kwargs):
    model = GraphConvNet_p(
        k = 9,  # neighbor num (default:9)
        gconv = 'mr',  # graph conv layer {edge, mr}
        channels=[48, 96, 240, 384],  # number of channels of deep features
        blocks = [2, 2, 6, 2],  # number of basic blocks in the backbone
        n_classes=1000,  # Dimension of out_channels
        act = 'gelu',  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        norm = 'batch',  # batch or instance normalization {batch, instance}
        bias=True,  # bias of conv layer True or False
        epsilon=0.2,  # stochastic epsilon for gcn
        use_stochastic=False,# stochastic for gcn, True or False
        dropout = 0.0,  # dropout rate
        drop_path = 0.0,
    )
    model.default_cfg= _cfg()
    return model
@register_model
def graphconvnetp_s(pretrained=False,**kwargs):
    model = GraphConvNet_p(
        k=9,  # neighbor num (default:9)
        gconv='mr',  # graph conv layer {edge, mr}
        channels=[80, 160, 400, 640],  # number of channels of deep features
        blocks=[2, 2, 6, 2],  # number of basic blocks in the backbone
        n_classes=1000,  # Dimension of out_channels
        act='gelu',  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        norm='batch',  # batch or instance normalization {batch, instance}
        bias=True,  # bias of conv layer True or False
        epsilon=0.2,  # stochastic epsilon for gcn
        use_stochastic=False,  # stochastic for gcn, True or False
        dropout=0.0,  # dropout rate
        drop_path=0.0,
    )
    model.default_cfg = _cfg()
    return model

if __name__ == '__main__':
    x = torch.randn((4,3,224,224))
    model = graphconvnet_ti()
    y = model.get_neighbors(x,11,0)
