# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from .gcn_lib import Grapher, act_layer

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


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD,
    ),
}


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


class Stem(nn.Module):
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


class Block(torch.nn.Module):
    def __init__(self,in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Block, self).__init__()
        self.grapher = Grapher(in_channels,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,r,n,drop_path,relative_pos)

        self.FFN = FFN(in_channels,in_channels*4,act=act,drop_path=drop_path)

    def forward(self,x,return_neighbors=False):

        x,neighbors = self.grapher(x)
        x= self.FFN(x)
        if return_neighbors:
            return neighbors
        return x


class ViG(nn.Module):
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
        self.stem = Stem(out_dim=channels, act=act)

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

    def get_neighbors(self, x:torch.Tensor, layer_idx:int, center_idx:int)->List:
        """
        :param x: input_tensor
        :param layer_idx: the layer's idx you want to visualize neighbors
        :param center_idx: the idx of node you want to visualize its neighbors
        :return: A list of the neighbor nodes' idx of a node(idx:node_idx)
        """
        x = self.stem(x) + self.pos_embed
        for i in range(self.n_blocks):
            if i == layer_idx:
                all_neighbors = self.backbone[i](x, return_neighbors=True)
                c_neighbors = all_neighbors[0, 0, center_idx, :]
                neighbors = c_neighbors.numpy().squeeze().tolist()
                return neighbors
            else:
                x = self.backbone[i](x)
@register_model
def vig_ti(pretrained=False, **kwargs):
    model = ViG(
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
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_s(pretrained=False, **kwargs):
    model = ViG(
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
def vig_b(pretrained=False, **kwargs):
    model = ViG(
        channels=640,
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

if __name__ == '__main__':
    model = timm.create_model('vig_ti')

    x = torch.ones((4, 3, 224, 224))
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1000 ** 2)
    print(len(model.get_diversity(x)))
    print(model(x))