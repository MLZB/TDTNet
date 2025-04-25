import math
from typing import Union, Tuple, Dict, List

import torch
import torchvision.models
from torch import nn, Tensor
from torch.nn import functional as f

class BaseConv(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 act: bool = True):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.norm_layer = nn.BatchNorm2d(out_channels)

        if act:
            self.act_layer = nn.ReLU(inplace=True)
        else:
            self.act_layer = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x:(B, in_channels, H, W)
        x = self.conv(x)        # x:(B, out_channels, H/S, W/S)
        x = self.norm_layer(x)  # x:(B, out_channels, H/S, W/S)
        x = self.act_layer(x)   # x:(B, out_channels, H/S, W/S)
        return x

class TDAttention(nn.Module):
    def __init__(self, channels: int, attn_drop: float = 0.1):
        super(TDAttention, self).__init__()
        self.channels = channels
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q, k, v:(B, C, H, W)
        h_attn = self._compute_h(q, k)                       # h_attn:(B, H, W, W)
        w_attn = self._compute_w(q, k)                       # w_attn:(B, W, H, H)
        result = self._compute_result(h_attn, w_attn, v)     # result:(B, C, H, W)
        return result

    def _compute_h(self, q: Tensor, k: Tensor) -> Tensor:
        # q, k:(B, C, H, W)
        h_q = q.transpose(1, 2).contiguous()                # h_q:(B, H, C, W)
        h_k = k.permute(0, 2, 3, 1).contiguous()            # h_k:(B, H, W, C)
        h_attn = (h_k @ h_q) / math.sqrt(self.channels)     # h_attn:(B, H, W, W)
        h_attn = f.softmax(h_attn, dim=-2)                  # h_attn:(B, H, W, W)
        h_attn = self.attn_drop(h_attn)                     # h_attn:(B, H, W, W)
        return h_attn

    def _compute_w(self, q: Tensor, k: Tensor) -> Tensor:
        # q, k:(B, C, H, W)
        w_q = q.transpose(1, 3).contiguous()                # w_q:(B, W, H, C)
        w_k = k.permute(0, 3, 1, 2).contiguous()            # w_k:(B, W, C, H)
        w_attn = (w_q @ w_k) / math.sqrt(self.channels)     # w_attn:(B, W, H, H)
        w_attn = f.softmax(w_attn, dim=-1)                  # w_attn:(B, W, H, H)
        w_attn = self.attn_drop(w_attn)                     # w_attn:(B, W, H, H)
        return w_attn

    def _compute_result(self, h_attn: Tensor, w_attn: Tensor, v: Tensor) -> Tensor:
        # h_attn:(B, H, W, W) w_attn:(B, W, H, H) v:(B, C, H, W)
        v_h = v.transpose(1, 2).contiguous()                # v_h:(B, H, C, W)
        v_h = v_h @ h_attn                                  # v_h:(B, H, C, W)
        v_h = v_h.transpose(1, 2).contiguous()              # v_h:(B, C, H, W)

        v_w = v.transpose(1, 3).contiguous()                # v_w:(B, W, H, C)
        v_w = w_attn @ v_w                                  # v_w:(B, W, H, C)
        v_w = v_w.transpose(1, 3).contiguous()              # v_w:(B, C, H, W)

        v = torch.cat((v_h, v_w), dim=1)                    # v:(B, 2C, H, W)
        return v

class TDTransformer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, attn_drop: float):
        super(TDTransformer, self).__init__()
        self.out_channels = out_channels
        self.conv = BaseConv(in_channels, 3 * out_channels, kernel_size=stride, stride=stride, padding=0, act=False)
        self.attn = TDAttention(out_channels, attn_drop=attn_drop)
        self.proj = BaseConv(2 * out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # x:(B, in_channels, H, W)
        qkv = self.conv(x)                                      # qkv:(B, 3*out_channels, H/S, W/S)
        q, k, v = torch.split(qkv, self.out_channels, dim=1)    # q、k:(B, out_channels, H/S, W/S)
        x = self.attn(q, k, v)                                  # x:(B, 2*out_channels, H/S, W/S)
        x = self.proj(x)                                        # x:(B, out_channels, H/S, W/S)
        return x

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, attn_drop: float):
        super(Block, self).__init__()
        self.tdt = TDTransformer(in_channels, in_channels, stride=1, attn_drop=attn_drop)
        self.conv = BaseConv(in_channels, out_channels, stride=stride, act=False)
        self.act_layer = nn.ReLU(inplace=True)

        # 残差结构
        if stride != 1 or in_channels != out_channels:
            self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.res = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x:(B, in_channels, H, W)
        res = self.res(x)               # res:(B, out_channels, H/S, W/S)
        x = self.tdt(x)                 # x:(B, in_channels, H, W)
        x = self.conv(x)                # x:(B, out_channels, H/S, W/S)
        x = x + res                     # x:(B, out_channels, H/S, W/S)
        x = self.act_layer(x)           # x:(B, out_channels, H/S, W/S)
        return x

# Two Dimension Transformer
class TDTNet5(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_block: List[int],
                 attn_drop: float = 0.1,
                 channels: int = 64
                 ):
        super(TDTNet5, self).__init__()
        self.in_channels = channels
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 56, 56))

        self.conv = BaseConv(3, channels, kernel_size=7, stride=2, padding=3, act=False)
        self.act = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(channels, 1, num_block[0], attn_drop)
        self.layer2 = self._make_layer(2 * channels, 2, num_block[1], attn_drop)
        self.layer3 = self._make_layer(4 * channels, 2, num_block[2], attn_drop)
        self.layer4 = self._make_layer(8 * channels, 2, num_block[3], attn_drop)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

        # 模型参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _make_layer(self,
                    channels: int,
                    stride: int,
                    num_block: int,
                    attn_drop: float):
        layer = nn.Sequential(*[Block(
            in_channels=self.in_channels if i == 0 else channels,
            out_channels=channels,
            stride=stride if i == 0 else 1,
            attn_drop=attn_drop
        ) for i in range(num_block)])

        self.in_channels = channels
        return layer

    def _forward_impl(self, x: Tensor):
        # x:(B, C, H, W) = (B, 3, 224, 224)
        x = self.conv(x)            # x:(B, in_channels, H/2, W/2)(B, C, 112, 112)
        x = self.max_pool(x)        # x:(B, C, 56, 56)

        x = x + self.pos_embed  # x:(B, C, 56, 56)
        x = self.layer1(x)          # x:(B, C, 56, 56)
        x = self.layer2(x)          # x:(B, 2C, 28, 28)
        x = self.layer3(x)          # x:(B, 4C, 14, 14)
        x = self.layer4(x)          # x:(B, 8C, 7, 7)

        x = self.avg_pool(x)        # x:(B, 8C, 1, 1)
        x = torch.flatten(x, 1)     # x:(B, 8C)
        x = self.fc(x)              # x:(B, num_classes)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def create_model(config_dict: Dict) -> TDTNet5:
    return TDTNet5(
        num_classes=config_dict['num_classes'],
        num_block=config_dict['num_block'],
        attn_drop=config_dict['attn_drop'],
        channels=config_dict['channels']
    )


def model_test(num_classes: int = 10):
    config_dict = {
        'num_classes': num_classes,
        'num_block': [2, 2, 2, 2],
        'attn_drop': 0.1,
        'channels': 64
    }
    return create_model(config_dict)


def tdt_small(num_classes: int = 10):
    config_dict = {
        'num_classes': num_classes,
        'num_block': [2, 2, 2, 2],
        'attn_drop': 0.1,
        'channels': 56
    }
    return create_model(config_dict)

def tdt_middle(num_classes: int = 10):
    config_dict = {
        'num_classes': num_classes,
        'num_block': [2, 3, 3, 2],
        'attn_drop': 0.1,
        'channels': 78
    }
    return create_model(config_dict)

def tdt_large(num_classes: int = 10):
    config_dict = {
        'num_classes': num_classes,
        'num_block': [3, 3, 3, 3],
        'attn_drop': 0.1,
        'channels': 90
    }
    return create_model(config_dict)

if __name__ == "__main__":
    model = model_test()
    device = torch.device('cuda')
    model.to(device)
    model.train()
    # model.eval()
    test_tensor = torch.randn([2, 3, 224, 224], device=device, dtype=torch.float)
    # test_tensor = torch.randint(-1, 1, [2, 3, 224, 224], device=device, dtype=torch.float)
    result_tensor = model(test_tensor)
    print(result_tensor)
