import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from model.DDConv import DDConv


__all__ = ['ResNet',
           'wide_resnet50_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

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
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
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
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feature_a = self.layer1(x)
        feature_b = self.layer2(feature_a)
        feature_c = self.layer3(feature_b)

        return [feature_a, feature_b, feature_c]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


class AttnBasicBlock(nn.Module):
    expansion: int = 1

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
            attention: bool = True,
    ) -> None:
        super(AttnBasicBlock, self).__init__()
        self.attention = attention
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttnBottleneck(nn.Module):
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
            attention: bool = True,
    ) -> None:
        super(AttnBottleneck, self).__init__()
        self.attention = attention
        # print("Attention:",self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
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
        out = self.relu(out)

        return out

class PADAA(nn.Module):

    def __init__(self, in_ch: int, kernel_size: int, dilation: int, attn_dim: int = None, attn_dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.bn0 = nn.BatchNorm2d(in_ch)


        self.dw1 = DDConv(channels=in_ch, DDConv_k=kernel_size, stride=1, dilation=1,
                          DDConv_posi_chans=4, DDConv_inter_chans=32, DDConv_inter_layers=1,
                          posi_map_hw=(1, 1))
        self.dw2 = DDConv(channels=in_ch, DDConv_k=kernel_size, stride=1, dilation=2,
                          DDConv_posi_chans=4, DDConv_inter_chans=32, DDConv_inter_layers=1,
                          posi_map_hw=(1, 1))
        self.dw3 = DDConv(channels=in_ch, DDConv_k=kernel_size, stride=1, dilation=4,
                          DDConv_posi_chans=4, DDConv_inter_chans=32, DDConv_inter_layers=1,
                          posi_map_hw=(1, 1))

        # pointwise after each depthwise to mix channels
        self.pw1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.pw3 = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        # attention dim
        attn_dim = attn_dim or max(16, in_ch // 2)
        self.attn_dim = attn_dim
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

        # Q from global semantic of x0
        self.q_proj = nn.Sequential(
            nn.Linear(in_ch, attn_dim, bias=True),
            nn.LayerNorm(attn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity(),
            nn.Linear(attn_dim, attn_dim, bias=True),
        )

        # K/V branches (shared structure but separate params)
        def kv_block():
            return nn.Sequential(
                nn.Linear(in_ch, attn_dim, bias=True),
                nn.LayerNorm(attn_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity(),
                nn.Linear(attn_dim, attn_dim, bias=True),
            )

        self.k1 = kv_block()
        self.k2 = kv_block()
        self.k3 = kv_block()

        self.v1 = kv_block()
        self.v2 = kv_block()
        self.v3 = kv_block()

        self.pw = nn.Conv2d(attn_dim, in_ch, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(in_ch)
        self.relu_out = nn.ReLU(inplace=True)


        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, max(8, in_ch // 16), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_ch // 16), in_ch, kernel_size=1, bias=True),
            nn.Sigmoid()
        )


        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(inplace=True)
        )


        self._init_weights()

    def _init_weights(self):
        # convs: kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                try:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                except Exception:
                    pass

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert C == self.in_ch, f"expected in_ch={self.in_ch}, got {C}"


        x0 = self.bn0(x)


        x1 = self.dw1(x0)
        x1 = self.pw1(x1)
        x1 = self.relu(self.bn1(x1))

        x2 = self.dw2(x0)
        x2 = self.pw2(x2)
        x2 = self.relu(self.bn2(x2))

        x3 = self.dw3(x0)
        x3 = self.pw3(x3)
        x3 = self.relu(self.bn3(x3))


        g1 = F.adaptive_avg_pool2d(x1, 1).reshape(B, C)  # [B, C]
        g2 = F.adaptive_avg_pool2d(x2, 1).reshape(B, C)
        g3 = F.adaptive_avg_pool2d(x3, 1).reshape(B, C)
        g0 = F.adaptive_avg_pool2d(x0, 1).reshape(B, C)


        q = self.q_proj(g0)                    # [B, D]
        k1 = self.k1(g1); k2 = self.k2(g2); k3 = self.k3(g3)   # [B, D]
        v1 = self.v1(g1); v2 = self.v2(g2); v3 = self.v3(g3)   # [B, D]

        # optional dropout on q
        q = self.attn_dropout(q)


        d = q.shape[-1]
        scale = max(1.0, d ** 0.5)

        s1 = torch.sum(q * k1, dim=-1) / scale  # [B]
        s2 = torch.sum(q * k2, dim=-1) / scale
        s3 = torch.sum(q * k3, dim=-1) / scale
        scores = torch.stack([s1, s2, s3], dim=1)  # [B, 3]
        alpha = F.softmax(scores, dim=1)            # [B, 3]


        v = alpha[:, 0:1] * v1 + alpha[:, 1:2] * v2 + alpha[:, 2:3] * v3  # broadcasting -> [B, D]
        v_sp = v.view(B, -1, 1, 1)                 # [B, D, 1, 1]
        f_sp = v_sp.expand(-1, -1, H, W)           # [B, D, H, W]

        fused = self.pw(f_sp)                      # [B, in_ch, H, W]
        fused = self.relu_out(self.bn_out(fused))  # [B, in_ch, H, W]

        # 8. gate + residual
        gate = self.gate(fused)                    # [B, in_ch, 1, 1]
        gated = fused * gate                       # [B, in_ch, H, W] (broadcast)
        residual = gated + x

        out = self.expand(residual)                # [B, in_ch*2, H/2, W/2]
        return out


class PADAALayer(nn.Module):
    def __init__(self, block: Type[Union[object, object]], layers: int, **kwargs):
        super().__init__()
        be = block.expansion

        self.conv1 = conv3x3(64 * be, 128 * be, stride=2)
        self.bn1 = nn.BatchNorm2d(128 * be)
        self.conv2 = conv3x3(128 * be, 256 * be, stride=2)
        self.bn2 = nn.BatchNorm2d(256 * be)
        self.conv3 = conv3x3(128 * be, 256 * be, stride=2)
        self.bn3 = nn.BatchNorm2d(256 * be)
        self.relu = nn.ReLU(inplace=True)


        self.channel_reduce = nn.Sequential(
            nn.Conv2d(3072, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.gated_attn = PADAA(
            in_ch=1024,
            kernel_size=3,
            dilation=1,
            attn_dim=512,
            attn_dropout=0.0
        )

        # 权重初始化（额外）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                try:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                except Exception:
                    pass

    def forward(self, x: List[Tensor]) -> Tensor:
        x0, x1, x2 = x[0], x[1], x[2]

        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x0))))))
        l2 = self.relu(self.bn3(self.conv3(x1)))
        feature = torch.cat([l1, l2, x2], dim=1)

        feature = self.channel_reduce(feature)    # [B, 1024, H', W']

        out = self.gated_attn(feature)
        return out

def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs), PADAALayer(Bottleneck, 3, **kwargs)




