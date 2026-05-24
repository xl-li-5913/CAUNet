import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, bias=False):
        padding = (kernel_size - 1) // 2
        gn_groups = 1 if out_planes < 32 else 32
        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),
            nn.GroupNorm(gn_groups, out_planes),
            nn.ReLU(inplace=True)
        )

class DDConv(nn.Module):
    def __init__(self,
                 channels,
                 DDConv_k=3,
                 stride=1,
                 dilation=1,
                 DDConv_posi_chans=4,
                 DDConv_inter_chans=64,
                 DDConv_inter_layers=1,
                 DDConv_Bias=False,
                 posi_map_hw=(1,1),   # initial posi_map size; will be interpolated to input size
                 **kwargs):
        super(DDConv, self).__init__()

        self.DDConv_k = int(DDConv_k)
        self.DDConv_k_square = self.DDConv_k * self.DDConv_k
        self.stride = int(stride)
        self.channels = int(channels)
        self.dilation = int(dilation)

        out_chans = self.DDConv_k_square * self.channels

        ph, pw = posi_map_hw
        self.posi_map = nn.Parameter(torch.ones(1, DDConv_posi_chans, ph, pw))

        # weight generation net
        self.weight_layers = self._make_layers(DDConv_posi_chans, DDConv_inter_chans,
                                               out_chans, DDConv_inter_layers)
        self.bias_layers = None
        if DDConv_Bias:
            self.bias_layers = self._make_layers(DDConv_posi_chans, DDConv_inter_chans,
                                                 self.channels, DDConv_inter_layers)

        self.stat_kernel = 3

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers):
        layers = [_ConvBlock(in_chans, inter_chans, kernel_size=3, stride=1, bias=False)]
        for i in range(max(0, num_inter_layers-1)):
            layers.append(_ConvBlock(inter_chans, inter_chans, kernel_size=3, stride=1, bias=False))
        # 最后一层 3x3 conv -> 输出 (out_chans, H, W)
        layers.append(nn.Conv2d(inter_chans, out_chans, kernel_size=3, padding=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, f"DDConv channels mismatch: {C} vs {self.channels}"

        # interpolate posi_map to input spatial size
        posi = F.interpolate(self.posi_map, size=(H, W), mode='bilinear', align_corners=False)  # (1, posi_ch, H, W)

        weight = self.weight_layers(posi)  # (1, out_chans = k^2 * channels, H, W)
        # reshape to (1, channels, k^2, H, W)
        weight = weight.view(1, self.channels, self.DDConv_k_square, H, W)

        # build unfold with dilation and padding to keep same size
        pad = self.dilation * (self.DDConv_k - 1) // 2
        unfold = nn.Unfold(kernel_size=self.DDConv_k, dilation=self.dilation, padding=pad, stride=self.stride)

        # x_unfold: (B, C * k^2, L) where L = output H'*W' (here should equal H*W if stride=1)
        x_unf = unfold(x)  # (B, C * k^2, L)
        L = x_unf.shape[-1]
        # reshape to (B, C, k^2, H_out, W_out)
        H_out = (H + 2*pad - self.dilation*(self.DDConv_k-1) - 1)//self.stride + 1
        W_out = (W + 2*pad - self.dilation*(self.DDConv_k-1) - 1)//self.stride + 1
        x_unf = x_unf.view(B, self.channels, self.DDConv_k_square, H_out, W_out)

        # multiply weight (1, C, k^2, H_out, W_out) * x_unf (B, C, k^2, H_out, W_out)
        out = (weight * x_unf).sum(dim=2)  # (B, C, H_out, W_out)

        if self.bias_layers is not None:
            bias = self.bias_layers(posi)  # (1, channels, H_out, W_out)
            out = out + bias

        local_mean = F.avg_pool2d(out, kernel_size=self.stat_kernel, stride=1, padding=self.stat_kernel // 2)
        local_sq_mean = F.avg_pool2d(out ** 2, kernel_size=self.stat_kernel, stride=1,
                                         padding=self.stat_kernel // 2)
        local_std = torch.sqrt(torch.clamp(local_sq_mean - local_mean ** 2, min=1e-6))

        alpha = torch.exp(-local_std)

        out_enhanced = out * alpha

        return out_enhanced
