import torch
from torch import nn
from einops import rearrange
from ultralytics.nn.modules.conv import autopad

class SimAMWithSlicing(nn.Module):
    """升级版：针对小目标优化的细粒度网格切片无参数注意力"""
    def __init__(self, e_lambda=1e-4, grid_size=8): # 默认 grid_size=8 适配无人机
        super(SimAMWithSlicing, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.grid = grid_size

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 自适应 Padding 确保能被 grid 整除，防止报错
        pad_h = (self.grid - H % self.grid) % self.grid
        pad_w = (self.grid - W % self.grid) % self.grid
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
            
        _, _, H_pad, W_pad = x.shape
        
        # 将特征图划分为细粒度网格，增强局部对比度
        x_grids = rearrange(x, 'b c (g_h h) (g_w w) -> (b g_h g_w) c h w', 
                            g_h=self.grid, g_w=self.grid)

        n = x_grids.shape[2] * x_grids.shape[3] - 1
        x_minus_mu_square = (x_grids - x_grids.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        enhanced_grids = x_grids * self.activation(y)
        
        # 还原回原始形状
        out = rearrange(enhanced_grids, '(b g_h g_w) c h w -> b c (g_h h) (g_w w)', 
                        b=B, g_h=self.grid, g_w=self.grid)
        
        # 裁掉之前加的 padding
        return out[:, :, :H, :W] if (pad_h > 0 or pad_w > 0) else out


class Conv_SWS(nn.Module):
    """标准的卷积 + 升级版切片 SimAM 增强"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  
        super(Conv_SWS, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = SimAMWithSlicing() # 这里调用升级版的细粒度 SimAM

    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))


class SpatialAttention_CGA(nn.Module):
    """升级版：空洞空间注意，通过 Dilation 2 扩大感受野，捕捉上下文"""
    def __init__(self):
        super(SpatialAttention_CGA, self).__init__()
        # 使用 dilation=2, padding 从 3 改为 6 以保持特征图尺寸不变
        self.sa = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=6, dilation=2, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention_CGA(nn.Module):
    """升级版：双池化通道注意 (Avg捕捉全局背景 + Max捕捉小目标显著性)"""
    def __init__(self, dim, reduction=8):
        super(ChannelAttention_CGA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 新增 MaxPool
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid() # 修复：加入 Sigmoid 限制值域

    def forward(self, x):
        avg_out = self.ca(self.avg_pool(x))
        max_out = self.ca(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

    
class PixelAttention_CGA(nn.Module):
    """原版逻辑保留，用于生成精细的像素级融合权重"""
    def __init__(self, dim):
        super(PixelAttention_CGA, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class SCA_Fusion(nn.Module):
    """
    主融合类：类名和参数与原版完全一致，无需修改 YAML。
    内部集成了 UAV 定制优化：细粒度 SimAM + 双池化 CA + 空洞 SA
    """
    def __init__(self, dim, reduction=8):
        super(SCA_Fusion, self).__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        # self.conv = nn.Conv2d(dim, dim, 1, bias=True) # 原代码注释掉了，这里不作更改
        self.sigmoid = nn.Sigmoid()
        self.SWS = Conv_SWS(dim, dim) # 内部会调用升级版的 SimAM

    def forward(self, data):
        x, y = data
        initial = x + y
        
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        
        # 优化：归一化加权融合，避免 initial+x+y 导致数值膨胀
        result = pattn2 * x + (1 - pattn2) * y 
        
        result = self.SWS(result)
        return result