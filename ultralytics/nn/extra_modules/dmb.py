import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from itertools import repeat
import collections.abc
import math
from functools import partial
from ..modules.conv import Conv, autopad 

__all__ = ['DMB', 'Warehouse_Manager', 'MSD_KWBlock']

def parse(x, n):
    if isinstance(x, collections.abc.Iterable):
        if len(x) == 1:
            return list(repeat(x[0], n))
        elif len(x) == n:
            return x
        else:
            raise ValueError('length of x should be 1 or n')
    else:
        return list(repeat(x, n))

class UAV_Attention(nn.Module):
    def __init__(self, in_planes, reduction, num_static_cell, num_local_mixture, norm_layer=nn.BatchNorm1d,
                 cell_num_ratio=1.0, nonlocal_basis_ratio=1.0, start_cell_idx=None):
        super(UAV_Attention, self).__init__()
        hidden_planes = max(int(in_planes * reduction), 16)
        self.kw_planes_per_mixture = num_static_cell + 1
        self.num_local_mixture = num_local_mixture
        self.kw_planes = self.kw_planes_per_mixture * num_local_mixture

        self.num_local_cell = int(cell_num_ratio * num_local_mixture)
        self.num_nonlocal_cell = num_static_cell - self.num_local_cell
        self.start_cell_idx = start_cell_idx

        self.fc1 = nn.Linear(in_planes * 2, hidden_planes, bias=(norm_layer is not nn.BatchNorm1d))
        self.norm1 = norm_layer(hidden_planes)
        self.act1 = nn.ReLU(inplace=True)

        if nonlocal_basis_ratio >= 1.0:
            self.map_to_cell = nn.Identity()
            self.fc2 = nn.Linear(hidden_planes, self.kw_planes, bias=True)
        else:
            self.map_to_cell = self.map_to_cell_basis
            self.num_basis = max(int(self.num_nonlocal_cell * nonlocal_basis_ratio), 16)
            self.fc2 = nn.Linear(hidden_planes, (self.num_local_cell + self.num_basis + 1) * num_local_mixture, bias=False)
            self.fc3 = nn.Linear(self.num_basis, self.num_nonlocal_cell, bias=False)
            self.basis_bias = nn.Parameter(torch.zeros([self.kw_planes]), requires_grad=True).float()

        self.temp_bias = torch.zeros([self.kw_planes], requires_grad=False).float()
        self.temp_value = 0
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temp_value):
        self.temp_value = temp_value

    def init_temperature(self, start_cell_idx, num_cell_per_mixture):
        if num_cell_per_mixture >= 1.0:
            num_cell_per_mixture = int(num_cell_per_mixture)
            for idx in range(self.num_local_mixture):
                assigned_kernel_idx = int(idx * self.kw_planes_per_mixture + start_cell_idx)
                self.temp_bias[assigned_kernel_idx] = 1
                start_cell_idx += num_cell_per_mixture
            return start_cell_idx
        else:
            num_mixture_per_cell = int(1.0 / num_cell_per_mixture)
            for idx in range(self.num_local_mixture):
                if idx % num_mixture_per_cell == (idx // num_mixture_per_cell) % num_mixture_per_cell:
                    assigned_kernel_idx = int(idx * self.kw_planes_per_mixture + start_cell_idx)
                    self.temp_bias[assigned_kernel_idx] = 1
                    start_cell_idx += 1
                else:
                    assigned_kernel_idx = int(idx * self.kw_planes_per_mixture + self.kw_planes_per_mixture - 1)
                    self.temp_bias[assigned_kernel_idx] = 1
            return start_cell_idx

    def map_to_cell_basis(self, x):
        x = x.reshape([-1, self.num_local_cell + self.num_basis + 1])
        x_local, x_nonlocal, x_zero = x[:, :self.num_local_cell], x[:, self.num_local_cell:-1], x[:, -1:]
        x_nonlocal = self.fc3(x_nonlocal)
        x = torch.cat([x_nonlocal[:, :self.start_cell_idx], x_local, x_nonlocal[:, self.start_cell_idx:], x_zero], dim=1)
        x = x.reshape(-1, self.kw_planes) + self.basis_bias.reshape(1, -1)
        return x

    def forward(self, x):
        x_flat = x.reshape(*x.shape[:2], -1) 
        
        x_avg = x_flat.mean(dim=-1)
        x_max = x_flat.max(dim=-1)[0]
        
        x_pooled = torch.cat([x_avg, x_max], dim=1) 

        x_out = self.act1(self.norm1(self.fc1(x_pooled)))
        x_out = self.map_to_cell(self.fc2(x_out)).reshape(-1, self.kw_planes_per_mixture)
        x_out = x_out / (torch.sum(torch.abs(x_out), dim=1).view(-1, 1) + 1e-3)
        x_out = (1.0 - self.temp_value) * x_out.reshape(-1, self.kw_planes) \
            + self.temp_value * self.temp_bias.to(x_out.device).view(1, -1)
        return x_out.reshape(-1, self.kw_planes_per_mixture)[:, :-1]


class KWconvNd(nn.Module):
    dimension = None
    permute = None
    func_conv = None

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, warehouse_id=None, warehouse_manager=None):
        super(KWconvNd, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = parse(kernel_size, self.dimension)
        self.stride = parse(stride, self.dimension)
        self.padding = parse(padding, self.dimension)
        self.dilation = parse(dilation, self.dimension)
        self.groups = groups
        self.bias = nn.Parameter(torch.zeros([self.out_planes]), requires_grad=True).float() if bias else None
        self.warehouse_id = warehouse_id
        self.warehouse_manager = [warehouse_manager]

    def init_attention(self, cell, start_cell_idx, reduction, cell_num_ratio, norm_layer, nonlocal_basis_ratio=1.0):
        self.cell_shape = cell.shape
        self.groups_out_channel = self.out_planes // self.cell_shape[1]
        self.groups_in_channel = self.in_planes // self.cell_shape[2] // self.groups
        self.groups_spatial = 1
        for idx in range(len(self.kernel_size)):
            self.groups_spatial = self.groups_spatial * self.kernel_size[idx] // self.cell_shape[3 + idx]
        num_local_mixture = self.groups_out_channel * self.groups_in_channel * self.groups_spatial
        
        self.attention = UAV_Attention(self.in_planes, reduction, self.cell_shape[0], num_local_mixture,
                                       norm_layer=norm_layer, nonlocal_basis_ratio=nonlocal_basis_ratio,
                                       cell_num_ratio=cell_num_ratio, start_cell_idx=start_cell_idx)
        return self.attention.init_temperature(start_cell_idx, cell_num_ratio)

    def forward(self, x):
        kw_attention = self.attention(x).type(x.dtype)
        batch_size = x.shape[0]
        x_reshaped = x.reshape(1, -1, *x.shape[2:])
        weight = self.warehouse_manager[0].take_cell(self.warehouse_id).reshape(self.cell_shape[0], -1).type(x.dtype)
        aggregate_weight = torch.mm(kw_attention, weight)
        aggregate_weight = aggregate_weight.reshape([batch_size, self.groups_spatial, self.groups_out_channel,
                                                     self.groups_in_channel, *self.cell_shape[1:]])
        aggregate_weight = aggregate_weight.permute(*self.permute)
        aggregate_weight = aggregate_weight.reshape(-1, self.in_planes // self.groups, *self.kernel_size)
        output = self.func_conv(x_reshaped, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, *output.shape[2:])
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, *([1]*self.dimension))
        return output

class KWConv1d(KWconvNd):
    dimension = 1
    permute = (0, 2, 4, 3, 5, 1, 6)
    func_conv = F.conv1d

class KWConv2d(KWconvNd):
    dimension = 2
    permute = (0, 2, 4, 3, 5, 1, 6, 7)
    func_conv = F.conv2d

class KWConv3d(KWconvNd):
    dimension = 3
    permute = (0, 2, 4, 3, 5, 1, 6, 7, 8)
    func_conv = F.conv3d

class KWLinear(nn.Module):
    dimension = 1
    def __init__(self, *args, **kwargs):
        super(KWLinear, self).__init__()
        self.conv = KWConv1d(*args, **kwargs)

    def forward(self, x):
        shape = x.shape
        x = self.conv(x.reshape(shape[0], -1, shape[-1]).transpose(1, 2))
        x = x.transpose(1, 2).reshape(*shape[:-1], -1)
        return x

class MSD_KWConv2d(KWconvNd):
    dimension = 2
    permute = (0, 2, 4, 3, 5, 1, 6, 7)
    func_conv = F.conv2d

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilations=[1, 2, 3], groups=1,
                 bias=False, warehouse_id=None, warehouse_manager=None):
        super(MSD_KWConv2d, self).__init__(
            in_planes, out_planes, kernel_size, stride=stride, padding=padding, 
            dilation=1, groups=groups, bias=bias, 
            warehouse_id=warehouse_id, warehouse_manager=warehouse_manager
        )
        
        self.dilations = dilations
        fuse_channels = max(32, out_planes // 4) 
        self.fuse_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_planes, fuse_channels, 1, bias=False),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_channels, out_planes * len(dilations), 1, bias=False) 
        )

    def forward(self, x):
        kw_attention = self.attention(x).type(x.dtype)
        batch_size = x.shape[0]
        x_reshaped = x.reshape(1, -1, *x.shape[2:])
        weight = self.warehouse_manager[0].take_cell(self.warehouse_id).reshape(self.cell_shape[0], -1).type(x.dtype)
        aggregate_weight = torch.mm(kw_attention, weight)
        aggregate_weight = aggregate_weight.reshape([batch_size, self.groups_spatial, self.groups_out_channel,
                                                     self.groups_in_channel, *self.cell_shape[1:]])
        aggregate_weight = aggregate_weight.permute(*self.permute)
        aggregate_weight = aggregate_weight.reshape(-1, self.in_planes // self.groups, *self.kernel_size)

        outputs = []
        for d in self.dilations:
            adaptive_pad = d * (self.kernel_size[0] - 1) // 2 
            
            out = self.func_conv(x_reshaped, weight=aggregate_weight, bias=None, 
                                 stride=self.stride, padding=adaptive_pad,
                                 dilation=d, groups=self.groups * batch_size)
            out = out.view(batch_size, self.out_planes, *out.shape[2:])
            outputs.append(out)

        U = sum(outputs) 
        att_weights = self.fuse_fc(U) 
        att_weights = att_weights.view(batch_size, len(self.dilations), self.out_planes, 1, 1)
        att_weights = F.softmax(att_weights, dim=1) 

        out_fused = 0
        for i in range(len(self.dilations)):
            out_fused += outputs[i] * att_weights[:, i, :, :, :]

        if self.bias is not None:
            out_fused = out_fused + self.bias.reshape(1, -1, 1, 1)
            
        return out_fused


class Warehouse_Manager(nn.Module):
    def __init__(self, reduction=0.0625, cell_num_ratio=1, cell_inplane_ratio=1,
                 cell_outplane_ratio=1, sharing_range=(), nonlocal_basis_ratio=1,
                 norm_layer=nn.BatchNorm1d, spatial_partition=True):
        super(Warehouse_Manager, self).__init__()
        self.sharing_range = sharing_range
        self.warehouse_list = {}
        self.reduction = reduction
        self.spatial_partition = spatial_partition
        self.cell_num_ratio = cell_num_ratio
        self.cell_outplane_ratio = cell_outplane_ratio
        self.cell_inplane_ratio = cell_inplane_ratio
        self.norm_layer = norm_layer
        self.nonlocal_basis_ratio = nonlocal_basis_ratio
        self.weights = nn.ParameterList()

    def fuse_warehouse_name(self, warehouse_name):
        fused_names = []
        for sub_name in warehouse_name.split('_'):
            match_name = sub_name
            for sharing_name in self.sharing_range:
                if str.startswith(match_name, sharing_name):
                    match_name = sharing_name
            fused_names.append(match_name)
        fused_names = '_'.join(fused_names)
        return fused_names

    def reserve(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                bias=True, warehouse_name='default', enabled=True, layer_type='conv2d', dilations=[1, 2, 3]):
        
        # 加入 msd_conv2d 映射
        kw_mapping = {'conv1d': KWConv1d, 'conv2d': KWConv2d, 'conv3d': KWConv3d, 'linear': KWLinear, 'msd_conv2d': MSD_KWConv2d}
        org_mapping = {'conv1d': nn.Conv1d, 'conv2d': nn.Conv2d, 'conv3d': nn.Conv3d, 'linear': nn.Linear}

        if not enabled:
            _layer_type = org_mapping['conv2d'] if layer_type == 'msd_conv2d' else org_mapping[layer_type]
            if _layer_type is nn.Linear:
                return _layer_type(in_planes, out_planes, bias=bias)
            else:
                return _layer_type(in_planes, out_planes, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                 groups=groups, bias=bias)
        else:
            layer_type_cls = kw_mapping[layer_type]
            warehouse_name = self.fuse_warehouse_name(warehouse_name)
            weight_shape = [out_planes, in_planes // groups, *parse(kernel_size, layer_type_cls.dimension)]

            if warehouse_name not in self.warehouse_list.keys():
                self.warehouse_list[warehouse_name] = []
            self.warehouse_list[warehouse_name].append(weight_shape)

            if layer_type == 'msd_conv2d':
                return layer_type_cls(in_planes, out_planes, kernel_size, stride=stride, padding=padding,
                                      dilations=dilations, groups=groups, bias=bias,
                                      warehouse_id=int(list(self.warehouse_list.keys()).index(warehouse_name)),
                                      warehouse_manager=self)
            else:
                return layer_type_cls(in_planes, out_planes, kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, groups=groups, bias=bias,
                                      warehouse_id=int(list(self.warehouse_list.keys()).index(warehouse_name)),
                                      warehouse_manager=self)

    def store(self):
        warehouse_names = list(self.warehouse_list.keys())
        self.reduction = parse(self.reduction, len(warehouse_names))
        self.spatial_partition = parse(self.spatial_partition, len(warehouse_names))
        self.cell_num_ratio = parse(self.cell_num_ratio, len(warehouse_names))
        self.cell_outplane_ratio = parse(self.cell_outplane_ratio, len(warehouse_names))
        self.cell_inplane_ratio = parse(self.cell_inplane_ratio, len(warehouse_names))

        for idx, warehouse_name in enumerate(self.warehouse_list.keys()):
            warehouse = self.warehouse_list[warehouse_name]
            dimension = len(warehouse[0]) - 2

            out_plane_gcd, in_plane_gcd, kernel_size = warehouse[0][0], warehouse[0][1], warehouse[0][2:]
            for layer in warehouse:
                out_plane_gcd = math.gcd(out_plane_gcd, layer[0])
                in_plane_gcd = math.gcd(in_plane_gcd, layer[1])
                if not self.spatial_partition[idx]:
                    assert kernel_size == layer[2:]

            cell_in_plane = max(int(in_plane_gcd * self.cell_inplane_ratio[idx]), 1)
            cell_out_plane = max(int(out_plane_gcd * self.cell_outplane_ratio[idx]), 1)
            cell_kernel_size = parse(1, dimension) if self.spatial_partition[idx] else kernel_size

            num_total_mixtures = 0
            for layer in warehouse:
                groups_channel = int(layer[0] // cell_out_plane * layer[1] // cell_in_plane)
                groups_spatial = 1

                for d in range(dimension):
                    groups_spatial = int(groups_spatial * layer[2 + d] // cell_kernel_size[d])

                num_layer_mixtures = groups_spatial * groups_channel
                num_total_mixtures += num_layer_mixtures

            self.weights.append(nn.Parameter(torch.randn(
                max(int(num_total_mixtures * self.cell_num_ratio[idx]), 1),
                cell_out_plane, cell_in_plane, *cell_kernel_size), requires_grad=True))

    def allocate(self, network, _init_weights=partial(nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu')):
        num_warehouse = len(self.weights)
        end_idxs = [0] * num_warehouse

        for layer in network.modules():
            if isinstance(layer, KWconvNd):
                warehouse_idx = layer.warehouse_id
                start_cell_idx = end_idxs[warehouse_idx]
                end_cell_idx = layer.init_attention(self.weights[warehouse_idx],
                                                    start_cell_idx,
                                                    self.reduction[warehouse_idx],
                                                    self.cell_num_ratio[warehouse_idx],
                                                    norm_layer=self.norm_layer,
                                                    nonlocal_basis_ratio=self.nonlocal_basis_ratio)
                _init_weights(self.weights[warehouse_idx][start_cell_idx:end_cell_idx].view(
                    -1, *self.weights[warehouse_idx].shape[2:]))
                end_idxs[warehouse_idx] = end_cell_idx

        for warehouse_idx in range(len(end_idxs)):
            assert end_idxs[warehouse_idx] == self.weights[warehouse_idx].shape[0]

    def take_cell(self, warehouse_idx):
        return self.weights[warehouse_idx]


class DMB(nn.Module):
    def __init__(self, c1, c2, wm=None, wm_name=None, k=1, s=1, p=None, g=1, d=1, act=True) -> None:
        super().__init__()
        assert wm != None, 'wm param must be class Warehouse_Manager.'
        assert wm_name != None, 'wm_name param must not be None.'
        
        k_val = k[0] if isinstance(k, (tuple, list)) else k

        if k_val >= 3 and c1 >= 256:
            self.conv = wm.reserve(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d), 
                                   dilation=1, groups=g, bias=False, 
                                   warehouse_name=wm_name, layer_type='msd_conv2d', 
                                   dilations=[1, 2]) 
        else:
            self.conv = wm.reserve(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d), 
                                   dilation=d, groups=g, bias=False, 
                                   warehouse_name=wm_name, layer_type='conv2d')
            
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MSD_KWBlock(nn.Module):
    def __init__(self, c1, c2, wm=None, wm_name=None, k=3, s=1, p=None, g=1, dilations=[1, 2, 3], act=True) -> None:
        super().__init__()
        assert wm != None, 'wm param must be class Warehouse_Manager.'
        assert wm_name != None, 'wm_name param must not be None.'
        
        self.conv = wm.reserve(c1, c2, k, s, autopad(k, p), d=1, groups=g, bias=False, 
                               warehouse_name=wm_name, layer_type='msd_conv2d', dilations=dilations)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def get_temperature(iteration, epoch, iter_per_epoch, temp_epoch=20, temp_init_value=30.0, temp_end=0.0):
    total_iter = iter_per_epoch * temp_epoch
    current_iter = iter_per_epoch * epoch + iteration
    temperature = temp_end + max(0, (temp_init_value - temp_end) * ((total_iter - current_iter) / max(1.0, total_iter)))
    return temperature