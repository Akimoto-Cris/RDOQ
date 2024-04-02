import torch
import torch.nn as nn
import algo
import torch.nn.functional as F
from itertools import repeat
import collections.abc as container_abcs
from torch.profiler import record_function
import common
import time

tochannelfirst = lambda x: x.permute(0, 3, 1, 2) # einops.rearrange(x, 'b h w c -> b c h w')
tochannellast = lambda x: x.permute(0, 2, 3, 1) # einops.rearrange(x, 'b c h w -> b h w c')



def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

class quantConvLayer(nn.Module):
    def __init__(self, weight, bias, in_channels, out_channels, stride, padding, phi, delta, bit, nfilterbatch=1, actbit=-1, groups=-1, alpha_act=None, bcw=False):
        super(quantConvLayer, self).__init__()
        self.mod = quantConv2d(in_channels, out_channels, weight.shape[2], stride, padding, phi, delta, bit, nfilterbatch, actbit, alpha_act=alpha_act, bcw=bcw)

        with torch.no_grad():
            self.mod.weight = weight
            self.mod.bias = bias
            if groups != -1:
                self.mod.groups = groups
    
    @property
    def weight(self):
        return self.mod.weight

    def forward(self, x):
        x = self.mod(x)
        return x


class quantLinLayer(nn.Module):
    def __init__(self, weight, bias, in_channels, out_channels, phi, delta, bit, nfilterbatch=1, actbit=-1, alpha_act=None, bcw=False):
        super(quantLinLayer, self).__init__()

        self.mod = quantLinear(in_channels, out_channels, bias, phi, delta, bit, nfilterbatch, actbit, alpha_act=alpha_act, bcw=bcw)

        with torch.no_grad():
            self.mod.weight = weight
            self.mod.bias = bias
    
    @property
    def weight(self):
        return self.mod.weight

    def forward(self, x):
        x = self.mod(x)
        return x

class quantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, phi, delta, bit, nfilterbatch, actbits, alpha_act=None, bcw=False):
        super(quantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
        self.phi = phi
        self.delta = delta
        self.bit = bit
        self.quant = Quantize.apply
        # self.actquant = ActQuantizer.apply
        self.nfilterbatch = nfilterbatch
        self.actbits = actbits
        self.act_deltas = alpha_act
        # self.register_parameter('act_alpha', nn.Parameter(torch.tensor(alpha_act)))
        self.chans_per_group = min(self.nfilterbatch, self.in_channels)
        self.bcw = bcw
    
    def quantize_group(self, input, group_id):
        input[:, group_id*self.chans_per_group: (group_id+1)*self.chans_per_group] = common.quantize(input[:, group_id*self.chans_per_group: (group_id+1)*self.chans_per_group],
                                                                                               2 ** self.act_deltas[group_id], self.actbits[group_id])
        return input          

    def actquant(self, input):
        input_q = input.clone()
        for g in range(len(self.act_deltas)):
            input_q = self.quantize_group(input_q, g)
        return input_q

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        input_q = self.actquant(input)
        weight_q = self.quant(self.weight, self.phi, self.delta, self.bit, self.nfilterbatch)
        if self.bcw:
            weight_q = algo.bcorr_weight(self.weight, weight_q)
        return self.conv2d_forward(input_q, weight_q)


class quantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias, phi=None, delta=None, bit=None, nfilterbatch=None, actbits=None, alpha_act=None, bcw=False):
        super(quantLinear, self).__init__(in_features, out_features, bias=bias is not None)
        self.phi = phi
        self.delta = delta
        self.bit = bit
        self.quant = Quantize.apply
        # self.actquant = ActQuantizer.apply
        self.nfilterbatch = nfilterbatch
        self.actbits = actbits
        self.act_deltas = alpha_act
        # self.register_parameter('act_alpha', nn.Parameter(torch.tensor(alpha_act)))
        self.chans_per_group = min(self.nfilterbatch, self.in_features)
        self.bcw = bcw

    def quantize_group(self, input, group_id):
        input[..., group_id*self.chans_per_group: (group_id+1)*self.chans_per_group] = common.quantize(input[..., group_id*self.chans_per_group: (group_id+1)*self.chans_per_group],
                                                                                       2 ** self.act_deltas[group_id], self.actbits[group_id])
        return input          

    def actquant(self, input):
        input_q = input.clone()
        for g in range(len(self.act_deltas)):
            input_q = self.quantize_group(input_q, g)
        return input_q

    def forward(self, input):
        input_q = self.actquant(input)
        weight_q = self.quant(self.weight, self.phi, self.delta, self.bit, self.nfilterbatch)
        if self.bcw:
            weight_q = algo.bcorr_weight(self.weight, weight_q)
        # print(np.unique(weight_q.detach().numpy()))
        return F.linear(input_q, weight_q, self.bias)

class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, phi, delta, bit, nfilterbatch):
        quant = weight.clone()
        cnt = 0
        for f in range(0, quant.shape[0], nfilterbatch):
            st_layer = f
            ed_layer = f + nfilterbatch
            if ed_layer > quant.shape[0]:
                ed_layer = quant.shape[0]
            
            if bit[cnt] >= quant[st_layer:ed_layer].numel() and not bit[cnt] % quant[st_layer:ed_layer].numel():
                bit[cnt] = bit[cnt] / quant[st_layer:ed_layer].numel()
            quant[st_layer:ed_layer] = algo.deadzone_quantize(quant[st_layer:ed_layer], phi[cnt], 2**delta[cnt], bit[cnt])
            cnt = cnt + 1
        return quant

    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class ActQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, qbits):
        input = input / alpha                          # weights are first divided by alpha
        input_c = input.clamp(min=-1, max=1)

        n = 2. ** (qbits-1) - 1.
        input_q = input_c.mul_(n).round_().div_(n)
        ctx.save_for_backward(input, input_q)
        input_q = input_q.mul(alpha)               # rescale to the original range
        return input_q

    def backward(ctx, grad_output):
        grad_input = grad_output#.clone()             # grad for weights will not be clipped
        input, input_q = ctx.saved_tensors
        
        i = (input.abs() > 1.).float()
        sign = input.sign()
        grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i)))#.sum()
        grad_input = grad_input * (1 - i)
        
        return grad_input, grad_alpha, None


def uniform_quantize(k, activation=False):
    n = 2. ** (k - 1) - 1.
    recp_n = 1 / n

    class qfn_3(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input_save = input.mul(1/alpha)
            input_q = input_save.mul(n).clamp_(min=-n, max=n).round_().mul_(recp_n)
            ctx.save_for_backward(input_save, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            input, input_q = ctx.saved_tensors
            i = (input.abs() > 1).int()
            grad_alpha = (grad_output * (input.sign() * i + (input_q - input) * (1 - i)) )#.sum()
            return grad_output, grad_alpha

    def fast_qfn(input, scale):
        # shift = alpha.log2().ceil_()
        # m_o = (2 ** shift) / alpha
        return input.div_(scale).clamp_(min=-n, max=n).round_()

    # return qfn_3().apply
    return fast_qfn


"""class weight_quantize_fn(nn.Module):    # SAWB
    def __init__(self, w_bit, percentage=1.):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)
        self.percentage = percentage
        self.sawb_params = _sawb_asymm_lut[self.w_bit]

    def forward(self, x):
        if self.w_bit == 32: return x
        w_ = x.clone().detach()
        abs_w = w_.abs()
        Ew1 = abs_w.mean()
        Ew2 = abs_w.pow_(2).mean()
        alpha = self.sawb_params[0] * Ew2.sqrt_() - self.sawb_params[1] * Ew1
        # check_minmax=True
        N = 2 ** (self.w_bit - 1) - 1
        alpha = min(alpha, w_.min().abs())
        scale = alpha / N
        if self.percentage == 1:
            return self.uniform_q(x, alpha), scale
        mask = (torch.rand_like(x) <= self.percentage).int()
        return mask * self.uniform_q(x, alpha) + (1 - mask) * x
"""

class activation_quantize_fn(nn.Module):    # PACT
    def __init__(self, a_bit, act_scale=None):
        super(activation_quantize_fn, self).__init__()
        assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)
        self.act_scale = act_scale

    def forward(self, x):
        return self.uniform_q(x, self.act_scale), self.act_scale

    # def forward(self, x):
    #     return common.quantize(x, self.act_scale, self.a_bit) / self.act_scale, self.act_scale


### INT8 and INT4 GEMM implementation
import math
import einops
try:
    import cutlassconv_cuda
    import int8mm_cuda
except Exception as e:
    print(e)


def conv2d_int8(input, weight, stride=1, padding=1):
    # only input channel(tensor.size(3)) is a multiple of 16
    if input.size(3) % 16 != 0:
        padding_channels = 16 - input.size(3) % 16
        input_padded = F.pad(input, (0, padding_channels),"constant", 0)
        weight_padded = F.pad(weight, (0, padding_channels),"constant", 0)
    else:
        input_padded = input
        weight_padded = weight
    
    begin = time.perf_counter()
    if weight.size(1) <= 32:
        temp = cutlassconv_cuda.int8_conv_optimized(input_padded, weight_padded, stride, padding)
    else:
        temp = cutlassconv_cuda.int8_conv(input_padded, weight_padded, stride, padding)
    t = time.perf_counter() - begin
    # print(temp.shape)
    return temp, t


def conv2d_int4(input, weight, stride=1, padding=1):
    # only input channel(tensor.size(3)) is a multiple of 32
    if input.size(3) % 32 != 0:
        padding_channels = 32 - input.size(3) % 32
        input_padded = F.pad(input, (0, padding_channels),"constant", 0)
        weight_padded = F.pad(weight, (0, padding_channels),"constant", 0)
    else:
        input_padded = input
        weight_padded = weight
    input_padded = pack_int8_to_int4(input_padded.clone())
    weight_padded = pack_int8_to_int4(weight_padded.clone())

    begin = time.perf_counter()
    if weight.size(1) <= 32:
        temp = cutlassconv_cuda.int4_conv_optimized(input_padded, weight_padded, stride, padding)
    else:
        temp = cutlassconv_cuda.int4_conv(input_padded, weight_padded, stride, padding)
    t = time.perf_counter() - begin

    return temp, t

def conv2d_fp(input, weight, stride=1, padding=1):
    input_ = input.permute(0, 3, 1, 2).contiguous().float()
    weight_ = weight.permute(0, 3, 1, 2).contiguous().float()
    begin = time.perf_counter()
    temp = F.conv2d(input_, weight_, None, (stride, stride), (padding, padding))
    t = time.perf_counter() - begin
    return temp.permute(0,2,3,1).contiguous(), t


class INTQuantBase(nn.Module):
    def __init__(self, weight, bias, phi, delta, bit, nfilterbatch=1, actbit=-1, act_scale=None):
        super().__init__()
        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias
        self.phi = phi
        self.delta = torch.tensor(delta).cuda()
        self.bit = bit
        self.nfilterbatch = nfilterbatch
        self.actbit = actbit
        self.act_quantize_fn = activation_quantize_fn(actbit, act_scale)
        self.weight_ori = self.weight.data.clone()
        # import pdb; pdb.set_trace()
        self.weight.data = Quantize.apply(self.weight.data, self.phi, self.delta, self.bit, self.nfilterbatch)
        self.delta_ = self.expand_delta()
        self.scale_weight = 2 ** self.delta_
        self.weight.data = self.weight.data.div(self.scale_weight.view((-1,) + (1,) * (len(self.weight.shape) - 1)))
        self.time = None
    
    def expand_delta(self):
        return einops.repeat(self.delta, 'g -> (g c)', c=self.nfilterbatch)[:self.weight.shape[0]]

    @staticmethod
    def inspect_grad(x, name=""):
        if not x.is_leaf:
            x.register_hook(lambda p: print(name, p.abs().max()))

    def extra_forward(self, *args, **kwargs):
        pass

    def forward(self, input):
        if isinstance(self, Conv2d_Q):
            input = tochannellast(input)
        # input_ori = input.clone()
        self.act_quantize_fn.act_scale = input.abs().max() / (2 ** (self.actbit - 1))
        with record_function("input_quant"):
            input_q, scale_input = self.act_quantize_fn(input)
        # print((input_ori - input_q.float() * scale_input).pow(2).mean())
        out = self.extra_forward(input_q, self.weight_int8, scale_input, self.scale_weight) + (self.bias if self.bias is not None else 0)
        # if self.time > 0.0001:
        #     print(self.weight.shape, input.shape)
        if isinstance(self, Conv2d_Q):
            out = tochannelfirst(out)
        return out


class Conv2d_Q(INTQuantBase):
    def __init__(self, weight, bias, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, phi, delta, bit, nfilterbatch=1, actbit=-1, act_scale=None):
        super(Conv2d_Q, self).__init__(weight, bias, phi, delta, bit, nfilterbatch, actbit, act_scale)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        assert self.dilation[0] == 1 and self.groups == 1, "dilation and groups currently only support 1" 
        self.mma_dict = {8: conv2d_int8, 
                         4: conv2d_int4,
                         32: conv2d_fp}
        self.weight_int8 = self.weight.to(torch.int8)
    
    def extra_forward(self, x, weight, scale_input, scale_weight):
        temp, self.time = self.mma_dict[self.actbit](x.to(torch.int8), weight, self.stride[0], self.padding[0])
        return temp.float() * (scale_input * scale_weight)
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ", wbit={bit}, abit={actbit}"
        return s.format(**self.__dict__)

def roundoff4(size):
    return (size+3) // 4 * 4

def roundoffn(size, n=16):
    return (size+n - 1) // n * n


def mm_int8(lhs,rhs):
    m = roundoffn(lhs.size(0), 16)
    k = roundoffn(lhs.size(1), 16)
    n = roundoffn(rhs.size(0), 16)

    k_diff = k - lhs.size(1)
    n_diff = n - rhs.size(0)
    m_diff = m - lhs.size(0)

    if k_diff:
        A = F.pad(lhs, (0, k_diff, 0, m_diff), "constant", 0)
    else:
        A = lhs

    if n_diff or k_diff:
        B = F.pad(rhs, (0, k_diff, 0, n_diff), "constant", 0)
    else:
        B = rhs
    B_ = B.T.contiguous()
    begin = time.perf_counter()
    temp = int8mm_cuda.int8_mm(A, B_)
    t = time.perf_counter() - begin
    # print((temp == 0).count_nonzero())
    if n_diff:
        temp = temp[:lhs.size(0),:rhs.size(0)]
    return temp.contiguous(), t


def mm_fp(lhs,rhs):
    begin = time.perf_counter()
    temp = lhs.float().matmul(rhs.float().t())
    t = time.perf_counter() - begin
    return temp, t


def pack_int8_to_int4(data, transpose=False):
    new = data.reshape(-1)
    data_ = data.reshape(-1, 2)
    # data_[:, 0] = (data_[:, 0] < 0) * (data_[:, 0] - 112) + (data_[:, 0] >= 0) * data_[:, 0]
    new[:new.shape[0] // 2] = (data_[:, 0] << 4) + (data_[:, 1] >= 0) * data_[:, 1] + (data_[:, 1] < 0) * (16 + data_[:, 1])        # 2^7 - unsign(x) - dec(0111000)
    if transpose:
        return new.reshape(*data.shape[::-1])
    return new.reshape(*data.shape)


def pack_int8_to_int4_unsigned(data, transpose=False):
    new = data.reshape(-1)
    data_ = data.reshape(-1, 2)
    new[:new.shape[0] // 2] = (data_[:, 0] << 4) + data_[:, 1]
    if transpose:
        return new.reshape(*data.shape[::-1])
    return new.reshape(*data.shape)


def mm_int4(lhs,rhs): # the cuda extension only support n,k as a multiply of 4
    # use torch.nn.pad to pad 0 if these dimension doesn't satisfy the
    # requirement
    m = roundoffn(lhs.size(0), 32) # roundoff4(lhs.size(1))
    k = roundoffn(lhs.size(1), 32) # roundoff4(lhs.size(1))
    n = roundoffn(rhs.size(0), 32) # roundoff4(rhs.size(1))  #

    k_diff = k - lhs.size(1)
    n_diff = n - rhs.size(0)
    m_diff = m - lhs.size(0)

    if k_diff or m_diff:
        A = F.pad(lhs, (0, k_diff, 0, m_diff), "constant", 0)
    else:
        A = lhs

    if n_diff or k_diff:
        B = F.pad(rhs, (0, k_diff, 0, n_diff), "constant", 0)
    else:
        B = rhs

    # import pdb; pdb.set_trace()
    # print((torch.matmul(A.float(), B.T.contiguous().float()) - cutlassconv_cuda.int4_mm(pack_int8_to_int4(A), pack_int8_to_int4(B))).sum())
    A = A.clamp_(-8, 7)
    B = B.clamp_(-8, 7)
    A_ = pack_int8_to_int4(A.clone())
    B_ = pack_int8_to_int4(B.clone())
    begin = time.perf_counter()
    temp = cutlassconv_cuda.int4_mm(A_, B_)
    t = time.perf_counter() - begin

    if n_diff:
        temp = temp[:lhs.size(0),:rhs.size(0)]
    temp = temp[:lhs.size(0)]
    return temp, t


class Linear_Q(INTQuantBase):
    def __init__(self, weight, bias, in_features, out_features, phi, delta, bit, nfilterbatch=1, actbit=-1, act_scale=None):
        super(Linear_Q, self).__init__(weight, bias, phi, delta, bit, nfilterbatch, actbit, act_scale)
        self.mma_dict = {8: mm_int8,
                         4: mm_int4,
                         32: mm_fp}
        self.weight_int8 = self.weight.to(torch.int8)

    def extra_forward(self, input, weight, scale_input, scale_weight):
        # print(self.weight_ori.max(), (self.weight_int8.float() * scale_weight[0]).max())
        if len(input.shape) == 3:
            b, n_token = input.shape[:2] 
            return self.Lin_Fn(input.view(b * n_token, -1), weight, scale_input, scale_weight).view(b, n_token, -1)
        return self.Lin_Fn(input, weight, scale_input, scale_weight)
    
    def Lin_Fn(self, input, weight, scale_input, scale_weight):
        # import pdb; pdb.set_trace()
        temp, self.time = self.mma_dict[self.actbit](input.to(torch.int8), weight)
        out = temp.float() * scale_input * scale_weight
        return out

    def __repr__(self):
        return super(Linear_Q, self).__repr__().split(")")[0] + f"bit={self.bit})"
