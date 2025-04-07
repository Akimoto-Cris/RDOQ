import torch
import torch.nn as nn
import common
from header import *
from algo import bcorr_act, deadzone_quantize
import torch.nn.functional as F

class TransConv2d(nn.Module):
    def __init__(self, base, kern, bias, stride, padding, trantype, block, \
                 kern_coded, kern_delta, base_coded, base_delta, codekern, codebase):
        super(TransConv2d, self).__init__()

        if trantype == 'inter':
            self.conv1 = QuantConv2d(base.shape[1],base.shape[0],kernel_size=1,bias=False,\
                                     delta=base_delta,coded=base_coded,block=block,is_coded=codebase)
            self.conv2 = QuantConv2d(kern.shape[1],kern.shape[0],kernel_size=kern.shape[2],perm=True,\
                                     stride=stride,padding=padding,delta=kern_delta,coded=kern_coded,\
                                     block=block,is_coded=codekern)
            with torch.no_grad():
                self.conv1.weight[:] = base.reshape(self.conv1.weight.shape)
                self.conv2.weight[:] = kern.reshape(self.conv2.weight.shape)
                self.conv2.bias = bias
        elif trantype == 'exter':
            self.conv1 = QuantConv2d(kern.shape[1],kern.shape[0],kernel_size=kern.shape[2],bias=False,\
                                     stride=stride,padding=padding,delta=kern_delta,coded=kern_coded,\
                                     block=block,is_coded=codekern)
            self.conv2 = QuantConv2d(base.shape[1],base.shape[0],kernel_size=1,perm=True,\
                                     delta=base_delta,coded=base_coded,block=block,is_coded=codebase)
            with torch.no_grad():
                self.conv1.weight[:] = kern.reshape(self.conv1.weight.shape)
                self.conv2.weight[:] = base.reshape(self.conv2.weight.shape)
                self.conv2.bias = bias

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)

        return x

    def quantize(self):
        self.conv1.quantize()
        self.conv2.quantize()


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, delta, coded, block,\
                 is_coded, stride=1, padding=0, bias=False, perm=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size,\
                                          stride=stride,padding=padding,bias=bias)
        self.delta = delta
        self.coded = coded
        self.block = block
        self.perm = perm
        self.is_coded = is_coded
        self.quant = Quantize.apply

        if not self.is_coded:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        if self.is_coded:
            return self.conv2d_forward(input, self.quant(self.weight,self.delta,self.coded,\
                                                         self.block,self.perm))
        else:
            return self.conv2d_forward(input, self.weight)

    def quantize(self):
        if self.is_coded:
            self.quant(self.weight,self.delta,self.coded,self.block,self.perm,inplace=True)

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
            quant[st_layer:ed_layer, ...] = deadzone_quantize(quant[st_layer:ed_layer, ...], phi[cnt], 2**delta[cnt], bit[cnt])
            cnt = cnt + 1

        return quant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class ActQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, bit):
        return common.quantize(weight.clone(), alpha, bit)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# Wrapper module that perform A quantization before the wrapped layer
class QAWrapper(nn.Module):
    def __init__(self, layer, delta, coded, quantized=False):
        super(QAWrapper, self).__init__()
        self.quant = self.act_quant
        self.quantized = quantized
        self.layer = layer
        self.delta = delta
        self.coded = coded
        self.numel = 1
        self.sum_err_a = 0
        self.count = 0
        self.stats = True
        self.bca = None
        self.group_id = -1
        self.chans_per_group = 64
        self.mode = "in_channel"
        self.adaptive_delta = False
    
    def act_quant(self, input, delta, coded):
        bit = coded / input[0].numel()
        if self.adaptive_delta:
            new_alpha = ternary_search_alpha(input, bit)
            delta = torch.log2(new_alpha) - (bit-1)
        return common.quantize(input, 2 ** delta, bit)

    @property
    def weight(self):
        return self.layer.weight

    # @property
    # def bias(self):
    #     return self.layer.bias

    def quantize_group(self, input, group_id):
        if isinstance(self.layer, nn.Conv2d):
            input[:, group_id*self.chans_per_group: (group_id+1)*self.chans_per_group] = self.quant(input[:, group_id*self.chans_per_group: (group_id+1)*self.chans_per_group],
                                                                                    self.delta[group_id], self.coded[group_id])
        else:
            input[..., group_id*self.chans_per_group: (group_id+1)*self.chans_per_group] = self.quant(input[..., group_id*self.chans_per_group: (group_id+1)*self.chans_per_group],
                                                                                    self.delta[group_id], self.coded[group_id])
        return input                                                                                

    @staticmethod
    def permute(x):
        if len(x.shape) == 3:
            return x.transpose(1,2).contiguous()
        return x

    def forward(self, input):
        self.numel = input[0].numel()
        if self.quantized and self.mode == "in_channel":
            if self.group_id == -1:
                input_q = input.clone()
                for g in range(len(self.delta)):
                    input_q = self.quantize_group(input_q, g)
                # print("mse: ", (input - input_q).pow(2).mean(), self.layer)
            else:
                input_q = self.quantize_group(input, self.group_id)

            if self.stats:
                self.sum_err_a += (input - input_q).transpose(0,1).flatten(1).mean(1)
                self.count += 1
            if self.bca is not None:
                input_q = self.bca(self.permute(input), self.permute(input_q))
                input_q = self.permute(input_q)
            
            return self.layer(input_q)
        
        ret = self.layer(input)

        if self.quantized and self.mode == "out_channel":
            if self.group_id == -1:
                ret_q = ret.clone()
                for g in range(len(self.delta)):
                    ret_q = self.quantize_group(ret_q, g)
            else:
                ret_q = self.quantize_group(ret, self.group_id)
            
            if self.stats:
                self.sum_err_a += (ret - ret_q).transpose(0,1).flatten(1).mean(1)
                self.count += 1

            if self.bca is not None:
                ret_q = self.bca(self.permute(ret), self.permute(ret_q))
                ret_q = self.permute(ret_q)
            
            return ret_q

        return ret

    def extra_repr(self):
        dic = {'depth': [sum(self.coded) / self.numel], 'delta': self.delta, 'quantized': self.quantized}
        s = ('bit_depth={depth}, step_size={delta}, quantized={quantized}, mode={mode}')
        return self.layer.__repr__() + ', ' + s.format(**dic)

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

@torch.no_grad()
def ternary_search_alpha(input, bit, max_iter=20, estimate_alpha=None):
    if estimate_alpha is not None:
        low_alpha = estimate_alpha * 0.5
        high_alpha = estimate_alpha * 1.5
    else:
        low_alpha = torch.tensor([1e-6])
        high_alpha = input.abs().max()
    cnt = 0

    low_alpha = low_alpha.to(input.device)
    high_alpha = high_alpha.to(input.device)

    while cnt < max_iter and low_alpha < high_alpha:
        low_error = (input - common.quantize(input.clone(), low_alpha / (2**(bit-1)), bit)).pow(2).mean()
        high_error = (input - common.quantize(input.clone(), high_alpha / (2**(bit-1)), bit)).pow(2).mean()
        if low_error > high_error:
            low_alpha = low_alpha + (high_alpha - low_alpha) / 3
        else:
            high_alpha = high_alpha - (high_alpha - low_alpha) / 3

        cnt += 1

    return low_alpha

class DiffableQAWrapper(QAWrapper):
    def __init__(self, *args, **kwargs):
        super(DiffableQAWrapper, self).__init__(*args, **kwargs)
        self.quant = self.act_quant_diffable
        self.bcw = None
        self.bca = None
        self.deltas_act = None

        # Three params for weight quantizer
        self.phi = None
        self.delta = None
        self.bits = None

    @staticmethod
    def act_quant_diffable(input, delta, coded):
        return ActQuantize.apply(input, 2 ** delta, coded / input[0].numel())
    

    def get_group_input(self, input, group_id):
        if isinstance(self.layer, nn.Conv2d):
            return input[:, group_id*self.chans_per_group: (group_id+1)*self.chans_per_group]
        else:
            return input[..., group_id*self.chans_per_group: (group_id+1)*self.chans_per_group]

    def set_group_input(self, input, group_id, group_input):
        if isinstance(self.layer, nn.Conv2d):
            input[:, group_id*self.chans_per_group: (group_id+1)*self.chans_per_group] = group_input
        else:
            input[..., group_id*self.chans_per_group: (group_id+1)*self.chans_per_group] = group_input

    def get_group_weight(self, group_id):
        if isinstance(self.layer, nn.Conv2d):
            return self.layer.weight
        else:
            return input[..., group_id*self.chans_per_group: (group_id+1)*self.chans_per_group]

    def set_group_weight(self, group_id, group_weight):
        if isinstance(self.layer, nn.Conv2d):
            input[:, group_id*self.chans_per_group: (group_id+1)*self.chans_per_group] = group_input
        else:
            input[..., group_id*self.chans_per_group: (group_id+1)*self.chans_per_group] = group_input

    def forward(self, input):
         
        assert self.group_id == -1
        if self.quantized and self.mode == "in_channel":
            input_q = input.clone()
            for g in range(len(self.coded)):
                input_group = self.get_group_input(input, g)
                bit = self.coded[g] / input_group[0].numel()

                if self.adaptive_delta:
                    alpha = 2 ** (self.deltas_act[g] + bit - 1)
                    new_alpha = ternary_search_alpha(input_group, bit, estimate_alpha=alpha)
                    self.deltas_act[g] = torch.log2(new_alpha) - (bit-1)
                    
                input_group_q = self.quant(input_group, self.deltas_act[g], self.coded[g])

                if self.bca is not None:
                    input_group_q = self.bca(QAWrapper.permute(input_group), QAWrapper.permute(input_group_q))
                    input_group_q = QAWrapper.permute(input_group_q)
                
                self.set_group_input(input_q, g, input_group_q)
            input = input_q

        quant_weight = Quantize.apply(self.layer.weight, self.phi, self.delta, self.bits, self.chans_per_group)

        if self.bcw is not None:
            quant_weight = self.bcw(self.layer.weight, quant_weight)

        if isinstance(self.layer, nn.Conv2d):
            out = F.conv2d(input, quant_weight, 
                            self.layer.bias, self.layer.stride, self.layer.padding)
        else:
            out = F.linear(input, quant_weight, 
                            self.layer.bias)

        if self.quantized and self.mode == "out_channel":
            out_q = out.clone()
            for g in range(len(self.coded)):
                out_group = self.get_group_input(out, g)
                bit = self.coded[g] / out_group[0].numel()

                if self.adaptive_delta:
                    alpha = 2 ** (self.deltas_act[g] + bit - 1)
                    new_alpha = ternary_search_alpha(out_group, bit, estimate_alpha=alpha)
                    self.deltas_act[g] = torch.log2(new_alpha) - (bit-1)
                    
                out_group_q = self.quant(out_group, self.deltas_act[g], self.coded[g])

                if self.bca is not None:
                    try:
                        out_group_q = self.bca(QAWrapper.permute(out_group), QAWrapper.permute(out_group_q))
                        out_group_q = QAWrapper.permute(out_group_q)
                    except:
                        breakpoint()
                
                self.set_group_input(out_q, g, out_group_q)
            out = out_q
        return out