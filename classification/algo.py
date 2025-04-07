import common
import torch
import numpy as np
import scipy.io as io
import tunstall

import common
from tunstall import *
from common import *
from scipy.stats import entropy
from itertools import product
import rle
from time import time


INT_MIN = -2147483648
eps = 1e-12

def get_num_input_channels(tensor_weights):
    return tensor_weights.size()[1]

def get_ele_per_input_channel(tensor_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    return tensor_weights[:, 0, :, :].numel()

def get_input_channels(tensor_weights, st_id, ed_id):
    weights_copy = tensor_weights.clone()
    if len(weights_copy.shape) == 2:
        weights_copy = weights_copy[..., None, None]
    return weights_copy[:, st_id:ed_id, :, :]

def assign_input_channels(tensor_weights, st_id, ed_id, quant_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    tensor_weights[:, st_id:ed_id, :, :] = quant_weights
    return

def get_num_output_channels(tensor_weights):
    return tensor_weights.size()[0]

def get_ele_per_output_channel(tensor_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    return tensor_weights[0, :, :, :].numel()

def get_output_channels(tensor_weights, st_id, ed_id):
    weights_copy = tensor_weights.clone()
    if len(weights_copy.shape) == 2:
        weights_copy = weights_copy[..., None, None]
    return weights_copy[st_id:ed_id, :, :, :]

def get_output_channels_inds(tensor_weights, inds):
    weights_copy = tensor_weights.clone()
    return weights_copy[inds]

def get_input_channels_inds(tensor_weights, inds):
    weights_copy = tensor_weights.clone()
    return weights_copy[:, inds]

def assign_output_channels(tensor_weights, st_id, ed_id, quant_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    tensor_weights[st_id:ed_id, :, :, :] = quant_weights

def assign_input_channels(tensor_weights, st_id, ed_id, quant_weights):
    tensor_weights[:, st_id:ed_id, :, :] = quant_weights

def assign_output_channels_inds(tensor_weights, inds, quant_weights):
    tensor_weights[inds, ...] = quant_weights

def assign_input_channels_inds(tensor_weights, inds, quant_weights):
    tensor_weights[:, inds, ...] = quant_weights


#####################

def get_num_blocks(tensor_weights, block_size=8):
    return torch.tensor(tensor_weights.shape[:2]).float().div(block_size).ceil().long()
    

def get_ele_per_block(tensor_weights, block_size=8):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    return tensor_weights[0, 0, :, :].numel()

def get_block(tensor_weights, st_id_x, ed_id_x, st_id_y, ed_id_y):
    weights_copy = tensor_weights.clone()
    if len(weights_copy.shape) == 2:
        weights_copy = weights_copy[..., None, None]
    return weights_copy[st_id_x:ed_id_x, st_id_y:ed_id_y, :, :]

def assign_block(tensor_weights, st_id_x, ed_id_x, st_id_y, ed_id_y, quant_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    tensor_weights[st_id_x:ed_id_x, st_id_y:ed_id_y, :, :] = quant_weights
    return


# def get_ele_per_output_channel(tensor_weights):
#     if len(tensor_weights.shape) == 2:
#         tensor_weights = tensor_weights[..., None, None]
#     return tensor_weights[0, :, :, :].numel()


def deadzone_ratio(data, ratio):
    left = 0.0
    right = torch.max(data)
    max_itr = 50
    nitems = data.numel()

    if ratio < eps:
        return -1.0

    for i in range(0, max_itr):
        #print('left %f right %f' % (left, right))
        mid = (left + right) * 0.5
        num_deadzone = ((data >= (-1.0 * mid)) & (data <= mid)).float().sum()
        ratio_deadzone = 1.0 * num_deadzone / (1.0 * nitems)
        if ratio_deadzone > ratio:
            right = mid
        else:
            left = mid

        if(left + eps >= right):
            break

    return left


def deadzone_quantize(data, phi, stepsize, b):
    if b <= eps:
        return data * 0

    min_point = -(2**(b-1)) * stepsize
    max_point = (2**(b-1) - 1) * stepsize
    if phi < 0:
        return (((data / stepsize).round()) * stepsize).clamp(min_point , max_point)

    mask_deadzone = ((data < (-1.0 * phi)) | (data > phi))
    sign_data = torch.sign(data)
    q_data = (((torch.abs(data) - phi) / stepsize).round()).clamp(0, 2**(b-1) - 1)

    A = (q_data * stepsize + stepsize * 0.5 + phi)
    B = A * sign_data
    C = B * (mask_deadzone.float())

    return C

#    return (q_data * stepsize + stepsize * 0.5 + phi) * sign_data * mask_deadzone

def deadzone_quantize_idx(data, phi, stepsize, b):
    if b <= eps:
        return data * 0

    if phi < 0:
        return ((data / stepsize).round()).clamp(-2**(b-1) , 2**(b-1) - 1)

    mask_deadzone = ((data < (-1.0 * phi)) | (data > phi)).float()
    sign_data = torch.sign(data)
    q_data = (((torch.abs(data) - phi) / stepsize).round()).clamp(0, 2**(b-1) - 1) + 1

    return q_data * sign_data * mask_deadzone

def uniform_quantize_centroid(data, stepsize, ncentroid):
    min_point = -((ncentroid - 1)/2) * stepsize
    max_point = +((ncentroid - 1)/2) * stepsize
    return (((data / stepsize).round()) * stepsize).clamp(min_point , max_point)


def log_quantize(data, phi, stepsize, b):
    if b <= eps:
        return data * 0

    if phi > 0:
        mask_deadzone = ((data < (-1.0 * phi)) | (data > phi)).float()
    sign_data = torch.sign(data)
    q_data = (torch.log2(torch.abs(data) + eps) / stepsize).round().clamp(-2**(b-1), 0)

    A = q_data.exp2().mul(stepsize)
    B = A * sign_data

    C = B * (mask_deadzone.float()) if phi > 0 else B
    
    return B

def calc_entropy(data):
    data_copy = data.clone()
    #value, counts = np.unique(data_copy.cpu().numpy(), return_counts=True)
    value, counts = np.unique(data_copy.cpu().numpy(), return_counts=True)
    nitems = data.numel()
    probs = counts / nitems

    return entropy(probs, base=2)

def load_rd_curve(archname, layers, maxdeadzones, maxrates, datapath):
    nlayers = len(layers)
    rd_rate = [0] * nlayers
    rd_dist = [0] * nlayers
    rd_phi = [0] * nlayers
    rd_delta = [0] * nlayers

    for l in range(0, nlayers):
        [nfilters, ndepth, nheight, nwidth] = layers[l].weight.size()
        rd_rate[l] = [0] * nfilters
        rd_dist[l] = [0] * nfilters
        rd_phi[l] = [0] * nfilters
        rd_delta[l] = [0] * nfilters

        for f in range(0, nfilters):
            matpath = ('%s/%s_%03d_%04d.mat' % (datapath, archname, l, f))
            mat = io.loadmat(matpath)
            rd_rate[l][f] = mat['rd_rate']
            rd_dist[l][f] = mat['rd_dist']
            rd_phi[l][f] = mat['rd_phi']
            rd_delta[l][f] = mat['rd_delta']

        rd_rate[l] = np.array(rd_rate[l])
        rd_dist[l] = np.array(rd_dist[l])
        rd_phi[l] = np.array(rd_phi[l])
        rd_delta[l] = np.array(rd_delta[l])

    return rd_rate, rd_dist, rd_phi, rd_delta

def load_rd_curve_batch(archname, layers, maxdeadzones, maxrates, datapath, nchannelbatch, closedeadzone=0):
    nlayers = len(layers)
    rd_rate = [0] * nlayers
    rd_rate_entropy = [0] * nlayers
    rd_dist = [0] * nlayers
    rd_phi = [0] * nlayers
    rd_delta = [0] * nlayers
    rd_delta_mse = [0] * nlayers
    rd_dist_mse = [0] * nlayers

    for l in range(0, nlayers):
        nchannels = get_num_output_channels(layers[l].weight)
        nbatch = nchannels // nchannelbatch
        if (nchannels % nchannelbatch) != 0:
            nbatch += 1
        rd_rate[l] = [0] * nbatch
        rd_rate_entropy[l] = [0] * nbatch
        rd_dist[l] = [0] * nbatch
        rd_phi[l] = [0] * nbatch
        rd_delta[l] = [0] * nbatch
        rd_delta_mse[l] = [0] * nbatch
        rd_dist_mse[l] = [0] * nbatch
        cnt = 0

        for f in range(0, nchannels, nchannelbatch):
            matpath = ('%s/%s_%03d_%04d.mat' % (datapath, archname, l, f))
            mat = io.loadmat(matpath)
            rd_rate[l][cnt] = mat['rd_rate']
            rd_rate_entropy[l][cnt] = mat['rd_rate_entropy']
            rd_dist[l][cnt] = mat['rd_dist']
            rd_phi[l][cnt] = mat['rd_phi']
            rd_delta[l][cnt] = mat['rd_delta']
            rd_delta_mse[l][cnt] = mat['rd_delta_mse']
            rd_dist_mse[l][cnt] = mat['rst_dist_mse']
            #print(rd_rate[l][cnt])

            if closedeadzone == 1:
                rd_rate[l][cnt] = rd_rate[l][cnt][0:1,:]
                rd_rate_entropy[l][cnt] = rd_rate_entropy[l][cnt][0:1,:]
                rd_dist[l][cnt] = rd_dist[l][cnt][0:1,:]
                rd_phi[l][cnt] = rd_phi[l][cnt][0:1,:]
                rd_delta[l][cnt] = rd_delta[l][cnt][0:1,:]
                rd_delta_mse[l][cnt] = rd_delta_mse[l][cnt][0:1,:]
                rd_dist_mse[l][cnt] = rd_dist_mse[l][cnt][0:1,:]

            cnt += 1

        rd_rate[l] = np.array(rd_rate[l])
        rd_rate_entropy[l] = np.array(rd_rate_entropy[l])
        rd_dist[l] = np.array(rd_dist[l])
        rd_phi[l] = np.array(rd_phi[l])
        rd_delta[l] = np.array(rd_delta[l])
        rd_delta_mse[l] = np.array(rd_delta_mse[l])
        rd_dist_mse[l] = np.array(rd_dist_mse[l])

    return rd_rate, rd_rate_entropy, rd_dist, rd_phi, rd_delta, rd_delta_mse, rd_dist_mse


def load_rd_curve_blk(archname, layers, maxdeadzones, maxrates, datapath, blocksize, closedeadzone=0):
    nlayers = len(layers)
    rd_rate = [0] * nlayers
    rd_rate_entropy = [0] * nlayers
    rd_dist = [0] * nlayers
    rd_phi = [0] * nlayers
    rd_delta = [0] * nlayers
    rd_delta_mse = [0] * nlayers
    rd_dist_mse = [0] * nlayers

    for l in range(0, nlayers):
        # nchannels = get_num_input_channels(layers[l].weight)
        n_block_dims = get_num_blocks(layers[l].weight, blocksize)
        block_id_list = [x for x in product(range(n_block_dims[0]), range(n_block_dims[1]))]
        nbatch = len(block_id_list)
        rd_rate[l] = [0] * nbatch
        rd_rate_entropy[l] = [0] * nbatch
        rd_dist[l] = [0] * nbatch
        rd_phi[l] = [0] * nbatch
        rd_delta[l] = [0] * nbatch
        rd_delta_mse[l] = [0] * nbatch
        rd_dist_mse[l] = [0] * nbatch

        for cnt in range(len(block_id_list)):
            block_x, block_y = block_id_list[cnt]
            matpath = ('%s/%s_%03d_blk%04dx%04d.mat' % (datapath, archname, l, block_x, block_y))
            mat = io.loadmat(matpath)
            rd_rate[l][cnt] = mat['rd_rate']
            rd_rate_entropy[l][cnt] = mat['rd_rate_entropy']
            rd_dist[l][cnt] = mat['rd_dist']
            rd_phi[l][cnt] = mat['rd_phi']
            rd_delta[l][cnt] = mat['rd_delta']
            rd_delta_mse[l][cnt] = mat['rd_delta_mse']
            rd_dist_mse[l][cnt] = mat['rst_dist_mse']
            #print(rd_rate[l][cnt])

            if closedeadzone == 1:
                rd_rate[l][cnt] = rd_rate[l][cnt][0:1,:]
                rd_rate_entropy[l][cnt] = rd_rate_entropy[l][cnt][0:1,:]
                rd_dist[l][cnt] = rd_dist[l][cnt][0:1,:]
                rd_phi[l][cnt] = rd_phi[l][cnt][0:1,:]
                rd_delta[l][cnt] = rd_delta[l][cnt][0:1,:]
                rd_delta_mse[l][cnt] = rd_delta_mse[l][cnt][0:1,:]
                rd_dist_mse[l][cnt] = rd_dist_mse[l][cnt][0:1,:]


        rd_rate[l] = np.array(rd_rate[l])
        rd_rate_entropy[l] = np.array(rd_rate_entropy[l])
        rd_dist[l] = np.array(rd_dist[l])
        rd_phi[l] = np.array(rd_phi[l])
        rd_delta[l] = np.array(rd_delta[l])
        rd_delta_mse[l] = np.array(rd_delta_mse[l])
        rd_dist_mse[l] = np.array(rd_dist_mse[l])

    return rd_rate, rd_rate_entropy, rd_dist, rd_phi, rd_delta, rd_delta_mse, rd_dist_mse


def load_rd_curve_batch_less_items(archname, layers, maxdeadzones, maxrates, datapath, nchannelbatch, closedeadzone=0):
    nlayers = len(layers)
    rd_rate = [0] * nlayers
    rd_dist = [0] * nlayers
    rd_phi = [0] * nlayers
    rd_delta = [0] * nlayers

    for l in range(0, nlayers):
        # nchannels = get_num_input_channels(layers[l].weight)
        nchannels = get_num_output_channels(layers[l].weight)
        nbatch = nchannels // nchannelbatch
        if (nchannels % nchannelbatch) != 0:
            nbatch += 1
        rd_rate[l] = [0] * nbatch
        rd_dist[l] = [0] * nbatch
        rd_phi[l] = [0] * nbatch
        rd_delta[l] = [0] * nbatch
        cnt = 0

        for f in range(0, nchannels, nchannelbatch):
            matpath = ('%s/%s_%03d_%04d.mat' % (datapath, archname, l, f))
            mat = io.loadmat(matpath)
            rd_rate[l][cnt] = mat['rd_rate']
            rd_dist[l][cnt] = mat['rd_dist']
            rd_phi[l][cnt] = mat['rd_phi']
            rd_delta[l][cnt] = mat['rd_delta']

            if closedeadzone == 1:
                rd_rate[l][cnt] = rd_rate[l][cnt][0:1,:]
                rd_dist[l][cnt] = rd_dist[l][cnt][0:1,:]
                rd_phi[l][cnt] = rd_phi[l][cnt][0:1,:]
                rd_delta[l][cnt] = rd_delta[l][cnt][0:1,:]

            cnt += 1

        rd_rate[l] = np.array(rd_rate[l])
        rd_dist[l] = np.array(rd_dist[l])
        rd_phi[l] = np.array(rd_phi[l])
        rd_delta[l] = np.array(rd_delta[l])

    return rd_rate, rd_dist, rd_phi, rd_delta

def cal_total_num_weights(layers):
    nweights = 0
    nlayers = len(layers)

    for l in range(0, nlayers):
        n_filter_elements = layers[l].weight.numel()
        nweights += n_filter_elements
        # print('layer %d weights %d, %s' % (l, n_filter_elements, layers[l].weight.shape))

    return nweights


def pareto_condition(layers, rd_rate, rd_dist, rd_phi, rd_delta, slope_lambda):
    nlayers = len(layers)
    pc_phi = [0] * nlayers
    pc_delta = [0] * nlayers
    pc_bits = [0] * nlayers
    pc_rate = [0] * nlayers

    for l in range(0, nlayers):
        [nfilters, ndepth, nheight, nwidth] = layers[l].weight.size()
        pc_phi[l] = [0] * nfilters
        pc_delta[l] = [0] * nfilters
        pc_bits[l] = [0] * nfilters
        pc_rate[l] = [0] * nfilters

        for f in range(0, nfilters):
            y_0 = slope_lambda * rd_rate[l][f, :, :] + rd_dist[l][f, :, :]
            ind = np.unravel_index(np.argmin(y_0, axis=None), y_0.shape)
            (deadzone, bit) = ind
            deadzone = int(deadzone)
            bit = int(bit)
            pc_phi[l][f] = rd_phi[l][f, deadzone, bit]
            pc_delta[l][f] = rd_delta[l][f, deadzone, bit]
            pc_bits[l][f] = bit
            pc_rate[l][f] = rd_rate[l][f, deadzone, bit]

    return pc_phi, pc_delta, pc_bits, pc_rate

def pareto_condition_batch_equal_bit(layers, rd_rate, rd_dist, rd_phi, rd_delta, ave_bit, nchannelbatch):
    nlayers = len(layers)
    pc_phi = [0] * nlayers
    pc_delta = [0] * nlayers
    pc_bits = [0] * nlayers
    pc_rate = [0] * nlayers
    pc_size = [0] * nlayers

    for l in range(0, nlayers):
        nchannels = get_num_output_channels(layers[l].weight)
        n_channel_elements = get_ele_per_output_channel(layers[l].weight)

        nbatch = nchannels // nchannelbatch
        if (nchannels % nchannelbatch) != 0:
            nbatch += 1

        pc_phi[l] = [0] * nbatch
        pc_delta[l] = [0] * nbatch
        pc_bits[l] = [0] * nbatch
        pc_rate[l] = [0] * nbatch
        pc_size[l] = [0] * nbatch
        cnt = 0

        for f in range(0, nchannels, nchannelbatch):
            st_layer = f
            ed_layer = f + nchannelbatch
            if f + nchannelbatch > nchannels:
                ed_layer = nchannels

            deadzone = int(0)
            bit = int(ave_bit)
            pc_phi[l][cnt] = rd_phi[l][cnt, deadzone, bit]
            pc_delta[l][cnt] = rd_delta[l][cnt, deadzone, bit]
            pc_bits[l][cnt] = bit
            pc_rate[l][cnt] = rd_rate[l][cnt, deadzone, bit]
            pc_size[l][cnt] = bit * n_channel_elements * (ed_layer - st_layer)
            cnt = cnt + 1

    return pc_phi, pc_delta, pc_bits, pc_rate, pc_size


def pareto_condition_batch(layers, rd_rate, rd_dist, rd_phi, rd_delta, slope_lambda, nchannelbatch):
    nlayers = len(layers)
    pc_phi = [0] * nlayers
    pc_delta = [0] * nlayers
    pc_bits = [0] * nlayers
    pc_rate = [0] * nlayers
    pc_size = [0] * nlayers
    pc_dist_sum = 0

    for l in range(0, nlayers):
        nchannels = get_num_output_channels(layers[l].weight)
        n_channel_elements = get_ele_per_output_channel(layers[l].weight)
        nbatch = nchannels // nchannelbatch
        if (nchannels % nchannelbatch) != 0:
            nbatch += 1
        pc_phi[l] = [0] * nbatch
        pc_delta[l] = [0] * nbatch
        pc_bits[l] = [0] * nbatch
        pc_rate[l] = [0] * nbatch
        pc_size[l] = [0] * nbatch
        cnt = 0

        for f in range(0, nchannels, nchannelbatch):
            st_layer = f
            ed_layer = f + nchannelbatch
            if f + nchannelbatch > nchannels:
                ed_layer = nchannels

            y_0 = slope_lambda * (rd_rate[l][cnt, :, :])  + rd_dist[l][cnt, :, :]
            #y_0 = slope_lambda * (rd_rate[l][cnt, :, :]) * (ed_layer - st_layer) + rd_dist[l][cnt, :, :]

            ind = np.unravel_index(np.argmin(y_0, axis=None), y_0.shape)
            (deadzone, bit) = ind
            deadzone = int(deadzone)
            bit = int(bit)
            pc_phi[l][cnt] = rd_phi[l][cnt, deadzone, bit]
            pc_delta[l][cnt] = rd_delta[l][cnt, deadzone, bit]
            pc_bits[l][cnt] = bit
            pc_rate[l][cnt] = rd_rate[l][cnt, deadzone, bit]
            pc_size[l][cnt] = bit * n_channel_elements * (ed_layer - st_layer)
            pc_dist_sum += rd_dist[l][cnt, deadzone, bit]
            cnt = cnt + 1

    return pc_phi, pc_delta, pc_bits, pc_rate, pc_size


def pareto_condition_blk(layers, rd_rate, rd_dist, rd_phi, rd_delta, slope_lambda, blocksize):
    nlayers = len(layers)
    pc_phi = [0] * nlayers
    pc_delta = [0] * nlayers
    pc_bits = [0] * nlayers
    pc_rate = [0] * nlayers
    pc_size = [0] * nlayers
    pc_dist_sum = 0

    for l in range(0, nlayers):
        n_block_dims = get_num_blocks(layers[l].weight, blocksize)
        block_id_list = [x for x in product(range(n_block_dims[0]), range(n_block_dims[1]))]
        nbatch = len(block_id_list)
        n_channel_elements = get_ele_per_block(layers[l].weight)
        
        pc_phi[l] = [0] * nbatch
        pc_delta[l] = [0] * nbatch
        pc_bits[l] = [0] * nbatch
        pc_rate[l] = [0] * nbatch
        pc_size[l] = [0] * nbatch

        for cnt in range(len(block_id_list)):
            block_x, block_y = block_id_list[cnt]
            st_id_x = blocksize * block_x 
            st_id_y = blocksize * block_y
            ed_id_x = min(blocksize * (block_x + 1), layers[l].weight.shape[0])
            ed_id_y = min(blocksize * (block_y + 1), layers[l].weight.shape[1])

            y_0 = slope_lambda * (rd_rate[l][cnt, :, :])  + rd_dist[l][cnt, :, :]
            #y_0 = slope_lambda * (rd_rate[l][cnt, :, :]) * (ed_layer - st_layer) + rd_dist[l][cnt, :, :]
            ind = np.unravel_index(np.argmin(y_0, axis=None), y_0.shape)
            
            (deadzone, bit) = ind
            deadzone = int(deadzone)
            bit = int(bit)
            pc_phi[l][cnt] = rd_phi[l][cnt, deadzone, bit]
            pc_delta[l][cnt] = rd_delta[l][cnt, deadzone, bit]
            pc_bits[l][cnt] = bit
            pc_rate[l][cnt] = rd_rate[l][cnt, deadzone, bit]
            pc_size[l][cnt] = bit * n_channel_elements * (ed_id_x - st_id_x) * (ed_id_y - st_id_y)
            pc_dist_sum += rd_dist[l][cnt, deadzone, bit]

    return pc_phi, pc_delta, pc_bits, pc_rate, pc_size


def pareto_condition_batch_less_items(layers, rd_rate, rd_dist, rd_phi, rd_delta, slope_lambda, nchannelbatch):
    nlayers = len(layers)
    pc_phi = [0] * nlayers
    pc_delta = [0] * nlayers
    pc_bits = [0] * nlayers
    pc_rate = [0] * nlayers
    pc_size = [0] * nlayers

    for l in range(0, nlayers):
        nchannels = get_num_output_channels(layers[l].weight)
        n_channel_elements = get_ele_per_output_channel(layers[l].weight)
        nbatch = nchannels // nchannelbatch
        if (nchannels % nchannelbatch) != 0:
            nbatch += 1
        pc_phi[l] = [0] * nbatch
        pc_delta[l] = [0] * nbatch
        pc_bits[l] = [0] * nbatch
        pc_rate[l] = [0] * nbatch
        pc_size[l] = [0] * nbatch
        cnt = 0

        for f in range(0, nchannels, nchannelbatch):
            st_layer = f
            ed_layer = f + nchannelbatch
            if f + nchannelbatch > nchannels:
                ed_layer = nchannels

            #y_0 = slope_lambda * (rd_rate[l][cnt, :, :])  + rd_dist[l][cnt, :, :]
            y_0 = slope_lambda * (rd_rate[l][cnt, :, :]) * (ed_layer - st_layer) + rd_dist[l][cnt, :, :]

            ind = np.unravel_index(np.argmin(y_0, axis=None), y_0.shape)
            (deadzone, bit) = ind
            deadzone = int(deadzone)
            bit = int(bit)
            pc_phi[l][cnt] = rd_phi[l][cnt, deadzone, bit]
            pc_delta[l][cnt] = rd_delta[l][cnt, deadzone, bit]
            pc_bits[l][cnt] = bit
            pc_rate[l][cnt] = rd_rate[l][cnt, deadzone, bit] * (ed_layer - st_layer)
            pc_size[l][cnt] = bit * n_channel_elements * (ed_layer - st_layer)
            cnt = cnt + 1

    return pc_phi, pc_delta, pc_bits, pc_rate, pc_size


def cal_total_rates(layers, pc_rate, nchannelbatch):
    nlayers = len(layers)
    total_rates = 0

    for l in range(0, nlayers):
        #[nfilters, ndepth, nheight, nwidth] = layers[l].weight.size()
        #nfilters = layers[l].weight.size()[0]
        nchannels = get_num_output_channels(layers[l].weight)
        cnt = 0
        for f in range(0, nchannels, nchannelbatch):
            total_rates = total_rates + pc_rate[l][cnt]
            cnt = cnt + 1

    return total_rates

def cal_total_rates_blk(layers, pc_rate, blocksize):
    return sum(sum(r for r in pc_rate[l]) for l in range(len(layers)))


def cal_network_entropy(net, nfilterbatch=1):
    layers = common.findconv(net, False)
    nweights = cal_total_num_weights(layers)
    total_rate = 0.0

    with torch.no_grad():
        for l in range(0, len(layers)):
            layer_weights = layers[l].mod.weight.clone()
            numel = layer_weights[0].numel()
            cnt = 0 
            for f in range(0, layer_weights.shape[0], nfilterbatch):
                st_layer = f
                ed_layer = f + nfilterbatch
                if ed_layer > layer_weights.shape[0]:
                    ed_layer = layer_weights.shape[0]
                layer_weights[st_layer:ed_layer] = deadzone_quantize(layer_weights[st_layer:ed_layer], \
                                                                              layers[l].mod.phi[cnt], \
                                                                              2**layers[l].mod.delta[cnt], \
                                                                              layers[l].mod.bit[cnt])
                filter_entropy = calc_entropy(layer_weights[st_layer:ed_layer])
                total_rate = total_rate + filter_entropy * numel * (ed_layer - st_layer)
                cnt += 1

    return total_rate / nweights


def cal_network_tunstall_size(net, n_bit):
    layers = common.findconv(net, False)
    nweights = cal_total_num_weights(layers)
    total_rate = 0.0

    with torch.no_grad():
        for l in range(0, len(layers)):
            layer_weights = layers[l].mod.weight.clone()
            numel = layer_weightsnumel()
            for f in range(0, layer_weights.shape[0]):
                phi = layers[l].mod.phi[f]
                delta = layers[l].mod.delta[f]
                bit = layers[l].mod.bit[f]
                layer_weights[f] = deadzone_quantize_idx(layer_weights[f], phi, 2**delta, bit)

            values, counts = np.unique(layer_weights.cpu().numpy(), return_counts=True)
            probs = counts / counts.sum()
            tunstall_csize = calc_tunstall_codes_size(values, probs, n_bit)
            total_rate = total_rate + tunstall_csize * numel

    return total_rate / nweights
    

def cal_network_multistage_tunstall_size(net, n_bit_tunstall, n_stage, nfilterbatch=1):
    layers = common.findconv(net, False)
    nweights = cal_total_num_weights(layers)
    total_rate = 0.0

    with torch.no_grad():
        for l in range(0, len(layers)):
            layer_weights = layers[l].mod.weight.clone()
            numel = layer_weights.numel()
            cnt = 0
            for f in range(0, layer_weights.shape[0], nfilterbatch):
                st_layer = f
                ed_layer = f + nfilterbatch
                if ed_layer > layer_weights.shape[0]:
                    ed_layer = layer_weights.shape[0]
                phi = layers[l].mod.phi[cnt]
                delta = layers[l].mod.delta[cnt]
                filter_bit_allocation = layers[l].mod.bit[cnt]
                layer_weights[st_layer:ed_layer] = deadzone_quantize_idx(layer_weights[st_layer:ed_layer], phi, 2**delta, filter_bit_allocation)
                cnt += 1

            tunstall_csize = cal_multistage_tunstall_codes_size(layer_weights, n_bit_tunstall, n_stage)
            total_rate = total_rate + tunstall_csize * numel

    return total_rate / nweights


import copy
def bcorr_weight(weight, weight_q, vcorr_weight=True):
    """
    Reference: 
    https://github.com/submission2019/cnn-quantization/blob/cc580c8d4d33d199bdbdae8c4ba6b625520cc874/pytorch_quantizer/quantization/inference/inference_quantization_manager.py#L352
    """
    # if weight_q is not None:
    bias_q = weight_q.view(weight_q.shape[0], -1).mean(-1)
    bias_q = bias_q.view(bias_q.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else bias_q.view(bias_q.numel(), 1)
    bias_orig = weight.view(weight.shape[0], -1).mean(-1)
    bias_orig = bias_orig.view(bias_orig.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else bias_orig.view(bias_orig.numel(), 1)

    if vcorr_weight:
        eps = torch.tensor([1e-8]).to(weight_q.device)
        var_corr = weight.view(weight.shape[0], -1).std(dim=-1) / (weight_q.view(weight_q.shape[0], -1).std(dim=-1) + eps)
        var_corr = (var_corr.view(var_corr.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else var_corr.view(var_corr.numel(), 1))

        # Correct variance
        weight_q = (weight_q - bias_q) * var_corr + bias_q

    # Correct mean
    return weight_q - bias_q + bias_orig


def bcorr_act(act, act_q, vcorr_act=False):
    """
    Reference: 
    https://github.com/submission2019/cnn-quantization/blob/cc580c8d4d33d199bdbdae8c4ba6b625520cc874/pytorch_quantizer/quantization/inference/inference_quantization_manager.py#L180
    """
    temp = act.transpose(0, 1).contiguous().view(act.shape[1], -1)
    temp_q = act_q.transpose(0, 1).contiguous().view(act_q.shape[1], -1)
    q_bias = temp.mean(-1) - temp_q.mean(-1)
    if vcorr_act:
        var_coor = temp.std(-1) / temp_q.std(-1)
        var_coor = einops.rearrange(var_coor, 'c -> () c' + ' ()' * (len(act_q.shape) - 2))
        m_t = einops.rearrange(temp.mean(-1), 'c -> () c' + ' ()' * (len(act_q.shape) - 2))
        act_q = (act_q - m_t) * var_coor + m_t
        # import pdb; pdb.set_trace()
    act_q += einops.rearrange(q_bias, 'c -> () c' + ' ()' * (len(act_q.shape) - 2))
    return act_q


import einops
def bcorr_act_v2(act, act_q, scale, steps=20):
    act_ = act.transpose(0, 1).contiguous().view(-1, act.shape[1])
    act_q_ = act_q.transpose(0, 1).contiguous().view(-1, act_q.shape[1])
    err_mean = act_.mean(0) - act_q_.mean(0)
    res = act_ - act_q_
    min_dist = torch.ones_like(err_mean) * 1e30
    best_shift = torch.zeros_like(err_mean).cuda()
    for i in range(-steps, steps + 1):
        offset = err_mean + i * scale
        dist = (res - offset).pow(2).sum(0)
        mask = (dist < min_dist).int()
        min_dist = min_dist * (1 - mask) + dist * mask
        best_shift = best_shift * (1 - mask) + offset * mask
    return act_q + einops.rearrange(best_shift, 'c -> () c' + ' ()' * (len(act_q.shape) - 2))


from functools import partial

def bcorr_act_factory(version=2, *args, **kwargs):
    if version == 1:
        return bcorr_act
    elif version == 2:
        return partial(bcorr_act_v2, *args, **kwargs)


def greedy_quantize(layers, rd_rate, rd_dist, rd_phi, rd_delta, 
                rd_act_dist, rd_act_bits, rd_act_delta, act_sizes, 
                nchannelbatch, target_rate):

    chngrp_indices = []
    nlayers = len(layers)
    i = 0
    for l in range(nlayers):
        chngrp_indices.append([])
        for g in range(len(rd_dist[l])):
            chngrp_indices[l].append(i)
            i += 1

    nweights = []
    nweights_layer = []
    for l in range(nlayers):
        nweights_layer.append([])
        for g in range(len(chngrp_indices[l])):
            nweights_layer[l].append(layers[l].weight[g * nchannelbatch: (g + 1) * nchannelbatch].numel())
        nweights += nweights_layer[l]
    
    for l in range(nlayers):
        nweights += act_sizes[l]

    total_weights = sum(nweights)

    for l in range(nlayers):
        chngrp_indices.append([])
        for g in range(len(rd_act_dist[l])):
            chngrp_indices[l + nlayers].append(i)
            i += 1

    rd_dist_uvld = [np.min(rd_dist[l][g], axis=0) for l in range(len(rd_dist)) for g in range(len(rd_dist[l]))]
    phi_select = [[np.argmin(rd_dist[l][g], axis=0).astype(int) for g in range(len(rd_dist[l]))] for l in range(len(rd_dist))]
    rd_rate_uvld = [rd_rate[l][g][0] / nweights_layer[l][g] for l in range(len(rd_rate)) for g in range(len(rd_rate[l]))]
    rd_dist_uvld += [rd_act_dist[l][g] for l in range(len(rd_act_dist)) for g in range(len(rd_act_dist[l]))]
    rd_rate_uvld += [rd_act_bits[l][g] for l in range(len(rd_act_bits)) for g in range(len(rd_act_bits[l]))]
    nchngrps = len(rd_dist_uvld)
    R = int(target_rate * total_weights)

    grd_bits = [0 for _ in range(nchngrps)]

    while R > 0:
        slopes = []
        for n in range(nchngrps):
            if grd_bits[n] < len(rd_dist_uvld[n]) - 1:
                slopes.append(rd_dist_uvld[n][grd_bits[n]] - rd_dist_uvld[n][grd_bits[n]+1])
            else:
                slopes.append(INT_MIN)
        max_slope_idx = slopes.index(max(slopes))
        grd_bits[max_slope_idx] += 1
        R -= nweights[max_slope_idx]
    
    grd_layer_bits = [np.array([grd_bits[chngrp_indices[l][g]] for g in range(len(chngrp_indices[l]))]).astype(int) for l in range(nlayers)]
    grd_phi = [np.array([rd_phi[l][g, phi_select[l][g][grd_layer_bits[l][g]], grd_layer_bits[l][g]] for g in range(len(grd_layer_bits[l]))]) for l in range(nlayers)]
    grd_delta = [np.array([rd_delta[l][g, phi_select[l][g][grd_layer_bits[l][g]], grd_layer_bits[l][g]] for g in range(len(grd_layer_bits[l]))]) for l in range(nlayers)]
    grd_size_layer = [grd_layer_bits[l] * np.array(nweights_layer[l]) for l in range(nlayers)]

    grd_act_bits_layer = [np.array([grd_bits[chngrp_indices[l][g]] for g in range(len(chngrp_indices[l]))]).astype(int) for l in range(nlayers, nlayers*2)]
    grd_act_delta = [np.array([rd_act_delta[l][g][grd_act_bits_layer[l][g]] for g in range(len(grd_act_bits_layer[l]))]) for l in range(nlayers)]
    grd_act_coded = [np.array([act_sizes[l][g] * grd_act_bits_layer[l][g] for g in range(len(grd_act_bits_layer[l]))]) for l in range(nlayers)]
    grd_act_dist = [np.array([rd_act_dist[l][g][grd_act_bits_layer[l][g]] for g in range(len(grd_act_bits_layer[l]))]) for l in range(nlayers)]

    return grd_phi, grd_delta, grd_layer_bits, grd_size_layer, grd_act_dist, grd_act_bits_layer, grd_act_delta, grd_act_coded



import tqdm, pickle, os, rle

def dp_quantize(layers, rd_rate, rd_dist, rd_phi, rd_delta, 
                rd_act_dist, rd_act_bits, rd_act_delta, act_sizes, 
                nchannelbatch, target_rate, device="cuda", piece_length=4096, G_dir="", act_dist_penalty_scale=1.0, smooth_dists=False, manual_offset=0):

    device = torch.device(device)

    chngrp_indices = []
    nlayers = len(layers)
    i = 0
    for l in range(nlayers):
        chngrp_indices.append([])
        for g in range(len(rd_dist[l])):
            chngrp_indices[l].append(i)
            i += 1

    num_weight_groups = i
    nweights = []
    nweights_layer = []
    for l in range(nlayers):
        nweights_layer.append([])
        for g in range(len(chngrp_indices[l])):
            nweights_layer[l].append(layers[l].weight[g * nchannelbatch: (g + 1) * nchannelbatch].numel())
        nweights += nweights_layer[l]
    
    for l in range(nlayers):
        nweights += act_sizes[l]

    total_weights = sum(nweights)
    sorted_chngrp_indices = np.argsort(nweights)
    sorted_nweights = sorted(nweights)

    for l in range(nlayers):
        chngrp_indices.append([])
        for g in range(len(rd_act_dist[l])):
            chngrp_indices[l + nlayers].append(i)
            i += 1

    rd_dist_uvld = [np.min(rd_dist[l][g], axis=0) for l in range(len(rd_dist)) for g in range(len(rd_dist[l]))]
    phi_select = [[np.argmin(rd_dist[l][g], axis=0).astype(int) for g in range(len(rd_dist[l]))] for l in range(len(rd_dist))]
    rd_rate_uvld = [rd_rate[l][g][0] / nweights_layer[l][g] for l in range(len(rd_rate)) for g in range(len(rd_rate[l]))]
    rd_dist_uvld += [rd_act_dist[l][g] * act_dist_penalty_scale for l in range(len(rd_act_dist)) for g in range(len(rd_act_dist[l]))]
    rd_rate_uvld += [rd_act_bits[l][g] for l in range(len(rd_act_bits)) for g in range(len(rd_act_bits[l]))]
    sorted_rd_dist = [torch.tensor(rd_dist_uvld[sort_idx]).half().to(device)  for sort_idx in sorted_chngrp_indices]
    sorted_rd_rate = [torch.tensor(rd_rate_uvld[sort_idx]).to(device) for sort_idx in sorted_chngrp_indices]

    if smooth_dists:
        for i in range(len(sorted_rd_dist)):
            sorted_rd_dist[i], _ = make_nonincreasing(sorted_rd_dist[i], sorted_rd_rate[i])

    nchngrps = len(rd_dist_uvld)
    R = int(target_rate * total_weights)
    
    tmp_g_dir = G_dir or f"./G_tmps_rate{target_rate}/" 
    if not os.path.exists(tmp_g_dir):
        os.makedirs(tmp_g_dir)
    if len(os.listdir(tmp_g_dir)) < nchngrps:
        # G = pickle.load(open(tmp_g_path, "rb"))

        # dp = torch.ones((nchngrps, R), device=device) * float('inf')
        inf_tensor = torch.tensor([float("inf")], device=device, dtype=torch.half)
        dp = torch.ones((2, R), device=device, dtype=torch.half) * inf_tensor

        # G = torch.zeros((nchngrps, R), dtype=torch.uint8, device=device)  # result indices of rd_phi (pruning amounts)

        # sizes = torch.tensor(list(range(R)), dtype=torch.long, device=device)
        max_bits = max(sorted_rd_rate[n].max() + 1 for n in range(nchngrps))
        # sizes = einops.repeat(sizes, "n -> n p", p=max_bits)

        closest_value_index_torch = lambda float_list, target: torch.argmin((float_list - target).abs())
        closest_value_index_2d = lambda float_list, target: torch.argmin((float_list - target).abs(), dim=1, keepdim=True)
        for n in tqdm.tqdm(range(nchngrps)):
            max_weights_n = min(sum(sorted_nweights[:n+1]) * max_bits, R)
            G = torch.zeros((max_weights_n,), dtype=torch.uint8, device=device)
            G[0] = closest_value_index_torch(sorted_rd_rate[n], 0)
            dp[1, 0] = sorted_rd_dist[n][G[0].item()] + (dp[0, 0] if n > 1 else 0)


            sizes_ = torch.tensor(list(range(1, 1+piece_length)), dtype=torch.long, device=device)
            sizes_ = einops.repeat(sizes_, "n -> n p", p=sorted_rd_rate[n].max().int().item() + 1).clone()

            num_pieces = torch.tensor((max_weights_n - 1) / piece_length).ceil().int().item()
            for ip in range(num_pieces):
                start = 1 + ip * piece_length
                end = min(max_weights_n, start + piece_length)
                # sizes_ = sizes[start: end, :len(sorted_rd_rate[n])]
                sizes_ = sizes_[: min(piece_length, end - start)]
                if ip in [0, num_pieces - 1]: # avoid redundant repeat operation
                    sorted_rd_rate_ = einops.repeat(sorted_rd_rate[n], "p -> n p", n=len(sizes_))
                    sorted_rd_dist_ = einops.repeat(sorted_rd_dist[n], "p -> n p", n=len(sizes_))


                allowed_bit_masks = (sizes_ > (sorted_nweights[n] * sorted_rd_rate_).floor())
                if n == 0:
                    G[start: end] = closest_value_index_2d(sorted_rd_rate_, (sizes_[:, 0] / sorted_nweights[n])[:, None])[:, 0]
                    dp[1, start: end] = torch.index_select(sorted_rd_dist[n], 0, G[start: end].long())
                else:
                    f_all_n = torch.where(allowed_bit_masks, sorted_rd_dist_ + dp[0, sizes_ - (sorted_nweights[n] * sorted_rd_rate_).floor().long()], inf_tensor) 
                    G[start: end] = torch.argmin(f_all_n, dim=1)
                    dp[1, start: end] = torch.min(f_all_n, dim=1)[0]
                    del f_all_n
                
                sizes_ += piece_length
            # import pdb; pdb.set_trace()
            if max_weights_n < R:
                dp[1, max_weights_n:] = dp[1, max_weights_n - 1]
            
            dp[0, :] = dp[1, :]
            del allowed_bit_masks, sizes_

            values, counts = run_length_encoding(G)

            with open(tmp_g_dir + f"/G_{n}.pkl", "wb") as f:
                pickle.dump({"values": values, "counts": counts}, f)

    dp_bits = [0 for _ in range(nchngrps)]
    remained_size = R
    for sort_idx, n in zip(sorted_chngrp_indices[::-1], range(nchngrps-1, -1, -1)):
        G_n_encoded = pickle.load(open(tmp_g_dir + f"/G_{n}.pkl", "rb"))
        G_n = run_length_decoding(G_n_encoded["values"], G_n_encoded["counts"]).to(device)
        dp_bits[sort_idx] = sorted_rd_rate[n][G_n[min(len(G_n) - 1, max(0, remained_size - 1))].long()].item()
        remained_size = max(0, remained_size - int(dp_bits[sort_idx] * sorted_nweights[n]))

    for j in range(len(dp_bits)):
        dp_bits[j] = min(manual_offset + dp_bits[j], sorted_rd_rate[n].max().item())

    dp_bits_layer = [np.array([dp_bits[chngrp_indices[l][g]] for g in range(len(chngrp_indices[l]))]).astype(int) for l in range(nlayers)]
    dp_phi = [np.array([rd_phi[l][g, phi_select[l][g][dp_bits_layer[l][g]], dp_bits_layer[l][g]] for g in range(len(dp_bits_layer[l]))]) for l in range(nlayers)]
    dp_delta = [np.array([rd_delta[l][g, phi_select[l][g][dp_bits_layer[l][g]], dp_bits_layer[l][g]] for g in range(len(dp_bits_layer[l]))]) for l in range(nlayers)]
    # dp_phi = [np.array([rd_phi[l][g, phi_select[l][g][dp_bits_layer[l][g]], dp_bits_layer[l][g]] for g in range(len(dp_bits_layer[l]))]) for l in range(nlayers)]
    dp_size_layer = [dp_bits_layer[l] * np.array(nweights_layer[l]) for l in range(nlayers)]

    dp_act_bits_layer = [np.array([dp_bits[chngrp_indices[l][g]] for g in range(len(chngrp_indices[l]))]).astype(int) for l in range(nlayers, nlayers*2)]
    dp_act_delta = [np.array([rd_act_delta[l][g][dp_act_bits_layer[l][g]] for g in range(len(dp_act_bits_layer[l]))]) for l in range(nlayers)]
    dp_act_coded = [np.array([act_sizes[l][g] * dp_act_bits_layer[l][g] for g in range(len(dp_act_bits_layer[l]))]) for l in range(nlayers)]
    dp_act_dist = [np.array([rd_act_dist[l][g][dp_act_bits_layer[l][g]] for g in range(len(dp_act_bits_layer[l]))]) for l in range(nlayers)]

    
    # for f in os.listdir(tmp_g_dir):
    #     os.remove(os.path.join(tmp_g_dir, f))
    # os.removedirs(tmp_g_dir)

    return dp_phi, dp_delta, dp_bits_layer, dp_size_layer, dp_act_dist, dp_act_bits_layer, dp_act_delta, dp_act_coded


def lg_quantize(layers, rd_rate, rd_dist, rd_phi, rd_delta, 
                rd_act_dist, rd_act_bits, rd_act_delta, act_sizes, 
                nchannelbatch, target_rate, act_dist_penalty_scale=1.0, smooth_dists=False, minrate=0):

    chngrp_indices = []
    nlayers = len(layers)
    i = 0
    for l in range(nlayers):
        chngrp_indices.append([])
        for g in range(len(rd_dist[l])):
            chngrp_indices[l].append(i)
            i += 1

    num_weight_groups = i
    nweights = []
    nweights_layer = []
    for l in range(nlayers):
        nweights_layer.append([])
        for g in range(len(chngrp_indices[l])):
            nweights_layer[l].append(layers[l].weight[g * nchannelbatch: (g + 1) * nchannelbatch].numel())
        nweights += nweights_layer[l]
    
    for l in range(nlayers):
        nweights += act_sizes[l]

    total_weights = sum(nweights)
    sorted_chngrp_indices = np.argsort(nweights)
    sorted_nweights = sorted(nweights)

    for l in range(nlayers):
        chngrp_indices.append([])
        for g in range(len(rd_act_dist[l])):
            chngrp_indices[l + nlayers].append(i)
            i += 1

    rd_dist_uvld = [np.min(rd_dist[l][g], axis=0) for l in range(len(rd_dist)) for g in range(len(rd_dist[l]))]
    phi_select = [[np.argmin(rd_dist[l][g], axis=0).astype(int) for g in range(len(rd_dist[l]))] for l in range(len(rd_dist))]
    rd_rate_uvld = [rd_rate[l][g][0] / nweights_layer[l][g] for l in range(len(rd_rate)) for g in range(len(rd_rate[l]))]
    rd_dist_uvld += [rd_act_dist[l][g] * act_dist_penalty_scale for l in range(len(rd_act_dist)) for g in range(len(rd_act_dist[l]))]
    rd_rate_uvld += [rd_act_bits[l][g] for l in range(len(rd_act_bits)) for g in range(len(rd_act_bits[l]))]
    sorted_rd_dist = [rd_dist_uvld[sort_idx] for sort_idx in sorted_chngrp_indices]
    sorted_rd_rate = [rd_rate_uvld[sort_idx] for sort_idx in sorted_chngrp_indices]
    sorted_nweights = sorted(nweights)

    if smooth_dists:
        for i in range(len(sorted_rd_dist)):
            sorted_rd_dist[i], _ = make_nonincreasing(sorted_rd_dist[i], sorted_rd_rate[i])
            sorted_rd_dist[i] = smooth(sorted_rd_dist[i], 0.2)

    nchngrps = len(rd_dist_uvld)
    R = int(target_rate * total_weights)

    lg_bits = [0 for _ in range(nchngrps)]
    
    def target_func(slope):
        size = 0
        for i in range(nchngrps):
            ys_0 = sorted_rd_dist[i] + (2 ** slope) * sorted_rd_rate[i]
            point = max(np.argmin(ys_0), minrate)
            size += sorted_rd_rate[i][point] * sorted_nweights[i]
        ret = - size.astype(float) / total_weights
        # print(f"Slope: {slope}, Size: {ret}")
        return ret

    begin = time()
    slope = common.binary_search(-1000, 0, target_func, -target_rate, max_iters=200)

    print(f"Slope search Time taken: {time() - begin}")
    print(f"Slope: {slope}")

    for i in range(nchngrps):
        ys_0 = sorted_rd_dist[i] + (2 ** slope) * sorted_rd_rate[i]
        point = max(np.argmin(ys_0), minrate)
        lg_bits[i] = sorted_rd_rate[i][point]


    lg_bits_layer = [np.array([lg_bits[chngrp_indices[l][g]] for g in range(len(chngrp_indices[l]))]).astype(int) for l in range(nlayers)]
    lg_phi = [np.array([rd_phi[l][g, phi_select[l][g][lg_bits_layer[l][g]], lg_bits_layer[l][g]] for g in range(len(lg_bits_layer[l]))]) for l in range(nlayers)]
    lg_delta = [np.array([rd_delta[l][g, phi_select[l][g][lg_bits_layer[l][g]], lg_bits_layer[l][g]] for g in range(len(lg_bits_layer[l]))]) for l in range(nlayers)]
    # lg_phi = [np.array([rd_phi[l][g, phi_select[l][g][lg_bits_layer[l][g]], lg_bits_layer[l][g]] for g in range(len(lg_bits_layer[l]))]) for l in range(nlayers)]
    lg_size_layer = [lg_bits_layer[l] * np.array(nweights_layer[l]) for l in range(nlayers)]

    lg_act_bits_layer = [np.array([lg_bits[chngrp_indices[l][g]] for g in range(len(chngrp_indices[l]))]).astype(int) for l in range(nlayers, nlayers*2)]
    lg_act_delta = [np.array([rd_act_delta[l][g][lg_act_bits_layer[l][g]] for g in range(len(lg_act_bits_layer[l]))]) for l in range(nlayers)]
    lg_act_coded = [np.array([act_sizes[l][g] * lg_act_bits_layer[l][g] for g in range(len(lg_act_bits_layer[l]))]) for l in range(nlayers)]
    lg_act_dist = [np.array([rd_act_dist[l][g][lg_act_bits_layer[l][g]] for g in range(len(lg_act_bits_layer[l]))]) for l in range(nlayers)]

    
    # for f in os.listdir(tmp_g_dir):
    #     os.remove(os.path.join(tmp_g_dir, f))
    # os.removedirs(tmp_g_dir)

    return lg_phi, lg_delta, lg_bits_layer, lg_size_layer, lg_act_dist, lg_act_bits_layer, lg_act_delta, lg_act_coded



def run_length_encoding(x):
    # Flatten the input tensor
    x_flat = x.view(-1)
    
    # Identify the start of each run
    change_indices = torch.cat((torch.tensor([0], device=x.device), (x_flat[1:] != x_flat[:-1]).nonzero(as_tuple=False).squeeze(1) + 1, torch.tensor([x_flat.size(0)], device=x.device)))
    
    # Calculate the lengths of each run
    run_lengths = change_indices[1:] - change_indices[:-1]
    
    # Gather the values of each run
    run_values = x_flat[change_indices[:-1]]
    
    return run_values, run_lengths

def run_length_decoding(values, lengths):
    # Calculate the total length of the decoded tensor
    total_length = lengths.sum().item()
    
    # Initialize an empty tensor to store the decoded values
    decoded = torch.empty(total_length, dtype=values.dtype, device=values.device)
    
    # Use cumulative sum to get the end indices of each run
    end_indices = lengths.cumsum(dim=0)
    
    # Use the end indices to fill the decoded tensor
    start_index = 0
    for value, end_index in zip(values, end_indices):
        decoded[start_index:end_index] = value
        start_index = end_index
    
    return decoded

@torch.no_grad()
def get_bops_model(net, layers, output_dimens, bit_weights, return_base=False):
    net.eval()

    layer_sizes = [torch.tensor(layer.weight.shape) for layer in layers]
    coded_acts = [layers[i].coded for i in range(0, len(layers))]

    nom_bops = 0.0
    denom_bops = 0.0

    for o_dim, size, b_w, c_a, m in zip(output_dimens, layer_sizes, bit_weights, coded_acts, layers):
        b_a_mean = sum(c_a) / o_dim.prod()
        if isinstance(m.layer, torch.nn.Conv2d):
            nom_bops += o_dim[0] * o_dim[-2:].prod() * size.prod() * np.array(b_w).mean() * b_a_mean + (0 if m.layer.bias is None else o_dim.prod() * 32 * 32)
            denom_bops += o_dim[0] * o_dim[-2:].prod() * size.prod() * 32 * 32 + (0 if m.layer.bias is None else o_dim.prod() * 32 * 32)
        elif isinstance(m.layer, torch.nn.Linear):
            nom_bops += o_dim[:-1].prod() * size.prod() * np.array(b_w).mean() * b_a_mean + (0 if m.layer.bias is None else o_dim.prod() * 32 * 32)
            denom_bops += o_dim[:-1].prod() * size.prod() * 32 * 32 + (0 if m.layer.bias is None else o_dim.prod() * 32 * 32)

    if return_base:
        return nom_bops, denom_bops
    return nom_bops / denom_bops


def make_nondecreasing(rd_dists, rd_amounts):
    for i in range(1, len(rd_dists)):
        rd_dists[i] = max(rd_dists[i-1], rd_dists[i])
    return rd_dists, rd_amounts

def make_nonincreasing(rd_dists, rd_amounts):
    for i in range(1, len(rd_dists)):
        rd_dists[i] = min(rd_dists[i-1], rd_dists[i])
    return rd_dists, rd_amounts

def smooth(rd_dists, weight=0.9):
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = copy.deepcopy(rd_dists)
    
    num_acc = 0
    for next_val in rd_dists:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - weight ** num_acc
        smoothed_val = last / debias_weight
        smoothed[num_acc-1] = smoothed_val

    return smoothed