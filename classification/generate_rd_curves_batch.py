import argparse
import pdb

import torch
import os
import common
import algo
import header

import scipy.io as io
from common import *
from algo import *
from header import *
from time import time
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

parser = argparse.ArgumentParser(description='generate rate-distortion curves')
parser.add_argument('--archname', default='mobilenetv2py', type=str,
                    help='name of network architecture: resnet18, resnet34, resnet50, densenet, etc')
parser.add_argument('--maxsteps', default=48, type=int,
                    help='number of Delta to enumerate')
parser.add_argument('--maxdeadzones', default=10, type=int,
                    help='number of sizes of dead zones')
parser.add_argument('--maxrates', default=11, type=int,
                    help='number of bit rates')
parser.add_argument('--gpuid', default=0, type=int,
                    help='gpu id')
parser.add_argument('--datapath', default='./ImageNet2012/', type=str,
                    help='imagenet dateset path')
parser.add_argument('--testsize', default=64, type=int,
                    help='number of images to run')
parser.add_argument('--numdatareads', default=2, type=int,
                    help='num threads to read images from file')
parser.add_argument('--batchsize', default=64, type=int,
                    help='batch size')
parser.add_argument('--nchannelbatch', default=16, type=int,
                    help='number of filters for each quantization batch')
parser.add_argument('--nprocessings', default=4, type=int,
                    help='total number of threads to run')
parser.add_argument('--modeid', default=0, type=int,
                    help='all filters (with N_filter % nprocessings = modeid) will be processed')
parser.add_argument('--override-checkpoint', default='', type=str)
parser.add_argument('--part_id', default=0, type=int, help="break total layers into parts and process each part in each process.")
parser.add_argument('--num_parts', default=5, type=int)
parser.add_argument('--bias_corr_weight', '-bcw', action="store_true")
parser.add_argument('--quant-type', default='linear', type=str, choices=['linear', 'log2'],)
parser.add_argument('--disable-deadzone', action="store_true")
parser.add_argument('--sort_channels', action="store_true")
parser.add_argument('--gen_approx_data', action="store_true")
# parser.add_argument('--bias_corr_act', '-bca', action="store_true", type=int)

args = parser.parse_args()
args.val_testsize = args.testsize

if args.disable_deadzone:
    args.maxdeadzones = 1


net = loadnetwork(args.archname, args.gpuid)

path_output = ('%s_ndz_%04d_nr_%04d_ns_%04d_nf_%04d_%srdcurves_%schannelwise_%sopt_dist' % (args.archname, args.maxdeadzones, args.maxrates, \
                                                                   args.testsize, args.nchannelbatch, "bcw_" if args.bias_corr_weight else "", "sorted_" if args.sort_channels else "", "log2_" if args.quant_type == "log2" else ""))

if args.override_checkpoint != '':
    all_ = net.load_state_dict(torch.load(args.override_checkpoint), strict=False)
    print(all_)
    print('load checkpoint from %s' % args.override_checkpoint)
    path_output = f"ckpt_{args.override_checkpoint.split('/')[-1].split('.')[0]}_{path_output}"

isExists=os.path.exists(path_output)
if not isExists:
    os.makedirs(path_output)

# images, labels = loadvaldata(args.datapath, args.gpuid, testsize=args.testsize)
if "vit" in args.archname and "mae" not in args.archname:
    args.mean = [0.5,] * 3
    args.std = [0.5,] * 3
else:
    args.mean = IMAGENET_DEFAULT_MEAN
    args.std = IMAGENET_DEFAULT_STD
if "384" in args.archname:
    loader = get_calib_imagenet_dali_loader(args, 384, 384)
else:
    loader = get_calib_imagenet_dali_loader(args)

layers = findconv(net, False)
print('total number of layers: %d' % (len(layers)))

for l in range(0, len(layers)):
    layer_weights = layers[l].weight.clone()
    nchannels = get_num_output_channels(layer_weights)
    n_channel_elements = get_ele_per_output_channel(layer_weights)

net.eval()
# loader = torch.utils.data.DataLoader(images, batch_size=args.batchsize)
Y, labels = predict_dali_withgt(net, loader)
#Y, labels = predict_dali(net, loader)

top_1, top_5 = accuracy(Y, labels, topk=(1,5))
print('original network %s accuracy: top1 %5.2f top5 %5.2f' % (args.archname, top_1, top_5))

if args.gen_approx_data:
    grad_list = []
    net.train()
    for c, data in enumerate(loader):
        try:
            x = data[0]["data"]
        except:
            x, _ = data
            x = x.cuda()
        res = torch.norm(net(x), p=2) 
        res.backward()
        for idx,m in enumerate(layers):
            if len(grad_list) < len(layers):
                grad_list.append(m.weight.grad.data / len(loader))
            else:
                grad_list[idx] += m.weight.grad.data / len(loader)
        for p in net.parameters():
            if p.grad is not None:
                torch.nn.init.zeros_(p.grad.data)

import math
len_part = math.ceil(len(layers) / args.num_parts)

with torch.no_grad():
    for layerid in range(args.part_id * len_part, min((args.part_id + 1) * len_part, len(layers))):
        layer_weights = layers[layerid].weight.clone()
        nchannels = get_num_output_channels(layer_weights)
        n_channel_elements = get_ele_per_output_channel(layer_weights)
        print('filter size %d %d ' % (nchannels, n_channel_elements))
        if args.sort_channels:
            inds_channel_range = torch.argsort(layer_weights.view(nchannels, -1).max(1)[0])
            # rev_range_per_channel = torch.argsort(range_per_channel)

        for f in range(0, nchannels, args.nchannelbatch):
            
            st_id = f
            if f + args.nchannelbatch > nchannels:
                ed_id = nchannels
            else:
                ed_id = f + args.nchannelbatch
            
            if args.sort_channels:
                inds = inds_channel_range[st_id: ed_id]
            else:
                inds = list(range(st_id, ed_id))

            rst_phi = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_delta = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_entropy = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_dist = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_rate = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_rate_entropy = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_delta_mse = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_dist_mse = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            # rst_approx_dist = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())

            #scale = (layer_weights[st_id:ed_id, :].reshape(-1) ** 2).mean().sqrt().log2().floor()
            output_channels = get_output_channels_inds(layer_weights, inds)
            scale = (output_channels.reshape(-1) ** 2).mean().sqrt().log2().floor()
            
            end = time()

            for d in range(0, args.maxdeadzones):
                phi = deadzone_ratio(output_channels, d / args.maxdeadzones)
                for b in range(0, args.maxrates):

                    if b == 0:
                        start = scale - 10
                    else:
                        start = rst_delta[d, b-1] - 2

                    min_dist = 1e8
                    min_mse = 1e8
                    pre_mse = 1e8
                    pre_dist = 1e8
                    for s in range(0, args.maxsteps):
                        delta = start + s
                        quant_weights = output_channels.clone()
                        if args.quant_type == 'linear':
                            quant_weights = \
                                deadzone_quantize(output_channels, phi, 2**delta, b)
                        elif args.quant_type == 'log2':
                            quant_weights = \
                                log_quantize(output_channels, phi, 2**delta, b)
                        
                        if args.bias_corr_weight:
                            quant_weights = bcorr_weight(output_channels, quant_weights)

                        cur_mse = ((quant_weights - output_channels)**2).mean()
                        # pdb.set_trace()
                        assign_output_channels_inds(layers[layerid].weight, inds, quant_weights)
                        Y_hat = predict_dali(net, loader)
                        cur_dist = ((Y - Y_hat) ** 2).mean()

                        top_1 = accuracy(Y_hat, labels, topk=(1,))[0]
                        # print('%s | layer %d: filter %d deadzone ratio %6.6f bit rates %6.6f s %d: phi %2.12f delta %6.6f mse %6.6f entropy %6.6f rate %6.6f distortion %6.6f Top-1 %.2f | time %f' \
                        #     % (args.archname, layerid, f, d / args.maxdeadzones, b, s, phi, delta, \
                        #        cur_mse, b, b * n_channel_elements, cur_dist, top_1, time() - end))
                        end = time()

                        # rst_approx_dist[d, b] = (quant_weights - output_channels) * get_output_channels_inds(grad_list[layerid], inds)

                        if (cur_dist < min_dist):
                            rst_phi[d, b] = phi
                            rst_delta[d, b] = delta
                            rst_entropy[d, b] = calc_entropy(quant_weights)
                            rst_rate_entropy[d, b] = (ed_id - st_id) * rst_entropy[d, b] * n_channel_elements
                            rst_rate[d, b] = (ed_id - st_id) * b * n_channel_elements
                            rst_dist[d, b] = cur_dist
                            min_dist = cur_dist

                        if (cur_mse < min_mse):
                            rst_delta_mse[d, b] = delta
                            rst_dist_mse[d, b] = cur_mse
                            min_mse = cur_mse

                        if (cur_dist > pre_dist) and (cur_mse > pre_mse):
                            break

                        if b == 0:
                            break

                        pre_mse = cur_mse
                        pre_dist = cur_dist

                layers[layerid].weight[:] = layer_weights[:]

            io.savemat(('%s/%s_%03d_%04d.mat' % (path_output, args.archname, layerid, f)),
               {'rd_phi': rst_phi.cpu().numpy(), 'rd_delta': rst_delta.cpu().numpy(), 
               'rd_entropy': rst_entropy.cpu().numpy(), 'rd_rate': rst_rate.cpu().numpy(), 
                'rd_dist': rst_dist.cpu().numpy(), 'rd_rate_entropy': rst_rate_entropy.cpu().numpy(), 
                # 'rst_approx_dist': rst_approx_dist.cpu().numpy(),
                'rd_delta_mse': rst_delta_mse.cpu().numpy(), 'rst_dist_mse': rst_dist_mse.cpu().numpy()})
    
        if args.sort_channels:
            io.savemat(f'{path_output}/{args.archname}_{layerid:03d}_channel_inds.mat', {'channel_inds': inds_channel_range.cpu().numpy()})
