import argparse
import pdb

import torch
import os
import common
import algo
import header
import math
import scipy.io as io
from common import *
from algo import *
from header import *
from time import time
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='generate rate-distortion curves')
parser.add_argument('--archname', default='mobilenetv2py', type=str,
                    help='name of network architecture: resnet18, resnet34, resnet50, densenet, etc')
parser.add_argument('--maxsteps', default=32, type=int,
                    help='number of Delta to enumerate')
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
parser.add_argument('--nchannelbatch', default=64, type=int,
                    help='number of filters for each quantization batch')                    
parser.add_argument('--batchsize', default=64, type=int,
                    help='batch size')
parser.add_argument('--nprocessings', default=4, type=int,
                    help='total number of threads to run')
parser.add_argument('--modeid', default=0, type=int,
                    help='all filters (with N_filter % nprocessings = modeid) will be processed')
parser.add_argument('--part_id', default=0, type=int, help="break total layers into parts and process each part in each process.")
parser.add_argument('--num_parts', default=5, type=int)
parser.add_argument('--bias_corr_act', '-bca', action="store_true")
parser.add_argument('--Amse', action="store_true", help="output MSE of current layer activation in repl. of Ysse")
parser.add_argument('--profile', action="store_true")

args = parser.parse_args()
args.val_testsize = args.testsize

if args.profile:
    from torch.profiler import profile, record_function, ProfilerActivity

maxsteps = 32
maxrates = args.maxrates

path_output = ('./%s_nr_%04d_ns_%04d_nf_%04d_%srdcurves_channelwise_opt_dist_act%s' % (args.archname, args.maxrates, \
                                                                   args.testsize, args.nchannelbatch, "bca_" if args.bias_corr_act else "", "_Amse" if args.Amse else ""))
isExists=os.path.exists(path_output)
if not isExists:
    os.makedirs(path_output)


net = loadnetwork(args.archname, args.gpuid)
# images, labels = loadvaldata(args.datapath, args.gpuid, testsize=args.testsize)
if "vit" in args.archname and "mae" not in args.archname:
    args.mean = [0.5,] * 3
    args.std = [0.5,] * 3
else:
    args.mean = IMAGENET_DEFAULT_MEAN
    args.std = IMAGENET_DEFAULT_STD
if "384" in args.archname:
    loader = get_val_imagenet_dali_loader(args, 384, 384)
else:
    loader = get_val_imagenet_dali_loader(args)
    
neural, layers = convert_qconv(net, stats=False)

net.eval()
neural.eval()
hookedlayers = hooklayers(layers)

# import pdb; pdb.set_trace()
print('total number of layers: %d' % (len(layers)))

# loader = torch.utils.data.DataLoader(images, batch_size=args.batchsize)
Y, labels = predict_dali_withgt(net, loader)
Y_cats = gettop1(Y)
top_1, top_5 = accuracy(Y, labels, topk=(1,5))
print('original network %s accuracy: top1 %5.2f top5 %5.2f' % (args.archname, top_1, top_5))
dimens = [hookedlayers[i].input if isinstance(layers[i].layer, nn.Conv2d) else hookedlayers[i].input.flip(0) for i in range(0,len(hookedlayers))]
if args.Amse:
    fp_acts = [h.input_tensor.clone() for h in hookedlayers]
    for h in hookedlayers:
        h.close()
else:
    for l in hookedlayers:
        l.close()

len_part = len(layers) // args.num_parts 

with torch.no_grad():
    for l in range(args.part_id * len_part, (args.part_id + 1) * len_part):
        layer_weights = layers[l].weight.clone()
        nchannels = layers[l].layer.in_channels if isinstance(layers[l].layer, nn.Conv2d) else layers[l].layer.in_features
        ngroups = math.ceil(nchannels / args.nchannelbatch)
        layers[l].coded, layers[l].delta = [0] * ngroups, [0] * ngroups

        acti_delta = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_coded = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_Y_sse = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_Y_top = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf

        for g in range(0, ngroups):
            layers[l].group_id = g

            chans_per_group = min(nchannels - g * args.nchannelbatch, args.nchannelbatch)
            coded = chans_per_group * (dimens[l][1:].prod() if len(dimens[l]) > 1 else 1)

            scale = 0
            start = scale 
            for b in range(0,maxrates):
                last_Y_sse = Inf
                for j in range(0,maxsteps):
                    sec = time()
                    delta = start + 0.25*j
                    layers[l].quantized, layers[l].coded[g], layers[l].delta[g], layers[l].chans_per_group = True, coded*b, delta, chans_per_group

                    for l_ in layers:
                        l_.count = 0
                    
                    if args.Amse:
                        hookedlayer_fpact = Hook(layers[l], fp_act=fp_acts[l])
                    if args.profile:
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, use_cuda=True,
                            on_trace_ready=lambda p: print(p.key_averages().table(sort_by="cpu_time_total", row_limit=15))) as prof:
                            
                            with record_function("forward"):
                                Y_hats = predict_dali(neural,loader)
                        exit()
                    else:
                        Y_hats = predict_dali(neural,loader)
                    top_1, top_5 = accuracy(Y_hats, labels, topk=(1, 5))
                    sec = time() - sec
                    # import pdb; pdb.set_trace()
                    acti_Y_sse[b,j,0] = ((Y_hats - Y)**2).mean() if not args.Amse else hookedlayer_fpact.accum_err_act
                    acti_Y_top[b,j,0] = top_1
                    acti_delta[b,j,0] = layers[l].delta[g]
                    acti_coded[b,j,0] = layers[l].coded[g]
                    # acti_mean[b,j] = hookedlayers[l].mean_err_a

                    mean_Y_sse = acti_Y_sse[b,j,0]
                    mean_Y_top = acti_Y_top[b,j,0]
                    #mean_W_sse = acti_W_sse[b,j,0]
                    
                    #print('%d, %d, %f' % (b, j, acti_Y_sse[b,j,0]))
                    if b >= 2 and mean_Y_sse > last_Y_sse or b == 0:
                        break

                    last_Y_sse = mean_Y_sse
                    if args.Amse:
                        hookedlayer_fpact.close()

                _, j = acti_Y_sse[b,:,0].min(0)
                delta = acti_delta[b,j,0]
                start = delta - 2
                mean_Y_sse = acti_Y_sse[b,j,0]
                mean_Y_top = acti_Y_top[b,j,0]
                print('%s | layer: %03d/%03d, channel: %d, delta: %+6.2f, '
                    'mse: %5.2e (%5.2e), top1: %5.2f, numel: %5.2e, rate: %4.1f, time: %5.2fs'\
                    % (args.archname, l, len(layers), g * args.nchannelbatch, delta, mean_Y_sse, mean_Y_sse, mean_Y_top,\
                        coded, b, sec))
            layers[l].quantized, layers[l].coded[g], layers[l].delta[g] = False, 0, 0

            io.savemat(('%s/%s_val_%03d_%04d_output_acti.mat' % (path_output, args.archname,l,g)),
                    {'acti_coded':acti_coded.cpu().numpy(),'acti_Y_sse':acti_Y_sse.cpu().numpy(),
                        'acti_Y_top':acti_Y_top.cpu().numpy(),'acti_delta':acti_delta.cpu().numpy()})
        layers[l].group_id = -1