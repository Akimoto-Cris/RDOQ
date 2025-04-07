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
import einops
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
parser.add_argument('--override-checkpoint', default='', type=str)
parser.add_argument('--part_id', default=0, type=int, help="break total layers into parts and process each part in each process.")
parser.add_argument('--num_parts', default=5, type=int)
parser.add_argument('--bias_corr_act', '-bca', action="store_true")
parser.add_argument('--Amse', action="store_true", help="output MSE of current layer activation in repl. of Ysse")
parser.add_argument('--kappa', default=1e-7, type=float)
parser.add_argument('--mute-print', action="store_true")
parser.add_argument('--profile', action="store_true")

args = parser.parse_args()
args.val_testsize = args.testsize


if args.profile:
    from torch.profiler import profile, record_function, ProfilerActivity

maxsteps = 32
maxrates = args.maxrates

net = loadnetwork(args.archname, args.gpuid)

path_output = ('./hessian_curves/%s_nr_%04d_ns_%04d_nf_%04d_%srdcurves_out_channelwise_opt_dist_act%s' % (args.archname, args.maxrates, \
                                                                   args.testsize, args.nchannelbatch, "bca_" if args.bias_corr_act else "", "_Amse" if args.Amse else ""))


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

net.eval()
layers = [m for m in net.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]


if not args.mute_print:
    print('total number of layers: %d' % (len(layers)))

# loader = torch.utils.data.DataLoader(images, batch_size=args.batchsize)
Y, labels = predict_dali_withgt(net, loader)
Y_cats = gettop1(Y)
top_1, top_5 = accuracy(Y, labels, topk=(1,5))
if not args.mute_print:
    print('original network %s accuracy: top1 %5.2f top5 %5.2f' % (args.archname, top_1, top_5))



def transform(x):
    if len(x.shape) == 4:
        return einops.rearrange(x, 'b c h w -> b (h w) c')
    elif len(x.shape) == 3:
        return x #einops.rearrange(x, 'b t c -> (b t) c')
    elif len(x.shape) == 2:
        return x[:, None, :]
    # return x.flatten(1)


import math
len_part = math.ceil(len(layers) / args.num_parts)

for l in range(args.part_id * len_part, min((args.part_id + 1) * len_part, len(layers))):
    nchannels = layers[l].out_channels if isinstance(layers[l], nn.Conv2d) else layers[l].out_features
    ngroups = math.ceil(nchannels / args.nchannelbatch)

    acti_delta = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
    acti_coded = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
    acti_Y_sse = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf

    hook = hooklayers([layers[l]])[0]
    backward_hook = hooklayers([layers[l]], backward=True)[0]

    grad_list = []
    fp_acts_list = []
    net.train()
    for c, data in enumerate(loader):
        try:
            x = data[0]["data"]
        except:
            x, _ = data
            x = x.cuda()
        res = torch.norm(net(x), p=2)
        # res = net(x).sum()
        res.backward()

        grad_list.append(backward_hook.output_tensor[0][None, ...])
        fp_acts_list.append(hook.output_tensor)

        for p in net.parameters():
            p.grad = None
        

    net.eval()
    dimens = {l: hook.output if isinstance(layers[l], nn.Conv2d) else hook.output.flip(0)}
    
    hook.close()
    backward_hook.close()

    with torch.no_grad():
        for g in range(0, ngroups):
            chans_per_group = min(nchannels - g * args.nchannelbatch, args.nchannelbatch)
            coded = chans_per_group * (dimens[l][1:].prod() if len(dimens[l]) > 1 else 1)

            grad = transform(torch.cat(grad_list, dim=0))[..., g * args.nchannelbatch: (g + 1) * args.nchannelbatch]
            fpact_group = transform(torch.cat(fp_acts_list, dim=0))[..., g * args.nchannelbatch: (g + 1) * args.nchannelbatch]

            # hessians_blockwise = [args.kappa * torch.eye(chans_per_group, device=getdevice()) + (1. / grad.shape[0]) * grad[:, t].T @ grad[:, t] for t in range(grad.shape[1])]

            scale = 0
            start = scale 
            for b in range(0,maxrates):
                last_Y_sse = Inf
                for j in range(0,maxsteps):
                    sec = time()
                    delta = start + 0.25*j

                    quantized_group = common.quantize(fpact_group.clone(), 2 ** delta, b)
                    delta_act = quantized_group - fpact_group

                    first_order_dist = 0 #(delta_act * grad).sum() / grad.shape[0]
                    # second_order_dist = sum(0.5 * delta_act[s, t][None, :] @ hessians_blockwise[t] @ delta_act[s, t][:, None] for t in range(len(hessians_blockwise)) for s in range(len(loader))) / len(loader)
                    second_order_dist = 0.5 * args.kappa * delta_act.pow(2).sum() / len(grad) 
                    # second_order_dist += 0.5 * sum(torch.diag(grad[:, t] @ delta_act[:, t].T).pow(2).sum() / grad.shape[0] for t in range(grad.shape[1])) / len(loader)
                    second_order_dist += 0.5 * torch.diag(grad.flatten(1) @ delta_act.flatten(1).T ).pow(2).sum() / len(grad)

                    cur_dist = first_order_dist + second_order_dist

                    sec = time() - sec
                    acti_Y_sse[b,j,0] = cur_dist
                    acti_delta[b,j,0] = delta
                    acti_coded[b,j,0] = coded*b
                    # acti_mean[b,j] = hookedlayers[l].mean_err_a

                    mean_Y_sse = acti_Y_sse[b,j,0]
                    #mean_W_sse = acti_W_sse[b,j,0]
                    
                    if b >= 2 and mean_Y_sse > last_Y_sse or b == 0:
                        break

                    last_Y_sse = mean_Y_sse

                _, j = acti_Y_sse[b,:,0].min(0)
                delta = acti_delta[b,j,0]
                start = delta - 2
                mean_Y_sse = acti_Y_sse[b,j,0]
                if not args.mute_print:
                    print('%s | layer: %03d/%03d, channel: %d, delta: %+6.2f, '
                        'mse: %5.2e, numel: %5.2e, rate: %4.1f, time: %5.2fs'\
                        % (args.archname, l, len(layers), g * args.nchannelbatch, delta, mean_Y_sse, \
                            coded, b, sec))

            io.savemat(('%s/%s_val_%03d_%04d_output_acti.mat' % (path_output, args.archname,l,g)),
                    {'acti_coded':acti_coded.cpu().numpy(),'acti_Y_sse':acti_Y_sse.cpu().numpy(),
                        'acti_delta':acti_delta.cpu().numpy()})
