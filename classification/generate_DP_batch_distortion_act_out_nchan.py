import common
import header
import argparse
import algo
import tunstall
import copy
from functools import partial
from huffman import *
from common import *
from header import *
from algo import *
from tunstall import *
import time
import scipy.io as io
import math, numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

parser = argparse.ArgumentParser(description='generate rate-distortion curves')
parser.add_argument('--archname', default='resnet34py', type=str,
                    help='name of network architecture: resnet18, resnet34, resnet50, densenet, etc')
parser.add_argument('--pathrdcurve', default='./rd_curves', \
                    type=str,
                    help='path of rate distortion curves')
parser.add_argument('--maxdeadzones', default=10, type=int,
                    help='number of sizes of dead zones')
parser.add_argument('--maxrates', default=11, type=int,
                    help='number of bit rates')
parser.add_argument('--minrate', default=0, type=int,
                    help='minimum bit rate to allocate')
parser.add_argument('--gpuid', default=2, type=int,
                    help='gpu id')
parser.add_argument('--datapath', default='./ImageNet2012/', type=str,
                    help='imagenet dateset path')
parser.add_argument('--testsize', default=-1, type=int,
                    help='number of images to evaluate')
parser.add_argument('--target_rates', nargs="+", type=float)
parser.add_argument('--gen_rate_curves', action="store_true")
parser.add_argument('--batchsize', default=128, type=int,
                    help='batch size')
parser.add_argument('--numworkers', default=16, type=int,
                    help='number of works to read images')
parser.add_argument('--tunstallbit', default=10, type=int,
                    help='length of Tunstall codes')
parser.add_argument('--nstage', default=5, type=int,
                    help='stage of tunstall coding')
parser.add_argument('--nchannelbatch', default=64, type=int,
                    help='number of channels for each quantization batch')
parser.add_argument('--closedeadzone', default=0, type=int,
                    help='swith to open or close dead zone')
parser.add_argument('--bitrangemin', default=0, type=int,
                    help='0 <= bitrangemin <= 10')
parser.add_argument('--bitrangemax', default=10, type=int,
                    help='0 <= bitrangemax <= 10')
parser.add_argument('--msqe', default=0, type=int,
                    help='use msqe to allocate bits')
parser.add_argument('--bit_rate', default=0, type=int,
                    help='use fixed-length code')
parser.add_argument('--relu_bitwidth', default=-1, type=int,
                    help='bit width of activations')
parser.add_argument('--quant-type', default='linear', type=str, choices=['linear', 'log2'])
parser.add_argument('--override-checkpoint', default='', type=str)
parser.add_argument('--bias_corr_weight', '-bcw', action="store_true")
parser.add_argument('--bias_corr_act', '-bca', action="store_true")
parser.add_argument('--bca_version', default=1, type=int)
parser.add_argument('--Amse', action="store_true")
parser.add_argument('--output_bit_allocation', action="store_true")
parser.add_argument('--bit_list', nargs="+", type=int)
parser.add_argument('--re-calibrate', action="store_true")
parser.add_argument('--re-calibrate-lr', default=0.0001, type=float)
parser.add_argument('--re-train', action="store_true")
parser.add_argument('--re-train-lr', default=0.0001, type=float)
parser.add_argument('--re-train-iter', default=20, type=int)
parser.add_argument('--re-train-epoch', default=1, type=int)
parser.add_argument('--smooth-dists', action="store_true")
parser.add_argument('--adaptive_act_delta', action="store_true")

parser.add_argument("--redo-dp", action="store_true")
parser.add_argument("--debug_layer_i", default=-1, type=int)
parser.add_argument("--act_dist_penalty_scale", default=1.0, type=float)
parser.add_argument("--act_curve_root_dir", default="./", type=str)
args = parser.parse_args()
args.val_testsize = args.testsize

if (args.re_train or args.re_calibrate) and args.re_train_epoch > 1:
    print(f"WARN: ignoring retrain iteration config, using retrain epoch={args.re_train_epoch}")
    args.re_train_iter = -1

tranname = "idt"
trantype = "exter"
# maxrates = 17
if args.target_rates is None:
    if args.gen_rate_curves:
        args.target_rates = np.arange(4, 8, 0.1)
        maxsteps = len(args.target_rates)
    else:
        args.target_rates = [4., 6., 8.]
        maxsteps = 3
else:
    maxsteps = len(args.target_rates)
codeacti = True

srcnet = loadnetwork(args.archname, args.gpuid)

if args.override_checkpoint != '':
    all_ = srcnet.load_state_dict(torch.load(args.override_checkpoint), strict=False)
    print(all_)
    print('load checkpoint from %s' % args.override_checkpoint)

tarnet = copy.deepcopy(srcnet)
tarnet = convert_qconv_new(tarnet, stats=False).cuda()
tarlayers = [layer for layer in tarnet.modules() if isinstance(layer, transconv.QAWrapper)]
srcnet = convert_qconv_new(srcnet, stats=False).cuda()
srclayers = [layer for layer in srcnet.modules() if isinstance(layer, transconv.QAWrapper)]

tardimens = hooklayers(tarlayers)
tarnet.eval()
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

Y, labels = predict_dali_withgt(tarnet, loader)
predict_fn = predict_dali

top_1, top_5 = accuracy(Y, labels, topk=(1,5))
print('original network %s accuracy: top1 %5.2f top5 %5.2f' % (args.archname, top_1, top_5))
Y_norm = Y / ((Y**2).sum().sqrt())

nlayers = len(tarlayers)

dimens = [tardimens[i].output if isinstance(tarlayers[i].layer, nn.Conv2d) else tardimens[i].output.flip(0) for i in range(0,len(tardimens))]
# print("\n".join((str(d) for d in dimens)))

rd_rate, rd_rate_entropy, rd_dist, rd_phi, rd_delta, rd_delta_mse, rd_dist_mse = \
    load_rd_curve_batch(args.archname, srclayers, args.maxdeadzones, args.maxrates, args.pathrdcurve, args.nchannelbatch, closedeadzone=args.closedeadzone)


if args.minrate > 0:
    for l in range(0,len(srclayers)):
        rd_dist[l][:, :, :args.minrate] = Inf
        rd_dist_mse[l][:, :, :args.minrate] = Inf


rd_act_dist = []
rd_act_bits = []
rd_act_delta = []
rd_act_coded = []
act_sizes = []
for l in range(0,len(srclayers)):
    ##load files here
    # layer_weights_idx = srclayers[l].weight.clone()
    nchannels = tarlayers[l].layer.out_channels if isinstance(tarlayers[l].layer, nn.Conv2d) else tarlayers[l].layer.out_features
    ngroups = math.ceil(nchannels / args.nchannelbatch)
    
    coded_g, delta_g = [0] * ngroups, [0] * ngroups
    rst_act_dist_groups = []
    rst_act_bit_groups = []
    rst_act_delta_groups = []
    act_sizes_group = []
    rst_act_coded_groups = []
    for g in range(ngroups):
        tarlayers[l].chans_per_group = min(nchannels - g * args.nchannelbatch, args.nchannelbatch)
        
        curve_test_size = int(args.pathrdcurve.split("ns_")[-1][:4])
        acti_Y_sse, acti_delta, acti_coded = loadrdcurves(args.archname, l, g, 'acti', args.nchannelbatch, args.Amse, curve_test_size, mode="out_channel", prefix=args.act_curve_root_dir)
        rst_act_bit_groups.append(np.arange(0, len(acti_Y_sse)))

        if args.minrate > 0:
            acti_Y_sse[:args.minrate] = Inf

        rst_act_dist_groups.append(np.min(acti_Y_sse[..., 0], axis=1))
        rst_act_delta_groups.append(np.array([acti_delta[b, np.argmin(acti_Y_sse[b]), 0] for b in range(len(acti_delta))]))
        act_sizes_group.append(acti_coded[1, 0, 0].astype(int))
        rst_act_coded_groups.append(np.array([acti_coded[b, np.argmin(acti_Y_sse[b]), 0] for b in range(len(acti_coded))]))
    rd_act_dist.append(rst_act_dist_groups)
    rd_act_bits.append(rst_act_bit_groups)
    rd_act_delta.append(rst_act_delta_groups)
    act_sizes.append(act_sizes_group)
    rd_act_coded.append(rst_act_coded_groups)




hist_sum_W_sse = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_Y_sse = torch.ones(maxsteps,device=getdevice()) * Inf
pred_sum_Y_sse = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_coded = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_coded_w = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_coded_a = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_coded_huffman = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_coded_tunstall = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_denom = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_denom_w = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_denom_a = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_Y_tp1 = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_Y_tp5 = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_non0s = torch.ones(maxsteps,len(srclayers),device=getdevice()) * Inf

if args.output_bit_allocation:  
    path_output = ('./%s_nr_%04d_nf_%04d_%s_bit_allocation/' % (args.archname, args.maxrates, args.nchannelbatch, "_Amse" if args.Amse else ""))
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # bits_allocated = {'w': np.zeros(len(srclayers)), 'a': np.zeros(len(srclayers))}
    # bits_allocated = {}
    # f = open(f'{path_output}/slope={slope}_bit_allocation.txt', "w")


solve_times = []
for j in range(len(args.target_rates)):
    target_rate = args.target_rates[j]

    G_dir = args.pathrdcurve.rstrip("/") + "_actoutchannel" + ("_Amse" if args.Amse else "") + "_DP_G_" + ("8.0" if args.gen_rate_curves else f"{target_rate:.1f}")
    if args.redo_dp:
        print(f"Removing {G_dir}")  
        os.system(f"rm -rf {G_dir}")

    pc_phi, pc_delta, pc_bits, pc_size, pc_act_dist, pc_act_bits, pc_act_delta, pc_act_coded = dp_quantize(srclayers, rd_rate, rd_dist, rd_phi, rd_delta, 
                                                                                                                    rd_act_dist, rd_act_bits, rd_act_delta, act_sizes, 
                                                                                                                    args.nchannelbatch, 
                                                                                                                    target_rate, device="cuda", piece_length=2**20, 
                                                                                                                    G_dir=G_dir, act_dist_penalty_scale=args.act_dist_penalty_scale,
                                                                                                                    smooth_dists=args.smooth_dists, manual_offset=0)
    hist_sum_W_sse[j] = hist_sum_Y_sse[j] = pred_sum_Y_sse[j] = hist_sum_coded_huffman[j] = hist_sum_coded_tunstall[j] = 0.0
    hist_sum_coded[j] = hist_sum_coded_w[j] = hist_sum_coded_a[j] = hist_sum_Y_tp1[j] = hist_sum_Y_tp5[j] = hist_sum_denom[j] = hist_sum_denom_w[j] = hist_sum_denom_a[j] = 0.0
    
    print("Allocated bits:")
    print(f"{pc_bits=}")
    print(f"{pc_act_bits=}")

    with torch.no_grad():
        if args.output_bit_allocation:
            to_write = "" 

        sec = time.time()

        solve_times.append(time.time() - sec)
        layer_weights_ = [0] * len(srclayers)
        total_rates_bits = cal_total_rates(srclayers, pc_size, args.nchannelbatch)
        # print("total_rates_bits", total_rates_bits)
        huffman_codec = huffman_coding()
        
        for l in range(0,len(srclayers)):
            ##load files here
            layer_weights = srclayers[l].weight.clone()
            # layer_weights_idx = srclayers[l].weight.clone()
            nchannels = tarlayers[l].layer.out_channels if isinstance(tarlayers[l].layer, nn.Conv2d) else tarlayers[l].layer.out_features
            ngroups = math.ceil(nchannels / args.nchannelbatch)
            tarlayers[l].mode = "out_channel"
            
            if codeacti:
                tarlayers[l].quantized = 0 < l < len(srclayers) - 1
                coded_g, delta_g = [0] * ngroups, [0] * ngroups
                for g in range(ngroups):
                    tarlayers[l].chans_per_group = min(nchannels - g * args.nchannelbatch, args.nchannelbatch)
                    tarlayers[l].adaptive_delta = args.adaptive_act_delta
                    
                    begin = time.time()
                    solve_times[-1] += time.time() - begin
                    coded_g[g] = pc_act_coded[l][g]
                    delta_g[g] = pc_act_delta[l][g]
                    pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + pc_act_dist[l][g]
                    hist_sum_coded[j] = hist_sum_coded[j] + pc_act_coded[l][g] #  - dimens[l][1:].prod() * tarlayers[l].chans_per_group 
                    hist_sum_coded_a[j] = hist_sum_coded_a[j] + pc_act_coded[l][g]
                    hist_sum_denom[j] = hist_sum_denom[j] + dimens[l][1:].prod() * tarlayers[l].chans_per_group
                    hist_sum_denom_a[j] = hist_sum_denom_a[j] + dimens[l][1:].prod() * tarlayers[l].chans_per_group
                    
                if args.output_bit_allocation:
                    # bits_allocated['a'][l] = sum([coded / (dimens[l][1:].prod() * tarlayers[l].chans_per_group) for coded in coded_g]) / len(coded_g)
                    # bits_allocated[f'a_{l}'] = np.array([coded / (dimens[l][1:].prod() * tarlayers[l].chans_per_group) for coded in coded_g])
                    to_write += f"a_{l}," + ",".join(str((coded / (dimens[l][1:].prod() * tarlayers[l].chans_per_group)).int().item()) for coded in coded_g) + f", {int(sum(coded_g))}" "\n"
                tarlayers[l].coded, tarlayers[l].delta = coded_g, delta_g
                if args.bias_corr_act:
                    # tarlayers[l].err_mean = acti_mean
                    tarlayers[l].bca = bcorr_act_factory(args.bca_version, scale=2**torch.tensor(delta_g).cuda())

            hist_sum_non0s[j,l] = (layer_weights != 0).any(1).sum()
            hist_sum_denom[j] = hist_sum_denom[j] + layer_weights.numel()
            hist_sum_denom_w[j] = hist_sum_denom_w[j] + layer_weights.numel()

            nchannels = get_num_output_channels(layer_weights)
            n_channel_elements = get_ele_per_output_channel(layer_weights)
            quant_weights = tarlayers[l].weight.clone()
            if "sort" in args.pathrdcurve:
                channel_inds = io.loadmat(f'{args.pathrdcurve}/{args.archname}_{l:03d}_channel_inds.mat')['channel_inds'][0]
                # channel_inds = torch.argsort(layer_weights.view(nchannels, -1).max(1)[0])


            for cnt, f in enumerate(range(0, nchannels, args.nchannelbatch)):
                st_layer = f
                ed_layer = f + args.nchannelbatch
                if f + args.nchannelbatch > nchannels:
                    ed_layer = nchannels
                
                if "sort" in args.pathrdcurve:
                    inds = channel_inds[st_layer: ed_layer]
                else:
                    inds = list(range(st_layer, ed_layer))
                # import pdb; pdb.set_trace()
                output_channels = get_output_channels_inds(layer_weights, inds).clone()
                if args.quant_type == 'linear':
                    quant_output_channels = deadzone_quantize(output_channels, pc_phi[l][cnt], 2**(pc_delta[l][cnt]), pc_bits[l][cnt])
                    quant_index_output_channels = deadzone_quantize_idx(output_channels, pc_phi[l][cnt], 2**(pc_delta[l][cnt]), pc_bits[l][cnt])
                elif args.quant_type == 'log2':
                    quant_output_channels = log_quantize(output_channels, pc_phi[l][cnt], 2**(pc_delta[l][cnt]), pc_bits[l][cnt])
                    quant_index_output_channels = log_quantize(output_channels, pc_phi[l][cnt], 2**(pc_delta[l][cnt]), pc_bits[l][cnt])
                if args.bias_corr_weight:
                    quant_output_channels = bcorr_weight(output_channels, quant_output_channels)
                
                assign_output_channels_inds(tarlayers[l].weight, inds, quant_output_channels)
                assign_output_channels_inds(quant_weights, inds, quant_index_output_channels)
                hist_sum_coded_w[j] = hist_sum_coded_w[j] + pc_bits[l][cnt] * output_channels.numel()
            
            if args.output_bit_allocation:
                # bits_allocated['w'][l] = sum(pc_bits[l]) / len(pc_bits[l])
                # bits_allocated[f'w_{l}'] = pc_bits[l]
                to_write += f"w_{l}," + str(pc_bits[l]).strip("[]") + f", {int(layer_weights.numel() * sum(pc_bits[l]) / len(pc_bits[l]))}" + "\n"


            hist_sum_W_sse[j] = hist_sum_W_sse[j] + ((srclayers[l].weight - tarlayers[l].weight)**2).sum()
        print(f'W rate: {float(hist_sum_coded_w[j]) / hist_sum_denom_w[j]}, A rate: {float(hist_sum_coded_a[j]) / hist_sum_denom_a[j]}')   
        hist_sum_coded_huffman[j] += hist_sum_coded[j]
        hist_sum_coded_tunstall[j] += hist_sum_coded[j]
        rates_tunstall = min(hist_sum_coded_huffman[j], total_rates_bits)
        rates_huffman = min(hist_sum_coded_huffman[j], total_rates_bits)
        hist_sum_coded[j] += total_rates_bits



    
    if args.re_calibrate:
        print("Re-calibrating...")
        args.val_testsize = 1024
        if "384" in args.archname:
            calib_loader = get_calib_imagenet_dali_loader(args, 384, 384)
        else:
            calib_loader = get_calib_imagenet_dali_loader(args)
        
        args.val_testsize = -1
        tarnet = retrain_bias(tarnet, calib_loader, lr=args.re_calibrate_lr, iters=args.re_train_iter, epochs=args.re_train_epoch)

        with torch.no_grad():        
            Y_hats = predict_fn(tarnet, loader)
            print("Re-calibrated accuracy: ", accuracy(Y_hats, labels, topk=(1, 5)))

    if args.re_train:
        print("Re-training...")
        if "384" in args.archname:
            train_loader, _ = get_trainval_imagenet_dali_loader(args, 384, 384)
        else:
            train_loader, _ = get_trainval_imagenet_dali_loader(args)

        net2 = loadnetwork(args.archname, args.gpuid)
        net2, net2_layers = convert_qconv_diffable(net2, stats=False)
        net2.load_state_dict(tarnet.state_dict())

        for l, (layer, phi, delta, bits) in enumerate(zip(net2_layers, pc_phi, pc_delta, pc_bits)):
            layer.phi = torch.from_numpy(phi).to(getdevice())
            layer.delta = torch.from_numpy(delta).to(getdevice())
            layer.bits = torch.from_numpy(bits).to(getdevice())
            layer.deltas_act = torch.from_numpy(pc_act_delta[l]).to(getdevice())
            layer.coded = pc_act_coded[l]
            layer.chans_per_group = args.nchannelbatch
            layer.quantized =  0 < l < len(net2_layers) - 1
            layer.mode = "out_channel"
           
            # layer.adaptive_delta = True

        net2 = retrain(net2, train_loader, lr=args.re_train_lr, iters=args.re_train_iter, epochs=args.re_train_epoch)
        
        for layer in net2_layers:
            layer.adaptive_delta = args.adaptive_act_delta
            layer.bcw = bcorr_weight if args.bias_corr_weight else None
            assert args.bca_version == 1
            layer.bca = bcorr_act_factory(args.bca_version) # Not support version 2

        with torch.no_grad():        
            Y_hats = predict_fn(net2, loader)
            print("Re-trained accuracy: ", accuracy(Y_hats, labels, topk=(1, 5)))
        
        del net2, net2_layers

    with torch.no_grad():        
        Y_hats = predict_fn(tarnet, loader)
        hist_sum_Y_tp1[j], hist_sum_Y_tp5[j] = accuracy(Y_hats, labels, topk=(1, 5))
        hist_sum_W_sse[j] = hist_sum_W_sse[j]/hist_sum_denom[j]
        hist_sum_Y_sse[j] = ((Y_hats - Y)**2).mean()
        hist_sum_coded[j] = hist_sum_coded[j]/hist_sum_denom[j]
        hist_sum_coded_huffman[j] = rates_huffman / hist_sum_denom[j]
        hist_sum_coded_tunstall[j] = rates_tunstall / hist_sum_denom[j]
        sec = time.time() - sec


    print('%s %s | target rate: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, topk: %5.2f (%5.2f), rate: %5.2f, tuns rate: %5.2f, huff rate: %5.2f' %\
            (args.archname, tranname, target_rate, hist_sum_Y_sse[j], pred_sum_Y_sse[j], \
            hist_sum_W_sse[j], hist_sum_Y_tp1[j], hist_sum_Y_tp5[j],  hist_sum_coded[j], hist_sum_coded_tunstall[j], hist_sum_coded_huffman[j]))
    print(f'Avg Optimization Time: {sum(solve_times) / len(solve_times):.3f} s')
    
    torch.save(tarnet.state_dict(), f'{args.archname}_act_out_nchan_{args.nchannelbatch}_target_{target_rate:.1f}.pth')

    with open(f'{args.archname}_acc_dist_curve_dp.txt', "a+") as f:
        f.write(f"{hist_sum_coded[j]}, {hist_sum_Y_tp1[j]}, {hist_sum_Y_sse[j]}\n")
    
    # if hist_sum_coded[j] == 0.0 or \
    #    hist_sum_Y_tp1[j] <= 0.002:
    #     break
    if args.output_bit_allocation:
        # io.savemat(f'{path_output}/slope={slope}_bit_allocation.mat', bits_allocated)
        with open(f'{path_output}/target={target_rate}_rate={hist_sum_coded[j]:5.2f}_bit_allocation.txt', "w") as f:
            f.write(to_write)

io.savemat(('%s_%s_sum_%d_output_%s%s.mat' % (args.archname, tranname, args.testsize, trantype, "" if not args.bias_corr_weight else "_bcw")),\
           {'hist_sum_Y_sse':hist_sum_Y_sse.cpu().numpy(),'hist_sum_Y_tp1':hist_sum_Y_tp1.cpu().numpy(),\
            'pred_sum_Y_sse':pred_sum_Y_sse.cpu().numpy(),'hist_sum_coded':hist_sum_coded.cpu().numpy(),\
            'hist_sum_W_sse':hist_sum_W_sse.cpu().numpy(),'hist_sum_denom':hist_sum_denom.cpu().numpy(),\
            'hist_sum_coded_huffman':hist_sum_coded_huffman.cpu().numpy(),'hist_sum_coded_tunstall':hist_sum_coded_tunstall.cpu().numpy(),\
            'hist_sum_non0s':hist_sum_non0s.cpu().numpy(),'hist_sum_Y_tp5':hist_sum_Y_tp5.cpu().numpy()})
