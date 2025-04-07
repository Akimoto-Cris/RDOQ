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
parser.add_argument('--gpuid', default=0, type=int,
                    help='gpu id')
parser.add_argument('--datapath', default='./ImageNet2012/', type=str,
                    help='imagenet dateset path')
parser.add_argument('--testsize', default=-1, type=int,
                    help='number of images to evaluate')
parser.add_argument('--maxslopesteps', default=192, type=int,
                    help='number of slopes to enumerate')
parser.add_argument('--slopes', nargs="+", type=float)
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
parser.add_argument('--bias_corr_weight', '-bcw', action="store_true")
parser.add_argument('--bias_corr_act', '-bca', action="store_true")
parser.add_argument('--bca_version', default=1, type=int)
parser.add_argument('--Amse', action="store_true")
parser.add_argument('--output_bit_allocation', action="store_true")
parser.add_argument("--act_curve_root_dir", default="./", type=str)
parser.add_argument('--re-train-iter', default=20, type=int)
parser.add_argument('--re-train-epoch', default=1, type=int)

parser.add_argument('--bit_list', nargs="+", type=int)
parser.add_argument('--output_bops', action="store_true")
parser.add_argument('--re-calibrate', action="store_true")

parser.add_argument('--re-train', action="store_true")
parser.add_argument('--re-train-lr', default=0.0001, type=float)


args = parser.parse_args()
args.val_testsize = args.testsize

if (args.re_train or args.re_calibrate) and args.re_train_epoch > 1:
    print(f"WARN: ignoring retrain iteration config, using retrain epoch={args.re_train_epoch}")
    args.re_train_iter = -1


tranname = "idt"
trantype = "exter"
maxsteps = args.maxslopesteps
# maxrates = 17
codeacti = True

srcnet = loadnetwork(args.archname, args.gpuid)

tarnet = copy.deepcopy(srcnet)
images, labels = loadvaldata(args.datapath, args.gpuid, testsize=args.testsize)
tarnet, tarlayers = convert_qconv(tarnet, stats=False)
srcnet, srclayers = convert_qconv(srcnet, stats=False)
tardimens = hooklayers(tarlayers)
# loader = torch.utils.data.DataLoader(images, batch_size=args.batchsize, num_workers=args.numworkers)
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
# Y = predict2(net, loader)
top_1, top_5 = accuracy(Y, labels, topk=(1,5))
print('original network %s accuracy: top1 %5.2f top5 %5.2f' % (args.archname, top_1, top_5))
Y_norm = Y / ((Y**2).sum().sqrt())

nlayers = len(tarlayers)

# loader = get_val_imagenet_dali_loader(args)

nweights = cal_total_num_weights(tarlayers)
print('total num layers %d weights on %s is %d' % (nlayers, args.archname, nweights))
dimens = [tardimens[i].output if isinstance(tarlayers[i].layer, nn.Conv2d) else tardimens[i].output.flip(0) for i in range(0,len(tardimens))]
# print("\n".join((str(d) for d in dimens)))

rd_rate, rd_rate_entropy, rd_dist, rd_phi, rd_delta, rd_delta_mse, rd_dist_mse = \
        load_rd_curve_batch(args.archname, srclayers, args.maxdeadzones, args.maxrates, args.pathrdcurve, args.nchannelbatch, \
                        closedeadzone=args.closedeadzone)
if args.bit_list is not None:
    bit_list = list(map(lambda x:x-1, args.bit_list))
    # import pdb; pdb.set_trace()
    rd_rate_entropy = [d[:, bit_list] for d in rd_rate_entropy]
    rd_dist = [d[:, bit_list] for d in rd_dist]
    rd_phi = [d[:, bit_list] for d in rd_phi]
    rd_delta = [d[:, bit_list] for d in rd_delta]

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


if args.slopes is not None:
    maxsteps = len(args.slopes)

solve_times = []
for j in range(0,maxsteps):
    hist_sum_W_sse[j] = hist_sum_Y_sse[j] = pred_sum_Y_sse[j] = hist_sum_coded_huffman[j] = hist_sum_coded_tunstall[j] = 0.0
    hist_sum_coded[j] = hist_sum_coded_w[j] = hist_sum_coded_a[j] = hist_sum_Y_tp1[j] = hist_sum_Y_tp5[j] = hist_sum_denom[j] = hist_sum_denom_w[j] = hist_sum_denom_a[j] = 0.0

    with torch.no_grad():
        slope = -35 + 0.5*j
        if args.slopes is not None:
            slope = args.slopes[j]

        if args.output_bit_allocation:
            to_write = "" 
        
        sec = time.time()

        if args.archname == 'mobilenetv3py':
            pc_phi, pc_delta, pc_bits, pc_rate, pc_size = \
                pareto_condition_batch_less_items(tarlayers, rd_rate, rd_dist, rd_phi, rd_delta, 2 ** slope,
                                        args.nchannelbatch)
        elif args.bit_rate == 1:
            if args.msqe == 1:
                pc_phi, pc_delta, pc_bits, pc_rate, pc_size = \
                    pareto_condition_batch(srclayers, rd_rate, rd_dist_mse, rd_phi, rd_delta_mse, 2**slope, args.nchannelbatch)
            else:
                pc_phi, pc_delta, pc_bits, pc_rate, pc_size = \
                    pareto_condition_batch(srclayers, rd_rate, rd_dist, rd_phi, rd_delta, 2 ** slope, args.nchannelbatch)
        else:
            pc_phi, pc_delta, pc_bits, pc_rate, pc_size = \
                pareto_condition_batch(srclayers, rd_rate_entropy, rd_dist, rd_phi, rd_delta, 2 ** slope, args.nchannelbatch)
        
        tis = time.time() - sec
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
            
            if codeacti:
                tarlayers[l].quantized = l > 0 # True
                coded_g, delta_g = [0] * ngroups, [0] * ngroups
                for g in range(ngroups):
                    tarlayers[l].chans_per_group = min(nchannels - g * args.nchannelbatch, args.nchannelbatch)
                    curve_test_size = int(args.pathrdcurve.split("ns_")[-1][:4])

                    acti_Y_sse, acti_delta, acti_coded = loadrdcurves(args.archname,l, g, 'acti', args.nchannelbatch, args.Amse, curve_test_size, mode="out_channel", prefix=args.act_curve_root_dir)
                    if args.bit_list is not None:
                        acti_Y_sse = acti_Y_sse[args.bit_list]
                        acti_delta = acti_delta[args.bit_list]
                        acti_coded = acti_coded[args.bit_list]
                    begin = time.time()
                    acti_Y_sse, acti_delta, acti_coded = findrdpoints(acti_Y_sse,acti_delta,acti_coded, 2**slope)
                    tis += time.time() - begin
                    

                    coded_g[g] = acti_coded[0]
                    delta_g[g] = acti_delta[0]
                    pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + acti_Y_sse[0]
                    hist_sum_coded[j] = hist_sum_coded[j] + acti_coded[0] #  - dimens[l][1:].prod() * tarlayers[l].chans_per_group 
                    hist_sum_coded_a[j] = hist_sum_coded_a[j] + acti_coded[0]
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
                output_channels = get_output_channels_inds(layer_weights, inds)
                quant_output_channels = deadzone_quantize(output_channels, pc_phi[l][cnt], 2**(pc_delta[l][cnt]), pc_bits[l][cnt])
                quant_index_output_channels = deadzone_quantize_idx(output_channels, pc_phi[l][cnt], 2**(pc_delta[l][cnt]), pc_bits[l][cnt])
                if args.bias_corr_weight:
                    quant_output_channels = bcorr_weight(output_channels, quant_output_channels)

                assign_output_channels_inds(tarlayers[l].weight, inds, quant_output_channels)
                assign_output_channels_inds(quant_weights, inds, quant_index_output_channels)
                hist_sum_coded_w[j] = hist_sum_coded_w[j] + pc_bits[l][cnt] * output_channels.numel()
            if args.output_bit_allocation:
                # bits_allocated['w'][l] = sum(pc_bits[l]) / len(pc_bits[l])
                # bits_allocated[f'w_{l}'] = pc_bits[l]
                to_write += f"w_{l}," + str(pc_bits[l]).strip("[]") + f", {int(layer_weights.numel() * sum(pc_bits[l]) / len(pc_bits[l]))}" + "\n"

            # if ((total_rates_bits / nweights) >= args.bitrangemin) and ((total_rates_bits / nweights) <= args.bitrangemax):
            #     hist_sum_coded_tunstall[j] += cal_multistage_tunstall_codes_size_fast(quant_weights, args.tunstallbit, args.nstage, evaluate=False) * nchannels * n_channel_elements
            #     hist_sum_coded_huffman[j] += huffman_codec.cal_huffman_code_length(quant_weights) * nchannels * n_channel_elements
                # print(cal_multistage_tunstall_codes_size_fast(quant_weights, args.tunstallbit, args.nstage, evaluate=False) * nchannels * n_channel_elements)
                # print(huffman_codec.cal_huffman_code_length(quant_weights) * nchannels * n_channel_elements)

            hist_sum_W_sse[j] = hist_sum_W_sse[j] + ((srclayers[l].weight - tarlayers[l].weight)**2).sum()
        
        solve_times.append(tis)
        print(f'Optimization Time: {tis:.3f} s')
        print(f'W rate: {float(hist_sum_coded_w[j]) / hist_sum_denom_w[j]}, A rate: {float(hist_sum_coded_a[j]) / hist_sum_denom_a[j]}')   

        
        hist_sum_coded_huffman[j] += hist_sum_coded[j]
        hist_sum_coded_tunstall[j] += hist_sum_coded[j]
        rates_tunstall = min(hist_sum_coded_huffman[j], total_rates_bits)
        rates_huffman = min(hist_sum_coded_huffman[j], total_rates_bits)
        hist_sum_coded[j] += total_rates_bits

        Y_hats = predict_fn(tarnet, loader) 
        hist_sum_Y_tp1[j], hist_sum_Y_tp5[j] = accuracy(Y_hats, labels, topk=(1, 5))
        hist_sum_W_sse[j] = hist_sum_W_sse[j]/hist_sum_denom[j]
        hist_sum_Y_sse[j] = ((Y_hats - Y)**2).mean()
        hist_sum_coded[j] = hist_sum_coded[j]/hist_sum_denom[j]
        hist_sum_coded_huffman[j] = rates_huffman / hist_sum_denom[j]
        hist_sum_coded_tunstall[j] = rates_tunstall / hist_sum_denom[j]
        sec = time.time() - sec

    if args.re_calibrate:
        print("Re-calibrating...")
        if "384" in args.archname:
            train_loader, _ = get_trainval_imagenet_dali_loader(args, 384, 384)
        else:
            train_loader, _ = get_trainval_imagenet_dali_loader(args)

        tarnet = retrain_bias(tarnet, train_loader, lr=1e-3, iters=1000)

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

        for layer, phi, delta, bits in zip(net2_layers, pc_phi, pc_delta, pc_bits):
            layer.phi = torch.tensor(phi).to(getdevice())
            layer.delta = torch.tensor(delta).to(getdevice())
            layer.bits = torch.tensor(bits).to(getdevice())
            layer.chans_per_group = args.nchannelbatch
            layer.quantized = True

        net2 = retrain(net2, train_loader, lr=args.re_train_lr, iters=args.re_train_iter, epochs=args.re_train_epoch)

        with torch.no_grad():        
            Y_hats = predict_fn(net2, loader)
            print("Re-trained accuracy: ", accuracy(Y_hats, labels, topk=(1, 5)))
        
        del net2, net2_layers

    print('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, topk: %5.2f (%5.2f), rate: %5.2f, tuns rate: %5.2f, huff rate: %5.2f' %\
            (args.archname, tranname, slope, hist_sum_Y_sse[j], pred_sum_Y_sse[j], \
            hist_sum_W_sse[j], hist_sum_Y_tp1[j], hist_sum_Y_tp5[j],  hist_sum_coded[j], hist_sum_coded_tunstall[j], hist_sum_coded_huffman[j]))
    print(f'Avg Optimization Time: {sum(solve_times) / len(solve_times):.3f} s')
    
    with open(f'{args.archname}_acc_dist_curve.txt', "a+") as f:
        f.write(f"{hist_sum_coded[j]}, {hist_sum_Y_tp1[j]}, {hist_sum_Y_sse[j]}\n")
    # if hist_sum_coded[j] == 0.0 or \
    #    hist_sum_Y_tp1[j] <= 0.002:
    #     break

            
    if args.output_bops:
        bops, base_bops = get_bops_model(tarnet, tarlayers, dimens, pc_bits, return_base=True)
        print(f'Bops: {bops / base_bops:.4f}')
        with open(f'{args.archname}_acc_dist_bops_curve.txt', "a+") as f:
            f.write(f'{float(hist_sum_coded_w[j]) / hist_sum_denom_w[j]}, {float(hist_sum_coded_a[j]) / hist_sum_denom_a[j]}, {hist_sum_Y_tp1[j]}, {bops:.4f}, {base_bops:.4f}\n')

    if args.output_bit_allocation:
        # io.savemat(f'{path_output}/slope={slope}_bit_allocation.mat', bits_allocated)
        with open(f'{path_output}/slope={slope}_rate={hist_sum_coded[j]:5.2f}_bit_allocation.txt', "w") as f:
            f.write(to_write)

io.savemat(('%s_%s_sum_%d_output_%s%s.mat' % (args.archname, tranname, args.testsize, trantype, "" if not args.bias_corr_weight else "_bcw")),\
           {'hist_sum_Y_sse':hist_sum_Y_sse.cpu().numpy(),'hist_sum_Y_tp1':hist_sum_Y_tp1.cpu().numpy(),\
            'pred_sum_Y_sse':pred_sum_Y_sse.cpu().numpy(),'hist_sum_coded':hist_sum_coded.cpu().numpy(),\
            'hist_sum_W_sse':hist_sum_W_sse.cpu().numpy(),'hist_sum_denom':hist_sum_denom.cpu().numpy(),\
            'hist_sum_coded_huffman':hist_sum_coded_huffman.cpu().numpy(),'hist_sum_coded_tunstall':hist_sum_coded_tunstall.cpu().numpy(),\
            'hist_sum_non0s':hist_sum_non0s.cpu().numpy(),'hist_sum_Y_tp5':hist_sum_Y_tp5.cpu().numpy()})


        # print('%s: slope %6.4f entropy rate %6.4f bits rate %6.4f tunstall rate %6.4f huffman rate %6.4f distortion %6.8f - top-1 %.2f top-5 %.2f | time: %f' % \
        #       (args.archname, slope, total_rates / nweights, total_rates_bits / nweights, rates_tunstall, \
        #        rates_huffman, \
        #        output_distortion, top_1, top_5, time() - end))
