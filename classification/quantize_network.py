import common
import quantize_conv_layer

from common import *
from quantize_conv_layer import *
from algo import *

def quantizeNetwork(net, archname, maxdeadzones, maxrates, slope_lambda, rdcurvepath, is_multi_gpus=False, \
                    nfilterbatch=1, dimens=None, bcw=False):
    layers = findconv(net, False)

    if isinstance(net, mobilenetv2py.MobileNetV2):
        flags_convrelu = []
        findConvRELU(net, flags_convrelu)
    nweights = cal_total_num_weights(layers)

    with torch.no_grad():
        rd_rate, rd_rate_entropy, rd_dist, rd_phi, rd_delta, rd_delta_mse, rd_dist_mse = \
            load_rd_curve_batch(archname, layers, maxdeadzones, maxrates, rdcurvepath, nfilterbatch)
        pc_phi, pc_delta, pc_bits, pc_rate, pc_size = \
            pareto_condition_batch(layers, rd_rate, rd_dist, rd_phi, rd_delta, 2 ** slope_lambda, nfilterbatch)
        total_rates = cal_total_rates(layers, pc_rate, nfilterbatch)
        for l in range(0, len(layers)):
            nchannels = layers[l].in_channels if isinstance(layers[l], nn.Conv2d) else layers[l].in_features
            ngroups = math.ceil(nchannels / nfilterbatch)
            coded_g, delta_g = [0] * ngroups, [0] * ngroups
            for g in range(ngroups):
                # layers[l].chans_per_group = min(nchannels - g * nfilterbatch, nfilterbatch)
                acti_Y_sse, acti_delta, acti_coded = loadrdcurves(archname,l, g, 'acti', nfilterbatch)
                acti_Y_sse, acti_delta, acti_coded = findrdpoints(acti_Y_sse,acti_delta,acti_coded, 2**slope_lambda)
                coded_g[g] = acti_coded[0]
                delta_g[g] = acti_delta[0]
            # acti_Y_sse, acti_delta, acti_coded, acti_mean = loadrdcurves(archname, l, 'acti')
            # acti_Y_sse, acti_delta, acti_coded, acti_mean = findrdpoints(acti_Y_sse,acti_delta,acti_coded, acti_mean, 2**slope_lambda)
            # actbit_layer = acti_coded[0] / dimens[l].prod()
            # if isinstance(net, mobilenetv2py.MobileNetV2):
            #     if flags_convrelu[l] == 0:
            #         actbit_layer = -1
            if isinstance(layers[l], nn.Conv2d):
                layers[l] = quantize_conv_layer.quantConvLayer(layers[l].weight, layers[l].bias, layers[l].weight.shape[1], layers[l].weight.shape[0], 
                                        layers[l].stride, layers[l].padding, pc_phi[l], pc_delta[l], pc_bits[l], nfilterbatch=nfilterbatch, actbit=coded_g, 
                                        groups=layers[l].groups, alpha_act=delta_g, bcw=bcw)
            elif isinstance(layers[l], nn.Linear):
                layers[l] = quantize_conv_layer.quantLinLayer(layers[l].weight, layers[l].bias, layers[l].weight.shape[1], layers[l].weight.shape[0], 
                                                              pc_phi[l], pc_delta[l], pc_bits[l], nfilterbatch=nfilterbatch, actbit=coded_g, 
                                                              alpha_act=delta_g, bcw=bcw)
            else:
                print(layers[l])
                raise NotImplemented

        net = replaceconv(net, layers, includenorm=False)

    if is_multi_gpus == False:
        return net.to(common.device), total_rates/nweights
    else:
        return net, total_rates/nweights


def is_ignored(name, param):
    # classifier or first layer or bias
    return ('fc' in name and param.shape[0] == 1000) or \
           ('bias' in name) or \
           (name == 'Conv2d_2a_3x3.conv.weight')

        #    ('weight' in name and param.shape[1] == 3) or \

import einops

def convert_intnet(net, archname, maxdeadzones, maxrates, rdcurvepath, is_multi_gpus=False, target_rate=8, target_actrate=8,
                    nfilterbatch=1, dimens=None):
    layers = findconv(net, False)

    if isinstance(net, mobilenetv2py.MobileNetV2):
        flags_convrelu = []
        findConvRELU(net, flags_convrelu)
    nweights = cal_total_num_weights(layers)

    with torch.no_grad():
        rd_rate, rd_rate_entropy, rd_dist, rd_phi, rd_delta, rd_delta_mse, rd_dist_mse = \
            load_rd_curve_batch(archname, layers, maxdeadzones, maxrates, rdcurvepath, nfilterbatch)

        for l in range(0, len(layers)):
            pc_phi = [rd_phi[l][c][0, target_rate] for c in range(len(rd_phi[l]))]
            pc_delta = [rd_delta[l][c][0, target_rate] for c in range(len(rd_delta[l]))]
            pc_bits = [rd_rate[l][c][0, target_rate] for c in range(len(rd_rate[l]))]

            acti_Y_sse, acti_delta, acti_coded = loadrdcurves(archname, l, 0, 'acti')
            # acti_Y_sse, acti_delta, acti_coded = findrdpoints(acti_Y_sse,acti_delta,acti_coded, 2**slope_lambda)
            best_j = np.argmin(acti_Y_sse[target_actrate, :, 0])
            acti_delta = [acti_delta[target_actrate, best_j, 0]]

            if isinstance(layers[l], nn.Conv2d):
                layers[l].weight.data = einops.rearrange(layers[l].weight.data, "c1 c2 h w -> c1 h w c2").contiguous()
                new = Conv2d_Q(layers[l].weight, layers[l].bias, 
                               layers[l].in_channels, layers[l].out_channels, layers[l].kernel_size, layers[l].stride, layers[l].padding, layers[l].dilation, layers[l].groups, 
                               pc_phi, pc_delta, pc_bits, nfilterbatch=nfilterbatch, actbit=target_actrate, act_scale=2 ** acti_delta[0])
                layers[l] = new
            if isinstance(layers[l], nn.Linear):
                new = Linear_Q(layers[l].weight, layers[l].bias, layers[l].in_features, layers[l].out_features, 
                               pc_phi, pc_delta, pc_bits, nfilterbatch=nfilterbatch, actbit=target_actrate, 
                               act_scale=2 ** acti_delta[0])
                layers[l] = new
            # else:
            #     print(layers[l])
            #     raise NotImplemented

        net = replaceconv(net, layers, includenorm=False)

    if not is_multi_gpus:
        return net.to(common.device)
    return net


def convert_int_mp_net(net, archname, maxdeadzones, maxrates, slope_lambda, rdcurvepath, is_multi_gpus=False, 
                    nfilterbatch=1, dimens=None, bit_list=[3,7]):
    assert bit_list is not None
    layers = findconv(net, False)

    if isinstance(net, mobilenetv2py.MobileNetV2):
        flags_convrelu = []
        findConvRELU(net, flags_convrelu)
    nweights = cal_total_num_weights(layers)

    with torch.no_grad():
        rd_rate, rd_rate_entropy, rd_dist, rd_phi, rd_delta, rd_delta_mse, rd_dist_mse = \
            load_rd_curve_batch(archname, layers, maxdeadzones, maxrates, rdcurvepath, nfilterbatch)
        bit_list_ = list(map(lambda x:x-1, bit_list))
        
        rd_rate = [d[..., bit_list_] for d in rd_rate]
        rd_dist = [d[..., bit_list_] for d in rd_dist]
        rd_phi = [d[..., bit_list_] for d in rd_phi]
        rd_delta = [d[..., bit_list_] for d in rd_delta]
        pc_phi, pc_delta, pc_bits, pc_rate, pc_size = pareto_condition_batch(layers, rd_rate, rd_dist, rd_phi, rd_delta, 2 ** slope_lambda, nfilterbatch)
        pc_bits = [[bit_list[pb_c] for pb_c in pb] for pb in pc_bits]

        for l in range(0, len(layers)):

            acti_Y_sse, acti_delta, acti_coded = loadrdcurves(archname, l, 0, 'acti')
            # import pdb; pdb.set_trace()
            acti_Y_sse = acti_Y_sse[bit_list]
            acti_delta = acti_delta[bit_list]
            acti_coded = acti_coded[bit_list]
            acti_Y_sse, acti_delta, acti_coded = findrdpoints(acti_Y_sse,acti_delta,acti_coded, 2**slope_lambda)
            actbits = [int(c / (dimens[l][1:].prod() * min(dimens[l][0], nfilterbatch))) for c in acti_coded]
            if isinstance(layers[l], nn.Conv2d):
                layers[l].weight.data = einops.rearrange(layers[l].weight.data, "c1 c2 h w -> c1 h w c2").contiguous()
                new = Conv2d_Q(layers[l].weight, layers[l].bias, 
                               layers[l].in_channels, layers[l].out_channels, layers[l].kernel_size, layers[l].stride, layers[l].padding, layers[l].dilation, layers[l].groups, 
                               pc_phi[l], pc_delta[l], pc_bits[l], nfilterbatch=nfilterbatch, actbit=actbits[0], act_scale=2 ** acti_delta[0])
                layers[l] = new
            if isinstance(layers[l], nn.Linear):
                new = Linear_Q(layers[l].weight, layers[l].bias, layers[l].in_features, layers[l].out_features, 
                               pc_phi[l], pc_delta[l], pc_bits[l], nfilterbatch=nfilterbatch, actbit=actbits[0], act_scale=2 ** acti_delta[0])
                layers[l] = new
            # else:
            #     print(layers[l])
            #     raise NotImplemented

        net = replaceconv(net, layers, includenorm=False)

    if not is_multi_gpus:
        return net.to(common.device)
    return net
