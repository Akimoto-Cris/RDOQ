import torch
import resnetpy
import vggpy
import alexnetpy
import densenetpy
import mobilenetv2py
import models_mae
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import transconv
import timm
import timm.models.vision_transformer as vit
import scipy.io as io
import torch.optim as optim
import numpy as np
import models_vit


import quantize_conv_layer
from quantize_conv_layer import *

device = torch.device("cuda:0")

rgb_avg = [0.5, 0.5, 0.5] # [0.485, 0.456, 0.406]
rgb_std = [0.5, 0.5, 0.5] # [0.229, 0.224, 0.225]
PARAMETRIZED_MODULE_TYPES = (torch.nn.Linear, 
                             torch.nn.Conv2d, 
                             transconv.TransConv2d, 
                             quantize_conv_layer.quantConvLayer)
NORM_MODULE_TYPES = (torch.nn.BatchNorm2d,
                     torch.nn.LayerNorm)

transdata = transforms.Compose(
	[transforms.Resize(256,interpolation=1),
	 transforms.CenterCrop(224),
	 transforms.ToTensor(),
	 transforms.Normalize(rgb_avg, rgb_std)])

mae_archname_to_vit_archname = {
    "mae_finetuned_vit_base": "vit_base_patch16_224",
    "mae_finetuned_vit_large": "vit_large_patch16_224",
    "mae_finetuned_vit_huge": "vit_huge_patch16_224"
}

def loadnetwork(archname, gpuid, act_bitwidth=-1):
    global device
    device = torch.device("cuda:" + str(gpuid) if torch.cuda.is_available() else "cpu")

    if archname == 'alexnetpy':
        net = alexnetpy.alexnet(pretrained=True)
    elif archname == 'vgg16py':
        net = vggpy.vgg16_bn(pretrained=True)
    elif archname == 'resnet18py':
        net = resnetpy.resnet18(pretrained=True, act_bitwidth=act_bitwidth)
    elif archname == 'resnet34py':
        net = resnetpy.resnet34(pretrained=True, act_bitwidth=act_bitwidth)
    elif archname == 'resnet50py':
        net = resnetpy.resnet50(pretrained=True, act_bitwidth=act_bitwidth)
    elif archname == 'densenet121py':
        net = densenetpy.densenet121(pretrained=True)
    elif archname == 'mobilenetv2py':
        net = mobilenetv2py.mobilenet_v2(pretrained=True)
    elif archname == 'vit_mae_base_patch16':
        net = models_mae.mae_vit_base_patch16(pretrained=True)
    elif "mae" in archname:
        net = models_vit.__dict__[archname](drop_path_rate=0.1, global_pool=True)
        pretrained_sd = torch.load(f"{archname}.pth")["model"]
        net.load_state_dict(pretrained_sd)
    elif "vit" in archname or "deit" in archname or "swin" in archname:
        # import pdb; pdb.set_trace()
        net = timm.create_model(archname, pretrained=True)
    
    return net.to(device)


def loadnetwork_multi_gpus(archname):

    if archname == 'alexnetpy':
        net = alexnetpy.alexnet(pretrained=True)
    elif archname == 'vgg16py':
        net = vggpy.vgg16_bn(pretrained=True)
    elif archname == 'resnet18py':
        net = resnetpy.resnet18(pretrained=True)
    elif archname == 'resnet34py':
        net = resnetpy.resnet34(pretrained=True)
    elif archname == 'resnet50py':
        net = resnetpy.resnet50(pretrained=True)
    elif archname == 'densenet121py':
        net = densenetpy.densenet121(pretrained=True)
    elif "deit" in archname:
        net = deitpy.__dict__[archname](pretrained=True)

    return net


def loadvaldata(datapath, gpuid, testsize=-1):
    global device
    device = torch.device("cuda:" + str(gpuid) if torch.cuda.is_available() else "cpu")

    images = datasets.ImageNet(
                root=datapath,
                split='val',transform=transdata)

    if testsize != -1:
        images.samples = images.samples[::len(images.samples) // testsize]
    labels = torch.tensor([images.samples[i][1] for i in range(0, len(images))])

    return images, labels.to(device)


def loadvaldata_multi_gpus(datapath, testsize=-1):
    images = datasets.ImageNet(
                root=datapath,
                split='val',transform=transdata)

    if testsize != -1:
        images.samples = images.samples[::len(images.samples) // testsize]
    labels = torch.tensor([images.samples[i][1] for i in range(0, len(images))])

    return images, labels


def loadtraindata(datapath, gpuid):
    global device
    device = torch.device("cuda:" + str(gpuid) if torch.cuda.is_available() else "cpu")

    images = datasets.ImageNet(
                root=datapath,
                split='train',transform=transdata)

    labels = torch.tensor([images.samples[i][1] for i in range(0, len(images))])

    return images, labels.to(device)

def loadtraindata_multi_gpus(datapath):

    images = datasets.ImageNet(
                root=datapath,
                split='train',transform=transdata)

    labels = torch.tensor([images.samples[i][1] for i in range(0, len(images))])

    return images, labels


import tqdm


def predict(net, images, batch_size=256, num_workers=16):
    global device
    y_hat = torch.zeros(0, device=device)
    loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            y_hat = torch.cat((y_hat,net(x)))
    return y_hat

def predict2(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, _ in iter(loader):
            x = x.to(device)
            y_hat = torch.cat((y_hat,net(x)))
    return y_hat


def predict2_withgt(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    y_gt = torch.zeros(0, device=device)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, y in tqdm.tqdm(iter(loader)):
            x = x.to(device)
            y_hat = torch.cat((y_hat,net(x)))
            y_gt = torch.cat((y_gt, y))
    return y_hat, y_gt


def predict_dali(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in iter(loader):
            x = data[0]["data"]
            res = net(x)
            y_hat = torch.cat((y_hat,res))
    loader.reset()
    return y_hat


def predict_dali_withgt(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    y_gt = torch.zeros(0, device=device)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            x = data[0]["data"]
            y_hat = torch.cat((y_hat,net(x)))
            y = data[0]["label"]
            y_gt = torch.cat((y_gt, y))
    loader.reset()
    return y_hat, y_gt


def retrain_bias(net, loader, lr=0.0001, iters=100, epochs=1):
    from timm.loss import LabelSmoothingCrossEntropy
    global device

    layers = [module for module in net.modules() if isinstance(module, transconv.QAWrapper)]

    # for layer in layers:
    #     layer.quantized = False

    for name, param in net.named_parameters():
        if "norm" in name: continue
        if "weight" in name:
            param.requires_grad = False
        if "bias" in name:
            param.requires_grad = True

    net.train()
    
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        pbar = tqdm.tqdm(loader)
        for i, data in enumerate(pbar):
            if iters != -1 and i > iters:
                break
            inputs, labels = data[0]["data"], data[0]["label"][:, 0].long()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item():.4f}")

        loader.reset()

    # for layer in layers:
    #     layer.quantized = True

    print('Finished tuning bias')
    net.eval()
    return net

import torch.optim.lr_scheduler as lrsched

def retrain(net, loader, lr=0.0001, iters=20, epochs=1, testloader=None):
    from timm.loss import LabelSmoothingCrossEntropy
    
    global device

    net.train()
    
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    sched = lrsched.CosineAnnealingLR(optimizer, epochs * iters)
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        pbar = tqdm.tqdm(loader)
        for i, data in enumerate(pbar):
            if iters != -1 and i > iters:
                break
            inputs, labels = data[0]["data"], data[0]["label"][:, 0].long()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #sched.step()

            if testloader is not None:
                with torch.no_grad():
                    Y, labels = predict_dali_withgt(net, testloader)
                acc1, acc5 = accuracy(Y, labels, topk=(1, 5))
                print(f"Test: Top1: {acc1.item():.2f}, Top5: {acc5.item():.2f}")
            
            pbar.set_description(f"Loss: {loss.item():.4f}")

        loader.reset()

    print('Finished retraining')
    net.eval()
    return net



@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #pred.reshape(pred.shape[0], -1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def replaceconv(net,layers,includenorm=True):
    pushconv([layers],net,includenorm,direction=1)
    return net

def findconv(net,includenorm=True):
    layers = pushconv([[]],net,includenorm)
    return layers

def findConvRELU(container, flags):
    if isinstance(container, mobilenetv2py.MobileNetV2):
        findConvRELU(container.features,flags)
        findConvRELU(container.classifier,flags)
        return
    elif isinstance(container, mobilenetv2py.InvertedResidual):
        findConvRELU(container.conv,flags)
        return
    elif isinstance(container, mobilenetv2py.ConvBNReLU):
        flags.append(1)
        return
    elif isinstance(container,torch.nn.Sequential):
        for attr in range(0,len(container)):
            findConvRELU(container[attr],flags)
        return
    elif isinstance(container, torch.nn.Conv2d):
        flags.append(0)
        return
    elif isinstance(container, torch.nn.Linear):
        flags.append(0)
        return
    return


def quantize(weights, delta, b):
    if b > 0:
        minpoint = -(2**(b-1))*delta
        maxpoint = (2**(b-1) - 1)*delta
    else:
        minpoint = 0
        maxpoint = 0
    # return (delta*(weights/delta).round()).clamp(minpoint,maxpoint)
    return weights.div(delta).round_().mul_(delta).clamp_(minpoint, maxpoint)

    

def getdevice():
	global device
	return device


def pushattr(layers,container,attr,includenorm,direction, prefix=""):
    if isinstance(getattr(container,attr, None), PARAMETRIZED_MODULE_TYPES) or \
            (isinstance(getattr(container, attr, None), NORM_MODULE_TYPES) and includenorm):
        # setattr(container,attr,TimeWrapper(getattr(container,attr), prefix))

        if direction == 0:
            layers[0].append(getattr(container,attr))
        else:
            setattr(container,attr,layers[0][0])
            layers[0] = layers[0][1:len(layers[0])]
    # print(container.__class__.__name__, attr)

def pushlist(layers,container,attr,includenorm,direction, prefix=""):
    if isinstance(container[attr], PARAMETRIZED_MODULE_TYPES) or \
            (isinstance(container[attr], NORM_MODULE_TYPES) and includenorm):
        # container[attr] = TimeWrapper(container[attr], prefix)
        if direction == 0:
            layers[0].append(container[attr])
        else:
            container[attr] = layers[0][0]
            layers[0] = layers[0][1:len(layers[0])]
    else:
        pushconv(layers,container[attr],includenorm,direction, prefix=prefix)

def pushconv(layers, container, includenorm=True, direction=0, prefix="model"):
    # if isinstance(container,resnetpy.ResNet):
    #     pushattr(layers,container,'conv1',includenorm,direction)
    #     pushattr(layers,container,'bn1',includenorm,direction)
    #     pushconv(layers,container.layer1,includenorm,direction)
    #     pushconv(layers,container.layer2,includenorm,direction)
    #     pushconv(layers,container.layer3,includenorm,direction)
    #     pushconv(layers,container.layer4,includenorm,direction)
    #     pushattr(layers,container,'fc',includenorm,direction)
    # elif isinstance(container,models.densenet.DenseNet):
    #     pushconv(layers,container.features,includenorm,direction)
    #     pushattr(layers,container,'classifier',includenorm,direction)
    # elif isinstance(container, alexnetpy.AlexNet):
    #     pushconv(layers,container.features,includenorm,direction)
    #     pushconv(layers,container.classifier,includenorm,direction)
    # elif isinstance(container, vggpy.VGG):
    #     pushconv(layers,container.features,includenorm,direction)
    #     pushconv(layers,container.classifier,includenorm,direction)
    # elif isinstance(container, resnetpy.BasicBlock):
    #     pushattr(layers,container,'conv1',includenorm,direction)
    #     pushattr(layers,container,'bn1',includenorm,direction)
    #     pushattr(layers,container,'conv2',includenorm,direction)
    #     pushattr(layers,container,'bn2',includenorm,direction)
    #     pushconv(layers,container.downsample,includenorm,direction)
    # elif isinstance(container, resnetpy.Bottleneck):
    #     pushattr(layers,container,'conv1',includenorm,direction)
    #     pushattr(layers,container,'bn1',includenorm,direction)
    #     pushattr(layers,container,'conv2',includenorm,direction)
    #     pushattr(layers,container,'bn2',includenorm,direction)
    #     pushattr(layers,container,'conv3',includenorm,direction)
    #     pushattr(layers,container,'bn3',includenorm,direction)
    #     pushconv(layers,container.downsample,includenorm,direction)
    # elif isinstance(container, models.densenet._DenseBlock):
    #     for l in range(0,25):
    #         if hasattr(container,'denselayer%d'%l):
    #             pushconv(layers,getattr(container,'denselayer%d'%l),includenorm,direction)
    # elif isinstance(container, models.densenet._DenseLayer):
    #     from timm.models.vision_transformer import VisionTransformer
    #     pushattr(layers,container,'conv1',includenorm,direction)
    #     pushattr(layers,container,'norm2',includenorm,direction)
    #     pushattr(layers,container,'conv2',includenorm,direction)
    # elif isinstance(container, models.densenet._Transition):
    #     pushattr(layers,container,'norm',includenorm,direction)
    #     pushattr(layers,container,'conv',includenorm,direction)
    # elif isinstance(container,(torch.nn.Sequential, torch.nn.ModuleList)):
    #     for attr in range(0,len(container)):
    #         pushlist(layers,container,attr,includenorm,direction, prefix=prefix+f".{attr}")

    # #elif isinstance(container, mobilenetv2py.MobileNetV2):
    # #    pushconv(layers, container.features, includenorm, direction)
    # #    pushconv(layers, container.classifier, includenorm, direction)
    # #elif isinstance(container, mobilenetv2py.ConvBNReLU):
    # #    for l in range(0, len(container.conv)):
    # #        pushconv(layers, container.conv[l], includenorm, direction)
    # #elif isinstance(container, mobilenetv2py.InvertedResidual):
    # #    for l in range(0, len(container.conv)):
    # #        pushconv(layers, container.conv[l], includenorm, direction)

    # elif isinstance(container, mobilenetv2py.MobileNetV2):
    #     pushconv(layers,container.features,includenorm, direction)
    #     pushconv(layers,container.classifier,includenorm, direction)
    # elif isinstance(container, mobilenetv2py.ConvBNReLU):
    #     #for l in range(0, len(container.conv)):
    #         #pushconv(layers, container.conv[l], includenorm, direction)
    #     for attr in range(0,len(container)):
    #         pushlist(layers,container,attr,includenorm,direction)

    # elif isinstance(container, mobilenetv2py.InvertedResidual):
    #     #for l in range(0, len(container.conv)):
    #         #pushconv(layers, container.conv[l], includenorm, direction)
    #     pushconv(layers, container.conv, includenorm, direction)
        
    # # elif isinstance(container, vitpy.VisionTransformer):
    # #     pushconv(layers, container.patch_embed, includenorm, direction, prefix=prefix+".patch_embed")
    # #     pushconv(layers, container.blocks, includenorm, direction, prefix=prefix+".blocks")
    # #     # pushattr(layers, container, "norm", includenorm, direction)
    # #     pushattr(layers, container, "head", includenorm, direction, prefix=prefix+".head")
    
    # # elif isinstance(container, deitpy.VisionTransformer):
    # #     pushconv(layers, container.patch_embed, includenorm, direction, prefix=prefix+".patch_embed")
    # #     pushconv(layers, container.blocks, includenorm, direction, prefix=prefix+".blocks")
    # #     # pushattr(layers, container, "norm", includenorm, direction)
    # #     pushattr(layers, container, "head", includenorm, direction, prefix=prefix+".head")
    # #     pushattr(layers, container, "head_dist", includenorm, direction, prefix=prefix+".head_dist")
            
    # elif isinstance(container, vit.PatchEmbed):
    #     pushattr(layers,container,'proj',includenorm,direction, prefix=prefix+".proj")
        
    # elif isinstance(container, vit.Block):
    #     # pushattr(layers, container, "norm1", includenorm, direction)
    #     pushconv(layers, container.attn, includenorm, direction, prefix=prefix+".attn")
    #     # pushattr(layers, container, "norm2", includenorm, direction)
    #     pushconv(layers, container.mlp, includenorm, direction, prefix=prefix+".mlp")
        
    # elif isinstance(container, vit.Attention):
    #     pushattr(layers, container, "qkv", includenorm, direction, prefix=prefix+".qkv")
    #     pushattr(layers, container, "proj", includenorm, direction, prefix=prefix+".proj")
    
    # elif isinstance(container, vit.Mlp):
    #     pushattr(layers, container, "fc1", includenorm, direction, prefix=prefix+".fc1")
    #     pushattr(layers, container, "fc2", includenorm, direction, prefix=prefix+".fc2")

    # # mae
    # elif isinstance(container, models_mae.MaskedAutoencoderViT):
    #     pushconv(layers, container.patch_embed, includenorm, direction, prefix=prefix+".patch_embed")
    #     pushconv(layers, container.blocks, includenorm, direction, prefix=prefix+".blocks")
    #     # pushattr(layers, container, "norm", includenorm, direction)
    #     pushattr(layers, container, "head", includenorm, direction, prefix=prefix+".head")

    # elif isinstance(container, models_mae.PatchEmbed):
    #     pushattr(layers,container,'proj',includenorm,direction, prefix=prefix+".proj")

    # elif isinstance(container, models_mae.Block):
    #     # pushattr(layers, container, "norm1", includenorm, direction)
    #     pushconv(layers, container.attn, includenorm, direction, prefix=prefix+".attn")
    #     # pushattr(layers, container, "norm2", includenorm, direction)
    #     pushconv(layers, container.mlp, includenorm, direction, prefix=prefix+".mlp")

    # elif isinstance(container, models_mae.Attention):
    #     pushattr(layers, container, "qkv", includenorm, direction, prefix=prefix+".qkv")
    #     pushattr(layers, container, "proj", includenorm, direction, prefix=prefix+".proj")

    # elif isinstance(container, models_mae.Mlp):
    #     pushattr(layers, container, "fc1", includenorm, direction, prefix=prefix+".fc1")
    #     pushattr(layers, container, "fc2", includenorm, direction, prefix=prefix+".fc2")
    
    # else:
    return [m for m in container.modules() if isinstance(m, PARAMETRIZED_MODULE_TYPES) or (isinstance(m, NORM_MODULE_TYPES) and includenorm)]

    # return layers[0]


import os, glob
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def

    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True, testsize=-1, args=None):
    if testsize != -1:
        labels = []
        files = []
        # import pdb; pdb.set_trace()
        for i, l in enumerate(sorted(os.listdir(data_dir))):
            ps = glob.glob(os.path.join(data_dir, l, "*.JPEG"))
            files += ps
            labels += [i] * len(ps)
        labels = labels[::len(files) // testsize][:-1]
        files = files[::len(files) // testsize][:-1]
        print(is_training, len(files))
        images, labels = fn.readers.file(files=files,
                                        labels=labels,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")
    else:
        images, labels = fn.readers.file(file_root=data_dir,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in  the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
    images = fn.resize(images,
                        device=dali_device,
                        size=size,
                        mode="not_smaller",
                        interp_type=types.INTERP_CUBIC)

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[d * 255 for d in args.mean],
                                      std=[d * 255 for d in args.std],
                                      mirror=False)
    labels = labels.gpu()
    return images, labels


def get_trainval_imagenet_dali_loader(args, crop_size=224, val_size=256):
    args.local_rank = 0
    args.dali_cpu = False
    args.world_size = 1
    args.workers = 1
    traindir = os.path.join(args.datapath, 'train')
    
    pipe = create_dali_pipeline(batch_size=args.batchsize,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=traindir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=True,
                                testsize=args.testsize,
                                args=args)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    # args.testsize = -1
    val_loader = get_val_imagenet_dali_loader(args, crop_size, val_size)
    return train_loader, val_loader


def get_val_imagenet_dali_loader(args, crop_size=224, val_size=256):
    args.local_rank = 0
    args.dali_cpu = False
    args.world_size = 1
    args.workers = 1
    valdir = os.path.join(args.datapath, 'val')
    pipe = create_dali_pipeline(batch_size=args.batchsize,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False,
                                testsize=args.val_testsize,
                                args=args)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    return val_loader


def get_calib_imagenet_dali_loader(args, crop_size=224, val_size=256):
    args.local_rank = 0
    args.dali_cpu = False
    args.world_size = 1
    args.workers = 1
    valdir = os.path.join(args.datapath, 'train')
    pipe = create_dali_pipeline(batch_size=args.batchsize,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False,
                                testsize=args.val_testsize,
                                args=args)
    pipe.build()
    calib_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    return calib_loader


class TimeWrapper(nn.Module):
    def __init__(self, wrapped_module, tag):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.tag = tag

    def forward(self, *args):
        ret = self.wrapped_module(*args)
        if isinstance(ret, torch.Tensor):
            print(f"ModuleType: {self.tag}, shape:", ret.shape, "count:", ret.numel())
        else:
            print(f"ModuleType: {self.tag}, shape:", [r.shape for r in ret], "count:", sum(r.numel() for r in ret))
        return ret
    
    @property
    def weight(self):
        return getattr(self.wrapped_module, "weight", None)


def convert_qconv(network, stats=True):
    layers = findconv(network, False)

    for l in range(0, len(layers)):
        layers[l] = transconv.QAWrapper(layers[l], [0], [0])
        layers[l].stats = stats

    network = replacelayer(network, [layers], (nn.Linear, nn.Conv2d))
    return network.cuda(), layers

def convert_qconv_new(network, stats=True):
    for name, module in reversed(network._modules.items()):

        if len(list(module.children())) > 0:
            network._modules[name] = convert_qconv_new(module, stats)

        if isinstance(module, PARAMETRIZED_MODULE_TYPES):
            layer_new = transconv.QAWrapper(module, [0], [0])
            layer_new.stats = stats
            network._modules[name] = layer_new

    return network


def convert_qconv_diffable(network, stats=True):
    layers = findconv(network, False)

    for l in range(0, len(layers)):
        layers[l] = transconv.DiffableQAWrapper(layers[l], [0], [0])
        layers[l].stats = stats

    network = replacelayer(network, [layers], (nn.Linear, nn.Conv2d))
    return network.cuda(), layers


def replacelayer(module, layers, classes):
    module_output = module
    # base case
    if isinstance(module, classes):
        module_output, layers[0] = layers[0][0], layers[0][1:]
    # recursive
    for name, child in module.named_children():
        module_output.add_module(name, replacelayer(child, layers, classes))
    del module
    return module_output


def hooklayers(layers, backward=False):
    return [Hook(layer, backward) for layer in layers]


def hooklayers_with_fp_act(layers, fp_acts):
    return [Hook(layer, fp_act=fp_act) for layer, fp_act in zip(layers, fp_acts)]


class Hook:
    def __init__(self, module, backward=False, fp_act=None):
        self.backward = backward
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        self.fp_act = fp_act

    def hook_fn(self, module, input, output):
        self.input_tensor = input[0]
        self.output_tensor = output if not self.backward else output[0]
        self.input = torch.tensor(self.input_tensor.shape[1:]) if self.input_tensor is not None else None
        self.output = torch.tensor(self.output_tensor.shape[1:]) if self.output_tensor is not None else None
        if getattr(module, 'stats', False):
            self.mean_err_a = module.sum_err_a / module.count
        if self.fp_act is not None:
            # self.accum_err_act = (self.fp_act - self.input[0]).div_(self.fp_act.max()).pow_(2).mean()
            # print(self.fp_act.unique())
            self.accum_err_act = (self.fp_act - self.input_tensor).div_(self.fp_act.abs().max()).pow_(2).mean()

    def close(self):
        self.hook.remove()


def gettop1(logits):
    return logits.max(1)[1]


def gettopk(logp,k=1):
    logp = logp.exp()
    logp = logp/logp.sum(1).reshape(-1,1)
    vals, inds = logp.topk(k,dim=1)

    return inds


def loadvarstats(archname,testsize):
    mat = io.loadmat(('%s_stats_%d.mat' % (archname, testsize)))
    return np.array(mat['cov'])


def loadrdcurves(archname,l,g,part,nchannelbatch=-1, Amse=False,testsize=64, mode="in_channel", prefix="./"):
    if nchannelbatch>0:
        # print(f"loading act curves from: {archname}_nr_0011_ns_0064_nf_{nchannelbatch:04d}_rdcurves_channelwise_opt_dist_act{'_Amse' if Amse else ''}")
        # print(f"loading act curves from: {archname}_nr_0011_ns_{batchsize:04d}_nf_{nchannelbatch:04d}_rdcurves_channelwise_opt_dist_act{'_Amse' if Amse else ''}")
        mat = io.loadmat(f'{prefix}/{archname}_nr_0011_ns_{testsize:04d}_nf_{nchannelbatch:04d}_rdcurves{"_out" if mode == "out_channel" else ""}_channelwise_opt_dist_act{"_Amse" if Amse else ""}/{archname}_val_{l:03d}_{g:04d}_output_{part}')
    else:
        # print(f"loading act curves from: {archname}_nr_0011_ns_0064_rdcurves_channelwise_opt_dist_act/{archname}_val_{l:03d}_0064_output_{part}")
        mat = io.loadmat(f'{prefix}/{archname}_nr_0011_ns_{testsize:04d}_rdcurves_channelwise_opt_dist_act/{archname}_val_{l:03d}_{testsize:04d}_output_{part}')
    return mat['%s_Y_sse'%part], mat['%s_delta'%part], mat['%s_coded'%part]
    #mat = io.loadmat('%s_%s_val_1000_%d_%d_output_%s_%s' % (archname,tranname,l+1,l+1,trantype,part))
    #return mat['%s_Y_sse'%part][l,0], mat['%s_delta'%part][l,0], mat['%s_coded'%part][l,0]


def findrdpoints(y_sse,delta,coded,lam_or_bit, is_bit=False, smooth_dists=True):
    # find the optimal quant step-size
    y_sse[np.isnan(y_sse)] = float('inf')
    ind1 = np.nanargmin(y_sse,1)
    ind0 = np.arange(ind1.shape[0]).reshape(-1,1).repeat(ind1.shape[1],1)
    ind2 = np.arange(ind1.shape[1]).reshape(1,-1).repeat(ind1.shape[0],0)
    inds = np.ravel_multi_index((ind0,ind1,ind2),y_sse.shape) # bit_depth x blocks
    y_sse = y_sse.reshape(-1)[inds]
    delta = delta.reshape(-1)[inds]
    coded = coded.reshape(-1)[inds]
    # mean = mean.reshape(-1)[inds]
    # find the minimum Lagrangian cost
    if is_bit:
        point = coded == lam_or_bit
    else:
        point = y_sse + lam_or_bit*coded == (y_sse + lam_or_bit*coded).min(0)
    return np.select(point, y_sse), np.select(point, delta), np.select(point, coded)#, np.select(point, mean)





def binary_search(min_val, max_val, target_func, target_val, epsilon=0.02, max_iters=40):
    l = min_val
    r = max_val
    cnt = 0
    while l < r:
        mid = (l + r) / 2
        y_mid = target_func(mid)

        if abs(y_mid - target_val) <= epsilon:
            return mid
        elif y_mid < target_val:
            l = mid
        elif y_mid > target_val:
            r = mid
        
        cnt += 1
        if cnt >= max_iters:
            y_l = target_func(l)
            y_r = target_func(r)
            if abs(y_mid - target_val) > abs(y_l - target_val) and abs(y_r - target_val) > abs(y_l - target_val):
                mid = l
            elif abs(y_mid - target_val) > abs(y_r - target_val) and abs(y_l - target_val) > abs(y_r - target_val):
                mid = r
            break
    return mid


# import pickle as pkl
# def loadmeanstd(archname, l, part):
#     with open(f'{archname}_nr_0011_ns_0064_rdcurves_channelwise_opt_dist_act/{archname}_val_{l:03d}_0064_output_{part}_meanstd.pkl', 'rb') as f:
#         d = pkl.load(f)
#     return d

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.v = 0
        self.sum = 0
        self.cnt = 0

    def update(self, v):
        self.v = v
        self.sum += v
        self.cnt += 1

    @property
    def avg(self):
        return self.sum / self.cnt
