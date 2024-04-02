# RDOQ: Effective Fine-Grained Quantization for DNNs via Rate-Distortion Optimization

Official implementation of the TPAMI submission: "RDOQ: Effective Fine-Grained Quantization for DNNs via Rate-Distortion Optimization".


## Project Structure

This repository contains the post-training quantization implementation of both ImageNet classification and 3D LiDAR based object detection models, organized in the subdirectory respectively:
1. [__ImageNet classification__](#ImageNet-classification): `./classification`
2. [__3D LiDAR based Object Detection__](#3D-LiDAR-based-Object-Detection): `./openpcdet`


## ImageNet classification
Example scripts to evaluate the accuracy on ImageNet validation dataset:

1. Generate rate-distortion curves for channel groups in ViT layer **weights**.
   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 generate_RD_curves_weight_inchangroup.py \
    --archname deit_base_patch16_224 \
    --datapath=/imagenet/ \
    --modeid=0 --gpuid=0 \
    --nchannelbatch 128 \
    --testsize=64 \
    --batchsize 64 
   ```
2. Generate rate-distortion curves for channel groups in ViT layer **activations**.
   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 generate_RD_curves_act_nchan.py \
    --archname deit_base_patch16_224 \
    --datapath=/imagenet/ \
    --modeid=0 --gpuid=0 \
    --nchannelbatch 128 \
    --testsize=64 \
    --batchsize 64
   ```
3. (Option #1) Pareto-frontier method for bit allocation:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 generate_RD_frontier_batch_distortion_act_nchan.py \
    --archname=deit_base_patch16_224 \
    --maxslopesteps 196 \
    --pathrdcurve=deit_base_patch16_224_ndz_0010_nr_0011_ns_0064_nf_0128_rdcurves_channelwise_opt_dist/ \
    --gpuid=0 \
    --datapath=/imagenet \
    --nchannelbatch=128 \
    --maxrates=11 \
    -bcw -bca
   ```
4. (Option #2) Dynamic Programming method for bit allocation:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 generate_DP_batch_distortion_act_nchan.py \
    --archname=deit_base_patch16_224 \
    --maxslopesteps 196 \
    --pathrdcurve=deit_base_patch16_224_ndz_0010_nr_0011_ns_0064_nf_0128_rdcurves_channelwise_opt_dist/ \
    --gpuid=0 \
    --datapath=/imagenet \
    --nchannelbatch=128 \
    --maxrates=11 \
    --target_rates 4 6 8 \
    -bcw -bca
   ```

## 3D LiDAR based Object Detection

### Preparation: 

Follow installation instructions in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

1. Generate rate-distortion curves for channel groups in detection model layer **weights**.
   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 generate_rd_curves_batch.py \
      --batch_size 2 --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext_vtc.yaml \
      --pretrained_model path/to/model.pth.tar \
      --workers 1
   ```

2. Dynamic Programming method for bit allocation:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 generate_DP_frontier_batch_act_nchan.py \
    --batch_size 32 --workers 4 \
    --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext_vtc.yaml \
    --pretrained_model path/to/model.pth.tar \
    --nchannelbatch 128 \
    --pathrdcurve VoxelNeXt_ndz_0010_nr_0011_ns_0001_nf_0128_rdcurves_channelwise_opt_dist/ \
    --target_rates 2 4 6 8 -bcw
   ```