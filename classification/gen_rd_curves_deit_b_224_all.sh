#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
NCHANNELBATCH=128
TESTSIZE=64
BATCHSIZE=64
MODELNAME="deit_base_patch16_224"
QUANT_TYPE="linear"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     echo "part $IDX"
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 generate_rd_curves_batch.py \
#         --archname $MODELNAME \
#         --datapath=/home/kaixin/ssd/imagenet/ \
#         --part_id $IDX --num_parts $CHUNKS \
#         --quant-type $QUANT_TYPE \
#         --nprocessings=1 --modeid=0 --gpuid=0 --nchannelbatch $NCHANNELBATCH --testsize $TESTSIZE --batchsize $BATCHSIZE & #--sort_channels --gen_approx_data
# done
# wait
# echo "All done"


# for IDX in $(seq 0 $((CHUNKS-1))); do
#     echo "part $IDX act curves"
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 generate_RD_curves_act_nchan.py \
#         --archname $MODELNAME \
#         --datapath=/home/kaixin/ssd/imagenet/ \
#         --part_id $IDX --num_parts $CHUNKS --nprocessings=1 --modeid=0 --gpuid=0 --testsize=$TESTSIZE \
#         --batchsize $BATCHSIZE --nchannelbatch $NCHANNELBATCH &  #--profile
# done
# wait
# echo "All done"

FORMAT_TESTSIZE="$(printf "%04d" "$TESTSIZE")"
if [ "$NCHANNELBATCH" -ge 10000 ]; then
    FORMAT_NCHANNELBATCH="$NCHANNELBATCH"
else
    FORMAT_NCHANNELBATCH="$(printf "%04d" "$NCHANNELBATCH")"
fi
# cp -r "tip/${MODELNAME}_nr_0011_ns_${FORMAT_TESTSIZE}_nf_${FORMAT_NCHANNELBATCH}_rdcurves_channelwise_opt_dist_act/" .
# cp -r "tip/${MODELNAME}_ndz_0010_nr_0011_ns_${FORMAT_TESTSIZE}_nf_${FORMAT_NCHANNELBATCH}_rdcurves_channelwise_opt_dist/" .
echo "Generating DP frontier"

# CUDA_VISIBLE_DEVICES=0 python3 generate_DP_batch_distortion_act_nchan.py \
#     --archname $MODELNAME \
#     --pathrdcurve "${MODELNAME}_ndz_0010_nr_0011_ns_${FORMAT_TESTSIZE}_nf_${FORMAT_NCHANNELBATCH}_rdcurves_channelwise_opt_dist/" \
#     --gpuid=0 \
#     --datapath=/home/kaixin/ssd/imagenet \
#     --nchannelbatch $NCHANNELBATCH \
#     --maxrates=11 \
#     --testsize -1 --batchsize 64 \
#     --target_rates 4 6 8 \
#     #--re-train --re-train-lr 0.000001
#     # --output_bit_allocation 
#     #-bcw -bca --bca_version 1


# CUDA_VISIBLE_DEVICES=1 python3 generate_DP_batch_distortion_act_nchan.py \
#     --archname $MODELNAME \
#     --pathrdcurve "${MODELNAME}_ndz_0010_nr_0011_ns_${FORMAT_TESTSIZE}_nf_${FORMAT_NCHANNELBATCH}_rdcurves_channelwise_opt_dist/" \
#     --gpuid=0 \
#     --datapath=/home/kaixin/ssd/imagenet \
#     --nchannelbatch $NCHANNELBATCH \
#     --maxrates=11 \
#     --testsize -1 --batchsize $BATCHSIZE \
#     --target_rates 4 6 \
#     --re-train --re-train-lr 0.0001 --re-train-iter 20 \

CUDA_VISIBLE_DEVICES=3 python3 generate_DP_batch_distortion_act_nchan.py \
    --archname $MODELNAME \
    --pathrdcurve "${MODELNAME}_ndz_0010_nr_0011_ns_${FORMAT_TESTSIZE}_nf_${FORMAT_NCHANNELBATCH}_rdcurves_channelwise_opt_dist/" \
    --gpuid=0 \
    --datapath=/home/kaixin/ssd/imagenet \
    --nchannelbatch $NCHANNELBATCH \
    --maxrates=11 \
    --testsize -1 --batchsize $BATCHSIZE \
    --target_rates 8 \
    --re-train --re-train-lr 0.0002 --re-train-iter 10 \
