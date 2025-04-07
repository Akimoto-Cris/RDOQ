#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
NCHANNELBATCH=64 
TESTSIZE=128
BATCHSIZE=4
MODELNAME="deit_base_patch16_224"
QUANT_TYPE="linear"
SOLVER="dp"
DATASET="/imagenet"


start_time=$(date +%s)
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "part $IDX, device ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 generate_rd_curves_batch_hessian.py \
        --archname $MODELNAME \
        --datapath=$DATASET \
        --part_id $IDX --num_parts $CHUNKS \
        --mute-print \
        --quant-type $QUANT_TYPE \
        --nprocessings=1 --modeid=0 --gpuid=0 --nchannelbatch $NCHANNELBATCH --testsize $TESTSIZE --batchsize $BATCHSIZE &  #--sort_channels --gen_approx_data
done
wait
echo "All done"


for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "part $IDX act curves"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 generate_RD_curves_act_out_nchan_hessian.py \
        --archname $MODELNAME \
        --datapath=$DATASET \
        --part_id $IDX --num_parts $CHUNKS --nprocessings=1 --modeid=0 --gpuid=0 --testsize=$TESTSIZE \
        --mute-print \
        --batchsize $BATCHSIZE --nchannelbatch $NCHANNELBATCH & #--profile
done
wait
echo "All done"

end_time=$(date +%s)
time_elapsed=$((end_time - start_time))
echo "Time elapsed for generating RD curves: $time_elapsed seconds"


FORMAT_TESTSIZE="$(printf "%04d" "$TESTSIZE")"
if [ "$NCHANNELBATCH" -ge 10000 ]; then
    FORMAT_NCHANNELBATCH="$NCHANNELBATCH"
else
    FORMAT_NCHANNELBATCH="$(printf "%04d" "$NCHANNELBATCH")"
fi

if [ "$SOLVER" == "lg" ]; then
    echo "Generating LG frontier"
    CUDA_VISIBLE_DEVICES=0 python3 generate_LG_batch_distortion_act_out_nchan.py \
        --archname $MODELNAME \
            --pathrdcurve "./hessian_curves/${MODELNAME}_ndz_0010_nr_0011_ns_${FORMAT_TESTSIZE}_nf_${FORMAT_NCHANNELBATCH}_rdcurves_channelwise_opt_dist/" \
            --gpuid=0 \
            --datapath=/home/kaixin/ssd/imagenet \
            --nchannelbatch $NCHANNELBATCH \
            --maxrates=11 \
            --testsize -1 --batchsize 64 \
            --minrate 3 \
            --target_rates 4 6 8 \
            --act_curve_root_dir "./hessian_curves" \
            --smooth-dists \
            -bcw -bca \
            --re-train --re-train-lr 0.00001 --re-train-epoch 1 --re-train-iter 100 
else
    echo "Generating DP frontier"
    CUDA_VISIBLE_DEVICES=1 python3 generate_DP_batch_distortion_act_out_nchan.py \
        --archname $MODELNAME \
        --pathrdcurve "./hessian_curves/${MODELNAME}_ndz_0010_nr_0011_ns_${FORMAT_TESTSIZE}_nf_${FORMAT_NCHANNELBATCH}_rdcurves_channelwise_opt_dist/" \
        --gpuid=0 \
        --datapath=$DATASET \
        --nchannelbatch $NCHANNELBATCH \
        --maxrates=11 \
        --testsize -1 --batchsize 64 \
        --minrate 3 \
        --target_rates 4 6 8 \
        --act_curve_root_dir "./hessian_curves" \
        --re-train --re-train-lr 0.00001 --re-train-epoch 1 --re-train-iter 100 \
        --smooth-dists \
        -bcw -bca

fi
