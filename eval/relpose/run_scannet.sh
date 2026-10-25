#!/bin/bash

set -e

workdir='.'
pose_memory="${POSE_MEMORY:-ttt}"
state_gate="${STATE_GATE:-channel}"
state_update="${STATE_UPDATE:-direct}"
model_names=("${pose_memory}_${state_gate}_${state_update}")

model_weights="${MODEL_WEIGHTS:-checkpoints/default_64_views.pth}"

# datasets=('scannet_s3_50' 'scannet_s3_90' 'scannet_s3_100' 'scannet_s3_150' 'scannet_s3_200' 'scannet_s3_300' 'scannet_s3_400' 'scannet_s3_500'
#             'scannet_s3_600' 'scannet_s3_700' 'scannet_s3_800' 'scannet_s3_900' 'scannet_s3_1000')
datasets=('scannet_s3_1000')

for model_name in "${model_names[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/relpose/${data}/${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes "${NUM_PROCESSES:-1}" --main_process_port 29550 eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --pose_memory "$pose_memory" --state_gate "$state_gate" --state_update "$state_update"
done
done
