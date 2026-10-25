#!/bin/bash

set -e

workdir='.'
pose_memory="${POSE_MEMORY:-ttt}"
state_gate="${STATE_GATE:-channel}"
state_update="${STATE_UPDATE:-direct}"
model_names=("${pose_memory}_${state_gate}_${state_update}")
model_weights="${MODEL_WEIGHTS:-checkpoints/default_64_views.pth}"
# datasets=('bonn_s1_50' 'bonn_s1_100' 'bonn_s1_110' 'bonn_s1_150' 'bonn_s1_200' 'bonn_s1_250' 'bonn_s1_300' 'bonn_s1_350' 'bonn_s1_400' 'bonn_s1_450' 'bonn_s1_500')
datasets=('bonn_s1_500')


for model_name in "${model_names[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}/${model_name}"
    echo "$output_dir"

    accelerate launch --num_processes "${NUM_PROCESSES:-1}" --main_process_port 29556 eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --pose_memory "$pose_memory" --state_gate "$state_gate" --state_update "$state_update"

    # scale&shift scale metric
    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "metric"

    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"

    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale&shift"
done
done
