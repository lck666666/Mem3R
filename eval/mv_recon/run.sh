#!/bin/bash

set -e

workdir='.'
pose_memory="${POSE_MEMORY:-ttt}"
state_gate="${STATE_GATE:-channel}"
state_update="${STATE_UPDATE:-direct}"
model_names=("${pose_memory}_${state_gate}_${state_update}")
model_weights="${MODEL_WEIGHTS:-checkpoints/default_64_views.pth}"

for model_name in "${model_names[@]}"; do

# for max_frames in 50 100 150 200 250 300 350 400
for max_frames in 200

do
    output_dir="${workdir}/eval_results/video_recon/7scenes_${max_frames}/${model_name}"
    echo "$output_dir"
    NCCL_TIMEOUT=360000 accelerate launch --num_processes "${NUM_PROCESSES:-1}" --main_process_port 29502 eval/mv_recon/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --pose_memory "$pose_memory" --state_gate "$state_gate" --state_update "$state_update" \
        --max_frames $max_frames \

done
done
