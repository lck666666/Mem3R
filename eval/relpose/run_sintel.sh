#!/bin/bash

set -e

workdir='.'
pose_memory="${POSE_MEMORY:-ttt}"
state_gate="${STATE_GATE:-channel}"
state_update="${STATE_UPDATE:-direct}"
model_names=("${pose_memory}_${state_gate}_${state_update}")

model_weights="${MODEL_WEIGHTS:-checkpoints/default_64_views.pth}"


datasets=('sintel')


for model_name in "${model_names[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/s1/relpose/${data}/${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes "${NUM_PROCESSES:-1}" --main_process_port 29550 eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --pose_memory "$pose_memory" --state_gate "$state_gate" --state_update "$state_update"
done
done
