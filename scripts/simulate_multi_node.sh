#!/bin/bash

#
# simulate multi nodes on one gpu. start N torchrun on X gpu locally.
# example how to run ./scripts/simulate_multi_node.sh 2 1  src/zeroband/train.py @configs/debug/debug.toml

# Function to get CUDA devices based on the number of GPUs and index
function get_cuda_devices() {
    local num_gpu=$1
    local index=$2
    local start_gpu=$((num_gpu * index))
    local end_gpu=$((start_gpu + num_gpu - 1))

    if [ "$num_gpu" -eq 1 ]; then
        echo $start_gpu
    else
        echo $(seq -s ',' $start_gpu $end_gpu)
    fi
}

# Check if at least three arguments were passed
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <N> <initial_peer> <num_gpu> [additional_python_args]"
    exit 1
fi

N=$1         # Set N from the first argument
NUM_GPU=$2
shift 2     # Remove the first three arguments so $@ contains only additional Python arguments

mkdir -p logs


for i in $(seq 0 $(($N - 1 )))
do
    CUDA_VISIBLE_DEVICES=$(get_cuda_devices $NUM_GPU $i) torchrun --nproc_per_node=$NUM_GPU --node-rank $i --rdzv-endpoint localhost:9999 --nnodes=$N  $@  > logs/log$i 2>&1 &
done

tail -f logs/log0
