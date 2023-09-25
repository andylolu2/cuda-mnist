#!/bin/bash

steps=5000

mkdir -p logs

for hidden_size in 128 256 512 1024 2048 4096 8192
do
    for seed in {1..3}
    do
        echo "Running experiment with hidden size $hidden_size and seed $seed for cuda"
        if [ ! -f ./logs/cuda_${hidden_size}_${seed}.log ]; then
            ./cuda/build/train_mnist ./data/MNIST/raw $steps $hidden_size $seed \
                > ./logs/cuda_${hidden_size}_${seed}.log
        else
            echo "Found existing experiment logs, skipping"
        fi
        
        echo "Running experiment with hidden size $hidden_size and seed $seed for torch"
        if [ ! -f ./logs/torch_${hidden_size}_${seed}.log ]; then
            python ./torch/train_mnist.py $steps $hidden_size $seed \
                > ./logs/torch_${hidden_size}_${seed}.log
        else
            echo "Found existing experiment logs, skipping"
        fi
    done
done

