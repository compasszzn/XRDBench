#!/bin/bash

# 定义参数

models=('cnn1' 'cnn2' 'cnn3' 'cnn4' 'cnn5' 'cnn6' 'cnn7' 'cnn8' 'cnn9' 'cnn10' 'cnn11' 'mlp' 'rnn' 'lstm' 'gru' 'birnn' 'bilstm' 'bigru' 'transformer' 'iTransformer' 'PatchTST')
seeds=('100' '200' '300' '400' '500')
tasks=('spg' 'crysystem')


for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        python main.py --model "$model" --task "$task"  --seed "$seed"    
    done
  done
done

