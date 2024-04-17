#!/bin/bash

# 定义参数


models=('cnn2' 'cnn3' 'cnn' 'pqnet' 'icsd' 'mp' 'autoanalyzer' 'xca')
tasks=('spg','crysystem')


for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for i in {1..1}; do
        # 构建日志文件名
        log_file="/home/zinanzheng/project/github/XRDBench/log_all/${model}_${task}.txt"
        
        # 执行 Python 脚本，并将输出追加到日志文件
        python main.py --model "$model" --task "$task"  >> "$log_file" 2>&1
        
        # 输出当前运行信息
        echo "Run $i completed. Log saved to $log_file"
    done
  done
done
