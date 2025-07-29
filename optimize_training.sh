#!/bin/bash

# 训练速度优化脚本
# 使用方法: bash optimize_training.sh

echo "=== 训练速度优化设置 ==="

# 1. 设置CUDA优化环境变量
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_CUDNN_V8_API_ENABLED=1

# 2. 设置PyTorch优化
export TORCH_SHOW_CPP_STACKTRACES=0
export PYTHONUNBUFFERED=1

# 3. 清理GPU缓存
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU缓存已清理')"

# 4. 检查GPU状态
echo "=== GPU状态检查 ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# 5. 检查CPU核心数
echo "=== CPU核心数 ==="
nproc

# 6. 运行优化训练
echo "=== 开始优化训练 ==="
python3 train_tom500_optimized.py \
    --batch_size 8 \
    --num_workers 16 \
    --compile_model \
    --validation_interval 20 \
    --max_epochs 150

echo "=== 训练完成 ==="
