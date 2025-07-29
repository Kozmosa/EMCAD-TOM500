# 训练速度优化指南

## 已实施的优化措施

### 1. 混合精度训练 (Automatic Mixed Precision)
- **效果**: 可提升 30-50% 的训练速度
- **内存节省**: 减少约 50% 的GPU内存占用
- **实现**: 在 `trainer.py` 中添加了 `autocast()` 和 `GradScaler`

### 2. 数据加载优化
- **增加预取**: `prefetch_factor=2`
- **持久化工作进程**: `persistent_workers=True`
- **异步数据传输**: `non_blocking=True`
- **效果**: 减少数据加载等待时间

### 3. 模型编译优化
- **PyTorch 2.0+ 编译**: `torch.compile(model)`
- **效果**: 可提升 10-20% 的推理速度

### 4. 验证频率优化
- **从每10个epoch改为每20个epoch验证**
- **效果**: 减少验证时间开销

### 5. CUDA 优化设置
- **启用 TensorCore**: `allow_tf32=True`
- **CUDNN benchmark**: `cudnn.benchmark=True`
- **内存优化**: 设置内存分配策略

## 使用方法

### 方法1: 使用优化脚本
```bash
# 给脚本执行权限
chmod +x optimize_training.sh

# 运行优化训练
./optimize_training.sh
```

### 方法2: 直接运行优化版训练脚本
```bash
python train_tom500_optimized.py \
    --batch_size 8 \
    --num_workers 16 \
    --compile_model \
    --validation_interval 20
```

## 进一步优化建议

### 1. 批次大小调整
- **当前**: 6 → **建议**: 8-12 (根据GPU内存)
- **计算方法**: GPU内存(GB) ÷ 2 = 大致批次大小

### 2. 数据预处理优化
- 使用 `dataset_tom500_optimized.py` 中的优化版数据集
- 启用数据缓存 (适用于小数据集)

### 3. 模型相关优化
```python
# 在模型初始化后添加
model = torch.compile(model, mode="reduce-overhead")  # 更激进的优化
```

### 4. 环境变量优化
```bash
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_CUDNN_V8_API_ENABLED=1
```

## 性能监控

使用 `performance_monitor.py` 监控训练性能：

```python
from utils.performance_monitor import TrainingMonitor

# 在训练开始前
monitor = TrainingMonitor()
monitor.start_monitoring()

# 在训练循环中记录迭代时间
start_time = time.time()
# ... 训练代码 ...
iteration_time = time.time() - start_time
monitor.log_iteration(iteration_time)

# 训练结束后
monitor.stop_monitoring()
print(monitor.get_summary())
```

## 预期性能提升

根据优化措施，预期可获得的性能提升：

1. **混合精度训练**: 30-50% 速度提升
2. **数据加载优化**: 10-20% 速度提升  
3. **模型编译**: 10-20% 速度提升
4. **批次大小优化**: 5-15% GPU利用率提升
5. **验证频率调整**: 5-10% 总训练时间节省

**总体预期**: 可获得 **50-80%** 的训练速度提升

## 注意事项

1. **混合精度**: 可能影响数值精度，建议监控loss收敛情况
2. **批次大小**: 增大批次可能需要调整学习率
3. **模型编译**: 首次编译需要额外时间，后续迭代才会加速
4. **内存使用**: 优化后可能使用更多GPU内存

## 问题排查

如果遇到问题，请检查：

1. **PyTorch版本**: 建议使用 PyTorch 2.0+
2. **CUDA版本**: 确保与PyTorch兼容
3. **GPU内存**: 如果OOM，请减小批次大小
4. **数据路径**: 确保数据路径正确

## 联系和支持

如有问题，可以检查：
- GPU使用率是否达到 80%+
- 数据加载是否为瓶颈
- 内存使用是否合理
