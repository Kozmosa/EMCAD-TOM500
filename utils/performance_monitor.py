import time
import torch
import threading
from datetime import datetime

class TrainingMonitor:
    def __init__(self, log_interval=60):  # 每60秒记录一次
        self.log_interval = log_interval
        self.start_time = time.time()
        self.iteration_times = []
        self.gpu_utils = []
        self.memory_usage = []
        self.running = False
        
    def start_monitoring(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        while self.running:
            try:
                # 监控GPU使用率
                if torch.cuda.is_available():
                    gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                    memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    
                    self.gpu_utils.append(gpu_util)
                    self.memory_usage.append((memory_used, memory_total))
                    
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"[{current_time}] GPU Mem: {memory_used:.1f}/{memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")
                
            except Exception as e:
                print(f"监控错误: {e}")
                
            time.sleep(self.log_interval)
    
    def log_iteration(self, iteration_time):
        self.iteration_times.append(iteration_time)
        
        # 计算平均速度
        if len(self.iteration_times) >= 10:
            avg_time = sum(self.iteration_times[-10:]) / 10
            iterations_per_sec = 1.0 / avg_time
            print(f"Average iteration time: {avg_time:.3f}s | Iterations/sec: {iterations_per_sec:.2f}")
    
    def get_summary(self):
        total_time = time.time() - self.start_time
        avg_iteration_time = sum(self.iteration_times) / len(self.iteration_times) if self.iteration_times else 0
        
        summary = f"""
=== 训练性能总结 ===
总训练时间: {total_time/3600:.2f} 小时
平均每次迭代时间: {avg_iteration_time:.3f} 秒
总迭代次数: {len(self.iteration_times)}
平均GPU使用率: {sum(self.gpu_utils)/len(self.gpu_utils):.1f}% (如果有数据)
"""
        return summary

# 在trainer.py中集成使用的辅助函数
def optimize_cuda_settings():
    """优化CUDA设置"""
    if torch.cuda.is_available():
        # 启用Tensor Core优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 设置内存分配策略
        torch.cuda.empty_cache()
        
        print("CUDA优化设置已启用")
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def check_data_loading_speed(dataloader, max_batches=10):
    """检查数据加载速度"""
    print("检查数据加载速度...")
    times = []
    
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
            
        start_time = time.time()
        # 模拟数据传输到GPU
        if torch.cuda.is_available():
            batch['image'] = batch['image'].cuda(non_blocking=True)
            batch['label'] = batch['label'].cuda(non_blocking=True)
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"平均数据加载时间: {avg_time:.3f} 秒/批次")
    print(f"数据加载速度: {len(times)/sum(times):.2f} 批次/秒")
    
    return avg_time

if __name__ == "__main__":
    print("性能监控工具已加载")
    print("使用方法:")
    print("1. monitor = TrainingMonitor()")
    print("2. monitor.start_monitoring()")
    print("3. 在训练循环中调用 monitor.log_iteration(iteration_time)")
    print("4. 训练结束后调用 monitor.stop_monitoring()")
    print("5. 查看总结: monitor.get_summary()")
