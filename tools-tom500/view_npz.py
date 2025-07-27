import numpy as np

def explore_npz(file_path):
    """探索npz文件的结构"""
    data = np.load(file_path)
    
    print(f"文件: {file_path}")
    print("=" * 50)
    
    for key in data.keys():
        arr = data[key]
        print(f"键: {key}")
        print(f"  形状: {arr.shape}")
        print(f"  数据类型: {arr.dtype}")
        print(f"  维度: {arr.ndim}")
        
        # 如果数组不太大，显示一些样本值
        if arr.size < 100:
            print(f"  数据预览: {arr}")
        else:
            print(f"  数据范围: [{arr.min()}, {arr.max()}]")
            
            # 对于标签数据，显示更详细的信息
            if key.lower() in ['label', 'labels', 'mask', 'gt', 'segmentation']:
                unique_values = np.unique(arr)
                print(f"  唯一标签值: {unique_values}")
                print(f"  标签类别数: {len(unique_values)}")
                
                # 统计每个标签的像素数量
                for val in unique_values:
                    count = np.sum(arr == val)
                    percentage = (count / arr.size) * 100
                    print(f"    标签 {val}: {count} 像素 ({percentage:.2f}%)")
            
            elif arr.ndim == 1:
                unique_values = np.unique(arr)
                print(f"  唯一值: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")
        print()

# 详细分析包含目标的文件
print("详细分析包含目标的文件:")
explore_npz('./datasets/TOM500_tun/train_npz/case0000_slice004.npz')

# 扫描更多文件，统计标签分布
print("\n" + "="*60)
print("扫描所有文件的标签统计:")
import os
npz_dir = './datasets/TOM500_tun/train_npz/'
if os.path.exists(npz_dir):
    files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    all_labels = set()
    files_with_targets = []
    
    # 检查前20个文件
    for file in files[:20]:
        file_path = os.path.join(npz_dir, file)
        data = np.load(file_path)
        if 'label' in data:
            unique_labels = np.unique(data['label'])
            all_labels.update(unique_labels)
            
            # 记录包含目标的文件
            if len(unique_labels) > 1 or (len(unique_labels) == 1 and unique_labels[0] != 0):
                files_with_targets.append((file, unique_labels))
            
            print(f"{file}: 标签值 = {unique_labels}")
    
    print(f"\n数据集标签总结:")
    print(f"所有标签值: {sorted(all_labels)}")
    print(f"标签类别总数: {len(all_labels)}")
    print(f"\n包含目标的文件数: {len(files_with_targets)}")
    
    if files_with_targets:
        print("包含目标的文件详情:")
        for file, labels in files_with_targets:
            print(f"  {file}: {labels}")