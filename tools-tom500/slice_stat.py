import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def analyze_single_slice_depth(slice_depth, npz_dir, case_slices, max_cases=400):
    """分析单个切片深度的标签情况"""
    cases_with_more_than_2_labels = 0
    total_cases_at_depth = 0
    
    for case_id in range(max_cases):
        if case_id in case_slices and slice_depth in case_slices[case_id]:
            total_cases_at_depth += 1
            
            file_path = os.path.join(npz_dir, f"case{case_id:04d}_slice{slice_depth:03d}.npz")
            
            if os.path.exists(file_path):
                try:
                    data = np.load(file_path)
                    if 'label' in data:
                        unique_labels = np.unique(data['label'])
                        if len(unique_labels) > 2:
                            cases_with_more_than_2_labels += 1
                except Exception as e:
                    print(f"读取文件出错 {file_path}: {e}")
    
    return slice_depth, cases_with_more_than_2_labels, total_cases_at_depth

def analyze_slice_labels():
    """分析每个深度切片中标签数大于2的case数量（多线程版本）"""
    npz_dir = './datasets/TOM500_tun/train_npz/'
    
    if not os.path.exists(npz_dir):
        print(f"目录不存在: {npz_dir}")
        return
    
    # 统计每个slice深度的标签情况
    slice_stats = defaultdict(int)  # slice_depth -> count of cases with >2 labels
    slice_total = defaultdict(int)  # slice_depth -> total cases at this slice
    
    files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    print(f"找到 {len(files)} 个npz文件")
    
    # 按case和slice组织文件
    case_slices = defaultdict(list)
    for file in files:
        # 解析文件名: case0000_slice004.npz
        parts = file.replace('.npz', '').split('_')
        if len(parts) == 2:
            case_part = parts[0]  # case0000
            slice_part = parts[1]  # slice004
            
            try:
                case_id = int(case_part.replace('case', ''))
                slice_id = int(slice_part.replace('slice', ''))
                case_slices[case_id].append(slice_id)
            except ValueError:
                continue
    
    print(f"找到 {len(case_slices)} 个case")
    
    # 使用多线程分析每个深度的slice
    max_slice = 20  # 根据你的说明，有20个slice
    max_workers = min(64, max_slice)  # 限制线程数量
    
    print(f"使用 {max_workers} 个线程并行处理...")
    
    # 使用线程池执行器
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_slice = {
            executor.submit(analyze_single_slice_depth, slice_depth, npz_dir, case_slices): slice_depth
            for slice_depth in range(max_slice)
        }
        
        # 收集结果并显示进度
        with tqdm(total=max_slice, desc="分析切片") as pbar:
            for future in as_completed(future_to_slice):
                slice_depth, cases_with_more_than_2_labels, total_cases_at_depth = future.result()
                slice_stats[slice_depth] = cases_with_more_than_2_labels
                slice_total[slice_depth] = total_cases_at_depth
                pbar.update(1)
    
    # 输出结果
    print("\n" + "="*60)
    print("每个深度切片中标签数大于2的case数量统计:")
    print("="*60)
    print(f"{'Slice':<10} {'Cases>2Labels':<15} {'TotalCases':<12} {'Percentage':<10}")
    print("-"*60)
    
    for slice_depth in range(max_slice):
        total = slice_total[slice_depth]
        count = slice_stats[slice_depth]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"slice{slice_depth:03d}    {count:<15} {total:<12} {percentage:.1f}%")
    
    # 额外分析：找出标签最丰富的slice
    print(f"\n标签数大于2的case最多的切片:")
    sorted_slices = sorted(slice_stats.items(), key=lambda x: x[1], reverse=True)
    for i, (slice_depth, count) in enumerate(sorted_slices[:5]):
        total = slice_total[slice_depth]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  第{i+1}名: slice{slice_depth:03d} - {count}个case ({percentage:.1f}%)")

def analyze_single_case(case_id, npz_dir, max_slices=20):
    """分析单个case的标签分布"""
    case_labels_per_slice = []
    rich_slice_info = []
    
    for slice_depth in range(max_slices):
        file_path = os.path.join(npz_dir, f"case{case_id:04d}_slice{slice_depth:03d}.npz")
        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                if 'label' in data:
                    unique_labels = np.unique(data['label'])
                    case_labels_per_slice.append(len(unique_labels))
                    if len(unique_labels) > 2:
                        rich_slice_info.append((slice_depth, len(unique_labels), unique_labels))
                else:
                    case_labels_per_slice.append(0)
            except:
                case_labels_per_slice.append(0)
        else:
            case_labels_per_slice.append(0)
    
    return case_id, case_labels_per_slice, rich_slice_info

def analyze_specific_cases():
    """额外分析：查看几个具体case的标签分布（多线程版本）"""
    npz_dir = './datasets/TOM500_tun/train_npz/'
    
    print(f"\n" + "="*60)
    print("具体case示例分析:")
    print("="*60)
    
    # 使用多线程分析前3个case
    max_workers = 3
    cases_to_analyze = 3
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {
            executor.submit(analyze_single_case, case_id, npz_dir): case_id
            for case_id in range(cases_to_analyze)
        }
        
        # 收集并按case_id顺序显示结果
        results = {}
        for future in as_completed(future_to_case):
            case_id, case_labels_per_slice, rich_slice_info = future.result()
            results[case_id] = (case_labels_per_slice, rich_slice_info)
        
        # 按顺序输出结果
        for case_id in range(cases_to_analyze):
            if case_id in results:
                case_labels_per_slice, rich_slice_info = results[case_id]
                
                print(f"\nCase {case_id:04d} 的标签分布:")
                
                # 显示标签数大于2的slice
                for slice_depth, label_count, unique_labels in rich_slice_info:
                    print(f"  slice{slice_depth:03d}: {label_count} 个标签 {unique_labels}")
                
                # 显示该case的标签数量变化趋势
                max_labels = max(case_labels_per_slice) if case_labels_per_slice else 0
                if max_labels > 2:
                    print(f"  最大标签数: {max_labels}")
                    rich_slices = [i for i, count in enumerate(case_labels_per_slice) if count > 2]
                    print(f"  标签丰富的slice: {rich_slices}")

if __name__ == "__main__":
    analyze_slice_labels()
    analyze_specific_cases()