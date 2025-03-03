import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def visualize_npy_folders(folder_paths):
    """
    合并可视化多个文件夹中的所有.npy文件
    
    Args:
        folder_paths (list of str): 包含.npy文件的文件夹路径列表
    """
    # 收集所有.npy文件路径及其所属文件夹
    all_files = []
    for folder in folder_paths:
        if not os.path.exists(folder):
            print(f"警告: 目录 {folder} 不存在")
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        if not files:
            print(f"在 {folder} 中没有找到.npy文件")
            continue
        all_files.extend(files)
    
    if not all_files:
        print("所有指定文件夹中均未找到.npy文件")
        return
    
    # 计算子图布局
    n_files = len(all_files)
    n_cols = min(3, n_files)  # 最多3列
    n_rows = (n_files + n_cols - 1) // n_cols  # 向上取整
    
    # 创建合并后的图形
    plt.figure(figsize=(5*n_cols, 4*n_rows))
    plt.suptitle("Multiple Folder Visualization", fontsize=16, y=1.02)
    
    # 遍历所有文件并显示
    for idx, file_path in enumerate(all_files, 1):
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        
        # 加载数据
        data = np.load(file_path)
        
        # 创建子图
        plt.subplot(n_rows, n_cols, idx)
        plt.imshow(data, cmap='viridis')
        plt.colorbar()
        plt.title(f"{file_path}/\n{file_name}\nShape: {data.shape}")
        plt.axis('off')
    
    # 调整布局
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并可视化多个文件夹中的.npy文件')
    parser.add_argument('--folder_paths', '-f', type=str, nargs='+', required=True,
                       help='包含.npy文件的文件夹路径列表（支持通配符）')
    args = parser.parse_args()
    
    visualize_npy_folders(args.folder_paths)