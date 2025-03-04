import numpy as np
import cv2
from PIL import Image
import os
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

def generate_random_mask(height, width, coverage=0.3):
    """
    生成随机连续形状的掩码
    Args:
        height: 图像高度
        width: 图像宽度
        coverage: 已知区域（值为0）的目标覆盖率
    Returns:
        numpy array: 掩码，其中1表示未知区域，0表示已知区域
    """
    mask = np.ones((height, width), dtype=np.float32)
    target_area = int(height * width * coverage)
    
    # 随机选择一个起始点
    start_y = np.random.randint(height // 4, 3 * height // 4)
    start_x = np.random.randint(width // 4, 3 * width // 4)
    
    # 使用队列进行区域生长
    queue = [(start_y, start_x)]
    mask[start_y, start_x] = 0
    current_area = 1
    
    # 定义8个方向
    directions = [
        (-1, 0),  # 上
        (1, 0),   # 下
        (0, -1),  # 左
        (0, 1),   # 右
        (-1, -1), # 左上
        (-1, 1),  # 右上
        (1, -1),  # 左下
        (1, 1)    # 右下
    ]
    
    while queue and current_area < target_area:
        # 随机选择队列中的一个点
        idx = np.random.randint(0, len(queue))
        y, x = queue[idx]
        queue[idx] = queue[-1]  # 将选中的点与最后一个点交换
        queue.pop()  # 移除最后一个点
        
        # 随机打乱方向
        np.random.shuffle(directions)
        
        # 向周围扩展
        for dy, dx in directions:
            new_y, new_x = y + dy, x + dx
            
            # 检查边界
            if 0 <= new_y < height and 0 <= new_x < width:
                # 如果相邻点是1，则将其变为0并加入队列
                if mask[new_y, new_x] == 1:
                    mask[new_y, new_x] = 0
                    queue.append((new_y, new_x))
                    current_area += 1
                    
                    if current_area >= target_area:
                        break
    
    return mask

def generate_depth_with_model(image_path, output_dir, model, processor):
    """
    Generate depth map and mask using DPT model
    
    Args:
        image_path (str): Path to the input RGB image
        output_dir (str): Directory to save the generated depth and mask
        model: DPT model for depth estimation
        processor: DPT image processor
    
    Returns:
        tuple: Paths to the generated depth map and mask files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and process input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 获取原始图像尺寸
    original_size = image.shape[:2][::-1]  # (width, height)
    
    # Process image for model input
    inputs = processor(images=image, return_tensors="pt")
    
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate depth map
    with torch.no_grad():
        outputs = model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
    
    # 将深度图调整回原始图像尺寸
    depth_map = cv2.resize(depth_map, original_size)
    
    # 反转深度值，使其从近到远对应从小到大
    depth_max = depth_map.max()
    depth_map = depth_max - depth_map
    
    # 生成随机连续形状的掩码
    mask = generate_random_mask(original_size[1], original_size[0], coverage=0.3)
    
    # Save depth map and mask
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    depth_path = os.path.join(output_dir, f"{base_name}_depth.npy")
    mask_path = os.path.join(output_dir, f"{base_name}_mask.npy")
    
    np.save(depth_path, depth_map)
    np.save(mask_path, mask)
    
    # Save visualizations - 反转可视化显示，使近处暗（值小），远处亮（值大）
    depth_vis = (((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())) * 255).astype(np.uint8)
    depth_vis = 255 - depth_vis  # 反转显示效果，保持近处看起来更亮
    mask_vis = (mask * 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_depth_vis.png"), depth_vis)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask_vis.png"), mask_vis)
    
    return depth_path, mask_path

def process_image_folder(input_folder, output_dir):
    """
    Process all images in a folder to generate depth maps and masks
    
    Args:
        input_folder (str): Folder containing input RGB images
        output_dir (str): Directory to save the generated files
    
    Returns:
        tuple: Lists of image paths, depth paths, and mask paths
    """
    # Initialize model
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    image_paths = []
    depth_paths = []
    mask_paths = []
    
    # 确保输入目录存在
    if not os.path.exists(input_folder):
        print(f"Warning: Input directory {input_folder} does not exist")
        return [], [], []
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            depth_path, mask_path = generate_depth_with_model(image_path, output_dir, model, processor)
            
            # 使用相对路径存储，与 test_cases 保持一致
            rel_image_path = os.path.relpath(image_path, os.path.dirname(input_folder))
            rel_depth_path = os.path.relpath(depth_path, os.path.dirname(output_dir))
            rel_mask_path = os.path.relpath(mask_path, os.path.dirname(output_dir))
            
            image_paths.append(rel_image_path)
            depth_paths.append(rel_depth_path)
            mask_paths.append(rel_mask_path)
    
    return image_paths, depth_paths, mask_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate random depth maps and masks for images")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing RGB images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for depth maps and masks")
    
    args = parser.parse_args()
    
    image_paths, depth_paths, mask_paths = process_image_folder(args.input_dir, args.output_dir)
    
    print(f"Processed {len(image_paths)} images")
    print("Generated files can be used with infer.py using the following parameters:")
    print(f"--input_image_paths {' '.join(image_paths)}")
    print(f"--known_depth_paths {' '.join(depth_paths)}")
    print(f"--masks_paths {' '.join(mask_paths)}")
