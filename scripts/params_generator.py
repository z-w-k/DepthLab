import numpy as np
import cv2
from PIL import Image
import os

def generate_random_depth_and_mask(image_path, output_dir):
    """
    Generate random depth map and mask for a given image
    
    Args:
        image_path (str): Path to the input RGB image
        output_dir (str): Directory to save the generated depth and mask
    
    Returns:
        tuple: Paths to the generated depth map and mask files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and process input image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    height, width = img.shape[:2]
    
    # Generate random depth map (values between 0 and 1)
    depth_map = np.random.uniform(0, 1, (height, width)).astype(np.float32)
    
    # Apply Gaussian blur to make depth map more natural
    depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
    
    # Generate mask with 30% random zeros
    mask = np.ones((height, width), dtype=np.float32)
    num_zeros = int(0.3 * height * width)
    zero_indices = np.random.choice(height * width, num_zeros, replace=False)
    mask.flat[zero_indices] = 0
    
    # Save depth map and mask
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    depth_path = os.path.join(output_dir, f"{base_name}_depth.npy")
    mask_path = os.path.join(output_dir, f"{base_name}_mask.npy")
    
    np.save(depth_path, depth_map)
    np.save(mask_path, mask)
    
    # Also save visualizations
    depth_vis = (depth_map * 255).astype(np.uint8)
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
    image_paths = []
    depth_paths = []
    mask_paths = []
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            depth_path, mask_path = generate_random_depth_and_mask(image_path, output_dir)
            
            image_paths.append(image_path)
            depth_paths.append(depth_path)
            mask_paths.append(mask_path)
    
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
