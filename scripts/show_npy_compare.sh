#!/bin/bash

# 获取输入参数
INPUT_DIR=$1

# 检查输入参数
if [ -z "$INPUT_DIR" ]; then
  echo "Usage: $0 <INPUT_DIR>"
  exit 1
fi

# 扩展通配符路径
matches=(../output/${INPUT_DIR}_*/depth_npy)

# 检查是否匹配到路径
if [ ${#matches[@]} -eq 0 ]; then
  echo "Error: No matching directories found for ../output/${INPUT_DIR}_*/depth_npy"
  exit 1
fi

# 运行 Python 脚本，传递扩展后的路径
python show_npy.py --folder_paths "../test_cases/${INPUT_DIR}" "${matches[@]}"