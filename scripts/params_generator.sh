#!/bin/bash

FOLDER_PATH=$1

# 检查输入参数
if [ -z "$FOLDER_PATH" ]; then
  echo "Usage: $0 <FOLDER_PATH>"
  exit 1
fi

export FOLDER_PATH  # 导出变量

echo "FOLDER_PATH is: ${FOLDER_PATH}"
# 设置输入和输出目录
INPUT_DIR="../test_cases/${FOLDER_PATH}"
OUTPUT_DIR="../test_cases/${FOLDER_PATH}"

# 创建必要的目录
mkdir -p $INPUT_DIR
mkdir -p $OUTPUT_DIR

# 运行 Python 脚本
python params_generator.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"

python show_npy.py --folder_paths "../test_cases/${FOLDER_PATH}"
