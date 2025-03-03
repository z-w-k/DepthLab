#!/usr/bin/env bash
set -e
set -x

# Define paths
BASE_CHECKPOINT_PATH='./checkpoints'
PRETRAINED_MODEL_NAME_OR_PATH="$BASE_CHECKPOINT_PATH/marigold-depth-v1-0"
IMAGE_ENCODER_PATH="$BASE_CHECKPOINT_PATH/CLIP-ViT-H-14-laion2B-s32B-b79K"
DENOSING_UNET_PATH="$BASE_CHECKPOINT_PATH/DepthLab/denoising_unet.pth"
REFERENCE_UNET_PATH="$BASE_CHECKPOINT_PATH/DepthLab/reference_unet.pth"
MAPPING_PATH="$BASE_CHECKPOINT_PATH/DepthLab/mapping_layer.pth"

# 默认值
MODE="normal"
FOLDER_PATH=""
TYPE="JPG"

# 解析命名参数
while getopts "m:f:t:" opt; do
  case $opt in
    m) MODE="$OPTARG" ;;
    f) FOLDER_PATH="$OPTARG" ;;
    t) TYPE="$OPTARG" ;;
    *) echo "Usage: $0 -m <MODE> -f <FOLDER_PATH> -t <FILE_TYPE>"; exit 1 ;;
  esac
done

# 检查必需参数
if  [ -z "$FOLDER_PATH" ] || [ -z "$TYPE" ]; then
  echo "Usage: $0 -f <FOLDER_PATH> -t <FILE_TYPE>"
  exit 1
fi

# 继续处理
echo "MODE is : ${MODE}"
echo "FOLDER_PATH is : ${FOLDER_PATH}"
echo "FILE_TYPE is : ${TYPE}"

# Define input and output paths
INPUT_IMAGE_PATH="test_cases/${FOLDER_PATH}/RGB.${TYPE}"
# 检查文件是否存在
if ls $INPUT_IMAGE_PATH 1> /dev/null 2>&1; then
    echo "文件存在: $INPUT_IMAGE_PATH"
else
    echo "文件不存在: $INPUT_IMAGE_PATH"
fi

KNOWN_DEPTH_PATH="test_cases/${FOLDER_PATH}/RGB_depth.npy"
MASK_PATH="test_cases/${FOLDER_PATH}/RGB_mask.npy"
OUTPUT_DIR="output/${FOLDER_PATH}_${MODE}"

# 动态生成参数
EXTRA_ARGS=""
if [[ "$MODE" == *"blend"* ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --blend"
fi
if [[ "$MODE" == *"refine"* ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --refine"
fi

export CUDA_VISIBLE_DEVICES=0
cd ..
python infer.py  \
    --seed 1234 \
    --denoise_steps 40 \
    --processing_res 512 \
    --normalize_scale 1 \
    --strength 0.8 \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_NAME_OR_PATH \
    --image_encoder_path $IMAGE_ENCODER_PATH \
    --denoising_unet_path $DENOSING_UNET_PATH \
    --reference_unet_path $REFERENCE_UNET_PATH \
    --mapping_path $MAPPING_PATH \
    --output_dir $OUTPUT_DIR \
    --input_image_paths $INPUT_IMAGE_PATH \
    --known_depth_paths $KNOWN_DEPTH_PATH \
    --masks_paths $MASK_PATH \
    $EXTRA_ARGS

cd scripts
sh show_npy_compare.sh "$FOLDER_PATH"