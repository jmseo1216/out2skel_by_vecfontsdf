#!/usr/bin/env bash
# 학습된 checkpoint를 사용해 origin dataset의 특정 glyph를 시각화한다.
set -euo pipefail

outline_dir="/home/jmseo1216/deepfont/origin_dataset/arial_pro_ttf_svg"
skeleton_dir="/home/jmseo1216/deepfont/origin_dataset/arial_pro_fnt_svg"
output_dir="./runs/segmatch_aug300_k48"
checkpoint="$output_dir/checkpoints/best_val_model.pt"
save_dir="$output_dir/visualizations"

image_size=128
svg_size=50
codepoint=65
k_segments=48
sigma=2.0
num_target_points=1000
exist_threshold=0.5

mkdir -p "$save_dir"

python visualize_sample.py \
    --outline_dir "$outline_dir" \
    --skeleton_dir "$skeleton_dir" \
    --checkpoint "$checkpoint" \
    --save_dir "$save_dir" \
    --codepoint "$codepoint" \
    --image_size "$image_size" \
    --svg_size "$svg_size" \
    --k_segments "$k_segments" \
    --sigma "$sigma" \
    --num_target_points "$num_target_points" \
    --exist_threshold "$exist_threshold"

# augmented 파일을 직접 보고 싶으면 위 python 명령에 아래 인자를 추가하고 --codepoint를 제거하면 된다.
# --filename "arial_aug_0000_65.svg"
