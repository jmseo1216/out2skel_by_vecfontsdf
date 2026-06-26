#!/usr/bin/env bash
# 65.svg 하나로 전처리, forward, loss, 시각화를 확인하는 Linux/macOS용 예시 스크립트입니다.
set -euo pipefail

outline_dir="/path/to/pro_ttf_svg"
skeleton_dir="/path/to/pro_fnt_svg"
image_size=128
svg_size=50
codepoint=65
k_segments=32
sigma=2.0
num_target_points=4000
output_dir="./runs/arial_test"
save_dir="$output_dir/sanity_65"

mkdir -p "$save_dir"

python visualize_sample.py \
    --outline_dir "$outline_dir" \
    --skeleton_dir "$skeleton_dir" \
    --codepoint "$codepoint" \
    --image_size "$image_size" \
    --svg_size "$svg_size" \
    --k_segments "$k_segments" \
    --sigma "$sigma" \
    --num_target_points "$num_target_points" \
    --save_dir "$save_dir"
