#!/usr/bin/env bash
# paired SVG 전체 데이터셋으로 학습을 시작하는 Linux/macOS용 예시 스크립트입니다.
set -euo pipefail

outline_dir="/path/to/pro_ttf_svg"
skeleton_dir="/path/to/pro_fnt_svg"
output_dir="./runs/arial_test"

image_size=128
svg_size=50
k_segments=32
batch_size=8
epochs=200
lr=1e-4
sigma=2.0
num_target_points=4000
pred_sample_points=16
length_min=2.0

w_render=1.0
w_pred_to_target=1.0
w_target_to_pred=1.0
w_exist=0.001
w_length=0.1

mkdir -p "$output_dir"

python train.py \
    --outline_dir "$outline_dir" \
    --skeleton_dir "$skeleton_dir" \
    --output_dir "$output_dir" \
    --image_size "$image_size" \
    --svg_size "$svg_size" \
    --k_segments "$k_segments" \
    --batch_size "$batch_size" \
    --epochs "$epochs" \
    --lr "$lr" \
    --sigma "$sigma" \
    --num_target_points "$num_target_points" \
    --pred_sample_points "$pred_sample_points" \
    --length_min "$length_min" \
    --w_render "$w_render" \
    --w_pred_to_target "$w_pred_to_target" \
    --w_target_to_pred "$w_target_to_pred" \
    --w_exist "$w_exist" \
    --w_length "$w_length"
