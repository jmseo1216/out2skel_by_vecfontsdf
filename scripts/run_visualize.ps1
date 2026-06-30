# Windows PowerShell용 checkpoint 시각화 스크립트입니다.
$ErrorActionPreference = "Stop"

$outline_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_ttf_svg"
$skeleton_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_fnt_svg"
$output_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\vecfont_line_skeleton_runs\segmatch_arial_k48"
$checkpoint = Join-Path $output_dir "checkpoints\best_val_model.pt"
$save_dir = Join-Path $output_dir "visualizations"

$image_size = 128
$svg_size = 50
$codepoint = 65
$k_segments = 48
$sigma = 2.0
$num_target_points = 1000
$exist_threshold = 0.5

New-Item -ItemType Directory -Force -Path $save_dir | Out-Null

python visualize_sample.py `
    --outline_dir $outline_dir `
    --skeleton_dir $skeleton_dir `
    --checkpoint $checkpoint `
    --save_dir $save_dir `
    --codepoint $codepoint `
    --image_size $image_size `
    --svg_size $svg_size `
    --k_segments $k_segments `
    --sigma $sigma `
    --num_target_points $num_target_points `
    --exist_threshold $exist_threshold

# augmented 파일을 직접 보고 싶으면 위 python 명령에 아래 인자를 추가하고 --codepoint를 제거하면 됩니다.
# --filename "arial_aug_0000_65.svg"
