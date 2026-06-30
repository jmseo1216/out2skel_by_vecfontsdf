# Windows PowerShell에서 65.svg 하나로 segment matching sanity 시각화를 실행합니다.
$ErrorActionPreference = "Stop"

$outline_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_ttf_svg"
$skeleton_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_fnt_svg"
$image_size = 128
$svg_size = 50
$codepoint = 65
$k_segments = 48
$sigma = 2.0
$num_target_points = 1000
$output_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\vecfont_line_skeleton_runs\segmatch_arial_k48"
$save_dir = Join-Path $output_dir "sanity_65"
$exist_threshold = 0.5

New-Item -ItemType Directory -Force -Path $save_dir | Out-Null

python visualize_sample.py `
    --outline_dir $outline_dir `
    --skeleton_dir $skeleton_dir `
    --codepoint $codepoint `
    --image_size $image_size `
    --svg_size $svg_size `
    --k_segments $k_segments `
    --sigma $sigma `
    --num_target_points $num_target_points `
    --exist_threshold $exist_threshold `
    --save_dir $save_dir
