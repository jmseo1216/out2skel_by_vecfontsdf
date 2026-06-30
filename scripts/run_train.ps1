# Windows PowerShell용 segment matching 학습 스크립트입니다.
$ErrorActionPreference = "Stop"

$outline_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_ttf_svg"
$skeleton_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_fnt_svg"
$output_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\vecfont_line_skeleton_runs\segmatch_arial_k48"

$image_size = 128
$svg_size = 50
$k_segments = 48
$batch_size = 8
$epochs = 100
$lr = 1e-4
$sigma = 2.0
$num_target_points = 1000
$pred_sample_points = 16

$val_ratio = 0.1
$test_ratio = 0.1
$checkpoint_every = 20
$val_every = 1
$log_every = 1

$w_match_segment = 5.0
$w_exist_bce = 1.0
$w_exist_count = 0.1
$w_total_length = 0.5
$w_render = 5.0
$w_pred_to_target = 0.5
$w_target_to_pred = 0.5
$render_fg_weight = 10.0

New-Item -ItemType Directory -Force -Path $output_dir | Out-Null

python train.py `
    --outline_dir $outline_dir `
    --skeleton_dir $skeleton_dir `
    --output_dir $output_dir `
    --image_size $image_size `
    --svg_size $svg_size `
    --k_segments $k_segments `
    --batch_size $batch_size `
    --epochs $epochs `
    --lr $lr `
    --sigma $sigma `
    --num_target_points $num_target_points `
    --pred_sample_points $pred_sample_points `
    --val_ratio $val_ratio `
    --test_ratio $test_ratio `
    --checkpoint_every $checkpoint_every `
    --val_every $val_every `
    --log_every $log_every `
    --w_match_segment $w_match_segment `
    --w_exist_bce $w_exist_bce `
    --w_exist_count $w_exist_count `
    --w_total_length $w_total_length `
    --w_render $w_render `
    --w_pred_to_target $w_pred_to_target `
    --w_target_to_pred $w_target_to_pred `
    --render_fg_weight $render_fg_weight
