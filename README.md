# out2skel_by_vecfontsdf

# VecFontSDF-inspired Skeleton Pipeline 실행 가이드

이 문서는 `outline SVG → laser skeleton line segments` 첫 버전 파이프라인을 Windows PowerShell에서 바로 실행하기 위한 안내서입니다. 코드 주석과 실행 스크립트는 한국어 설명을 포함하며, 함수명과 인자명은 Python 관례에 맞춰 영어로 유지했습니다.

## 1. 설치 방법

프로젝트 루트에서 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

주요 의존성은 `torch`, `numpy`, `Pillow`, `scipy`, `matplotlib`입니다.

## 2. 데이터 경로

PowerShell 스크립트의 기본 경로는 다음 Windows 경로로 설정되어 있습니다.

```powershell
$outline_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_ttf_svg"
$skeleton_dir = "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_fnt_svg"
```

각 디렉터리에는 `33.svg`부터 `126.svg`까지의 paired SVG가 있어야 합니다.

## 3. 좌표계 설명

원본 SVG와 CNN 이미지의 좌표계가 다르므로 전처리에서 아래 변환을 항상 적용합니다.

- SVG 좌표계: `50 x 50`, y-up
  - `x`는 왼쪽에서 오른쪽으로 증가합니다.
  - `y`는 아래에서 위로 증가합니다.
- 이미지 좌표계: `128 x 128`, y-down
  - `x`는 왼쪽에서 오른쪽으로 증가합니다.
  - `y`는 위에서 아래로 증가합니다.

변환식은 다음과 같습니다.

```text
x_img = x_svg / 50 * 128
y_img = (50 - y_svg) / 50 * 128
```

코드에서는 `image_size`와 `svg_size`를 CLI 인자로 받을 수 있으므로, 기본값은 `128`과 `50`이지만 필요하면 조정할 수 있습니다.

## 4. 65.svg sanity test 실행

아래 명령은 `65.svg` 하나를 로드하여 전처리 결과, 랜덤 초기화 모델 forward 결과, loss 값을 확인합니다.

```powershell
.\scripts\run_sanity_65.ps1
```

확인 항목은 다음과 같습니다.

- outline image
- target skeleton raster
- target distance transform
- target heatmap
- target sample points
- predicted segments
- predicted heatmap
- loss 값 출력

기본 출력 이미지는 다음 디렉터리에 저장됩니다.

```text
D:\deeplearning\marker\font\font_preprocessing\data_pipeline\vecfont_line_skeleton_runs\arial_test\sanity_65
```

## 5. 학습 실행

아래 명령은 paired SVG 전체 데이터셋으로 학습을 시작합니다.

```powershell
.\scripts\run_train.ps1
```

학습 스크립트는 output directory가 없으면 자동으로 생성합니다. 기본 output directory는 다음과 같습니다.

```text
D:\deeplearning\marker\font\font_preprocessing\data_pipeline\vecfont_line_skeleton_runs\arial_test
```

학습 중 epoch별 loss dictionary 평균을 출력하고, CSV log를 저장합니다.

```text
$output_dir\logs\train_log.csv
```

학습 완료 후 checkpoint는 기본적으로 아래 위치에 저장됩니다.

```text
$output_dir\checkpoints\font_skeleton_model.pt
```

## 6. checkpoint 시각화 실행

학습된 checkpoint를 불러와 특정 codepoint의 예측 선분과 heatmap을 시각화합니다.

```powershell
.\scripts\run_visualize.ps1
```

기본적으로 `$output_dir\checkpoints\font_skeleton_model.pt`를 로드하고, PNG 결과를 아래 디렉터리에 저장합니다.

```text
$output_dir\visualizations
```

`run_visualize.ps1`에서 다음 변수를 바꾸면 다른 checkpoint나 codepoint를 확인할 수 있습니다.

```powershell
$checkpoint = Join-Path $output_dir "checkpoints\font_skeleton_model.pt"
$codepoint = 65
$save_dir = Join-Path $output_dir "visualizations"
```

## 7. Linux/macOS 예시 스크립트

Linux/macOS용 placeholder 스크립트도 제공합니다. 사용 전에 `/path/to/pro_ttf_svg`, `/path/to/pro_fnt_svg`를 실제 데이터 경로로 수정하세요.

```bash
./scripts/run_sanity_65.sh
./scripts/run_train.sh
./scripts/run_visualize.sh
```

## 8. Loss 의미

### `L_pred_to_target`

예측된 선분 위에서 균일하게 sample point를 만들고, 각 점이 target skeleton distance transform에서 얼마나 멀리 떨어져 있는지 `grid_sample`로 bilinear sampling합니다. 예측 선분이 target skeleton 근처에 놓이도록 유도합니다.

### `L_target_to_pred`

target skeleton vector polyline에서 arc length 기준으로 샘플링한 점들이 예측 선분 중 하나에 의해 잘 덮이도록 합니다. 각 target point에서 모든 predicted segment까지의 point-to-segment distance를 계산하고, 활성 선분 중 최소 거리를 사용합니다.

### `L_render`

예측 선분을 soft rasterizer로 heatmap으로 변환한 뒤, target skeleton heatmap과 MSE로 비교합니다. 전체적인 raster 모양이 target skeleton과 비슷해지도록 돕습니다.

### `L_exist_sparse`

`sigmoid(pred_exist_logits)`의 평균을 줄여 불필요한 선분 사용을 억제합니다. 첫 버전에서는 target segment count를 직접 쓰지 않습니다.

### `L_length`

활성 선분 길이가 `length_min`보다 짧으면 penalty를 줍니다. 모든 선분이 점으로 collapse되는 것을 완화합니다.

## 9. 직접 Python 명령으로 실행하는 예시

PowerShell 스크립트를 쓰지 않고 직접 실행하려면 다음처럼 실행할 수 있습니다.

```powershell
python visualize_sample.py `
    --outline_dir "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_ttf_svg" `
    --skeleton_dir "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_fnt_svg" `
    --codepoint 65 `
    --image_size 128 `
    --svg_size 50 `
    --k_segments 32 `
    --sigma 2.0 `
    --num_target_points 4000
```

```powershell
python train.py `
    --outline_dir "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_ttf_svg" `
    --skeleton_dir "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\v2_dataset\arial\pro_fnt_svg" `
    --output_dir "D:\deeplearning\marker\font\font_preprocessing\data_pipeline\vecfont_line_skeleton_runs\arial_test" `
    --image_size 128 `
    --svg_size 50 `
    --k_segments 32 `
    --batch_size 8 `
    --epochs 200 `
    --lr 1e-4 `
    --sigma 2.0 `
    --num_target_points 4000 `
    --pred_sample_points 16 `
    --length_min 2.0 `
    --w_render 1.0 `
    --w_pred_to_target 1.0 `
    --w_target_to_pred 1.0 `
    --w_exist 0.001 `
    --w_length 0.1
```
