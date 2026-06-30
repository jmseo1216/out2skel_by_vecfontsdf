# Segment Matching Outline-to-Skeleton 실행 가이드

이 파이프라인은 outline SVG를 128x128 이미지로 렌더링하고, target skeleton SVG에서 실제 직선 segment 개수를 샘플별로 추출한 뒤, Hungarian matching으로 `K`개 예측 segment와 target segment를 학습합니다. 문자별 고정 segment 개수 규칙은 사용하지 않습니다.

## 설치

```bash
pip install -r requirements.txt
```

## 좌표계

- 원본 SVG: `svg_size x svg_size`, 기본 `50 x 50`, y-up
- 모델/이미지: `image_size x image_size`, 기본 `128 x 128`, y-down

변환식:

```text
x_img = x_svg / svg_size * image_size
y_img = (svg_size - y_svg) / svg_size * image_size
```

모델 출력 segment 좌표는 `[0, image_size]` 범위의 이미지 좌표입니다.

## Dataset pairing 규칙

`FontSkeletonDataset`은 `outline_dir`의 모든 `*.svg`를 scan하고, `skeleton_dir`에 같은 파일 이름이 있는 경우만 paired sample로 사용합니다.

지원 파일 이름 예시:

- `33.svg`
- `65.svg`
- `arial_aug_0000_33.svg`
- `arial_aug_0000_65.svg`
- `simplex_aug_0123_126.svg`

`codepoint`는 파일 이름의 마지막 숫자에서 추출합니다. 예를 들어 `arial_aug_0000_65.svg`의 codepoint는 `65`입니다.

## Segment count 분석

학습 전에 `K=48`이 충분한지 확인합니다.

```bash
bash scripts/analyze_segment_counts.sh
```

기본 설정:

```bash
skeleton_dir="/home/jmseo1216/deepfont/v2_all_dataset_aug300/aug_fnt"
k_segments=48
```

출력 항목:

- total files
- min segment count
- max segment count
- mean segment count
- files with segment count > K
- top 20 files with largest segment count

## 학습 실행

```bash
bash scripts/run_train.sh
```

기본 경로:

```bash
outline_dir="/home/jmseo1216/deepfont/v2_all_dataset_aug300/aug_ttf"
skeleton_dir="/home/jmseo1216/deepfont/v2_all_dataset_aug300/aug_fnt"
output_dir="./runs/segmatch_aug300_k48"
```

기본 학습 설정:

```bash
k_segments=48
batch_size=8
epochs=100
num_target_points=1000
val_ratio=0.1
test_ratio=0.1
checkpoint_every=20
val_every=1
```

저장 파일:

- `$output_dir/checkpoints/latest.pt`
- `$output_dir/checkpoints/best_val_model.pt`
- `$output_dir/checkpoints/font_skeleton_model.pt`
- `$output_dir/checkpoints/epoch_XXXX.pt`
- `$output_dir/logs/losses.csv`

CSV 컬럼:

```text
epoch,split,total,match_segment,exist_bce,exist_count,total_length,render,pred_to_target,target_to_pred
```

## 시각화 실행

```bash
bash scripts/run_visualize.sh
```

기본적으로 origin Arial SVG에서 `codepoint=65`를 시각화하고, checkpoint는 다음 파일을 사용합니다.

```bash
checkpoint="./runs/segmatch_aug300_k48/checkpoints/best_val_model.pt"
```

augmented 파일을 직접 보고 싶으면 다음처럼 `--filename`을 사용합니다.

```bash
python visualize_sample.py \
  --outline_dir /home/jmseo1216/deepfont/v2_all_dataset_aug300/aug_ttf \
  --skeleton_dir /home/jmseo1216/deepfont/v2_all_dataset_aug300/aug_fnt \
  --checkpoint ./runs/segmatch_aug300_k48/checkpoints/best_val_model.pt \
  --save_dir ./runs/segmatch_aug300_k48/visualizations \
  --filename arial_aug_0000_65.svg \
  --k_segments 48 \
  --exist_threshold 0.5
```

시각화는 다음 내용을 저장합니다.

1. outline image
2. target skeleton heatmap
3. target segments overlay
4. predicted active segments overlay
5. target + predicted overlay
6. Hungarian matched pairs

## Loss 설명

### `L_match_segment`

Hungarian matching으로 선택된 예측 segment와 target segment 사이의 방향 불변 endpoint L1 loss입니다. segment 방향이 뒤집혀도 같은 선분으로 취급합니다. 좌표 차이는 `image_size`로 정규화합니다.

### `L_exist_bce`

Hungarian matching된 예측 segment는 exist target 1, 나머지는 0으로 두고 `BCEWithLogitsLoss`를 계산합니다. 단순 glyph에서 불필요한 많은 segment가 켜지는 문제를 줄이기 위한 핵심 loss입니다.

### `L_exist_count`

샘플별 target segment 수와 `sigmoid(logits).sum()`으로 계산한 예측 활성 segment 수가 가까워지도록 합니다. target count는 skeleton SVG에서 직접 계산합니다.

### `L_total_length`

`sum(exist_prob * pred_length)`와 target skeleton 총 길이가 가까워지도록 합니다. 많은 긴 선분을 그리는 실패 모드를 줄입니다.

### `L_render`

예측 segment를 soft heatmap으로 렌더링한 뒤 target heatmap과 foreground-weighted MSE를 계산합니다. skeleton 픽셀 주변을 더 중요하게 보기 위해 `1 + render_fg_weight * target_heatmap` 가중치를 사용합니다.

### `L_pred_to_target`, `L_target_to_pred`

기존 distance 기반 Chamfer 방향 loss를 보조 loss로 유지합니다. 두 loss 모두 `image_size`로 정규화됩니다.

## 중요한 sanity check

아래 path는 target segment 3개로 추출되어야 합니다.

```svg
M 10.2 12.5 L 20.6 41.9 M 22.5 41.9 L 32.8 12.5 M 27.3 22.4 L 15.7 22.4
```

또한 각 샘플에서 다음이 만족되어야 합니다.

- `target_num_segments == target_segment_mask.sum()`
- `target_segments.shape == [K, 4]`
- `target_segment_mask.shape == [K]`
- `pred_exist_logits.shape == [B, K]`
- `pred_segments.shape == [B, K, 4]`
- `pred_segments`는 `[0, image_size]` 안에 있어야 합니다.
- target segment 수가 0이거나 K보다 크면 명확한 에러를 발생시킵니다.
