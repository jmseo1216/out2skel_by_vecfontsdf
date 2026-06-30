#!/usr/bin/env bash
# 학습 전 skeleton SVG별 target segment 수를 분석한다.
set -euo pipefail

skeleton_dir="/home/jmseo1216/deepfont/v2_all_dataset_aug300/aug_fnt"
k_segments=48
image_size=128
svg_size=50

python - <<'PY' "$skeleton_dir" "$k_segments" "$image_size" "$svg_size"
from pathlib import Path
import sys
from svg_utils import extract_line_segments_from_svg

skeleton_dir = Path(sys.argv[1])
k_segments = int(sys.argv[2])
image_size = int(sys.argv[3])
svg_size = float(sys.argv[4])
files = sorted(skeleton_dir.glob("*.svg"))
counts = []
for path in files:
    segs = extract_line_segments_from_svg(path, image_size=image_size, svg_size=svg_size)
    counts.append((path.name, len(segs)))

print(f"total files: {len(counts)}")
if not counts:
    raise SystemExit(0)
values = [c for _, c in counts]
print(f"min segment count: {min(values)}")
print(f"max segment count: {max(values)}")
print(f"mean segment count: {sum(values) / len(values):.2f}")
over = [(name, c) for name, c in counts if c > k_segments]
print(f"files with segment count > K({k_segments}): {len(over)}")
for name, c in over[:50]:
    print(f"  OVER_K {c:4d} {name}")
print("top 20 files with largest segment count:")
for name, c in sorted(counts, key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {c:4d} {name}")
PY
