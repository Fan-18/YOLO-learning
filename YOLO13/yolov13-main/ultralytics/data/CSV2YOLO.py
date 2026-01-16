
from pathlib import Path
from PIL import Image
import os

def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    cx = (box[0] + box[2] / 2.0) * dw
    cy = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return cx, cy, w, h

root = Path("ultralytics/data/VisDrone")
for d in ("VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"):
    ann_dir = root / d / "annotations"
    img_dir = root / d / "images"
    lbl_dir = root / d / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for f in ann_dir.glob("*.txt"):
        img_path = img_dir / f.name
        if not img_path.with_suffix(".jpg").exists():
            # 如果是 .jpg 以外的扩展名，请调整
            continue
        size = Image.open(img_path.with_suffix(".jpg")).size
        lines = []
        with open(f, "r") as fh:
            for row in [x.split(",") for x in fh.read().strip().splitlines()]:
                # VisDrone: row[4] == 0 表示忽略区域（示例中跳过）
                if row[4] == '0':
                    continue
                cls = int(row[5]) - 1  # VisDrone 类编号从 1 开始，减 1 变为 0-based
                box = tuple(map(int, row[:4]))
                cx, cy, w, h = convert_box(size, box)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        out_path = lbl_dir / f.name
        with open(out_path, "w") as out:
            out.writelines(lines)
print('转换完成')
