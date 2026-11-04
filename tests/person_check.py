"""Person check for object removal: is a person standing inside the masked region?

Runs the YOLOv8n-seg person class (COCO class 0) on the image and reports each person whose box is covered by the mask
for at least `min_cover` of its area. A removal candidate passes when no person reaches that coverage. People outside
the mask, whose boxes only touch its edge, stay below the threshold.

The weights are the BN-folded YOLOv8n-seg safetensors (format hartsyinference-yolo-folded-v1). They load into the
ultralytics architecture with strict=True, so a key or shape mismatch fails loudly.

Usage: PERSON_CHECK_WEIGHTS=<yolov8n-seg-folded.safetensors> person_check.py <image.png> <mask.png> [min_cover]
Prints one JSON line: {"fail": bool, "max_cover": float, "min_cover": float, "persons": [...]}.
Needs torch, ultralytics, safetensors, opencv-python and numpy.
"""
import json
import os
import sys

os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402

PERSON_CLASS = 0
CONF = 0.25
IOU = 0.45
SIZE = 640


def load_net(weights):
    net = YOLO("yolov8n-seg.yaml").model
    net.fuse(verbose=False)
    net.load_state_dict(load_file(weights), strict=True)
    return net.eval()


def letterbox(img):
    h, w = img.shape[:2]
    scale = min(SIZE / h, SIZE / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (SIZE - nw) // 2, (SIZE - nh) // 2
    padded = cv2.copyMakeBorder(resized, top, SIZE - nh - top, left, SIZE - nw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, left, top


def person_boxes(net, img):
    h, w = img.shape[:2]
    padded, scale, left, top = letterbox(img)
    x = torch.from_numpy(padded[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        out = net(x)
    pred = out[0] if isinstance(out, (list, tuple)) else out
    rows = non_max_suppression(pred, conf_thres=CONF, iou_thres=IOU, classes=[PERSON_CLASS], max_det=100, nc=80)[0]
    boxes = []
    for row in rows.tolist():
        x1, y1, x2, y2, conf = row[:5]
        x1 = min(max((x1 - left) / scale, 0.0), w - 1.0)
        x2 = min(max((x2 - left) / scale, 0.0), w - 1.0)
        y1 = min(max((y1 - top) / scale, 0.0), h - 1.0)
        y2 = min(max((y2 - top) / scale, 0.0), h - 1.0)
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    return boxes


def check(image_path, mask_path, min_cover=0.6, weights=None):
    weights = weights or os.environ.get("PERSON_CHECK_WEIGHTS")
    if not weights:
        raise SystemExit("set PERSON_CHECK_WEIGHTS to the yolov8n-seg-folded safetensors")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {image_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"cannot read mask: {mask_path}")
    if mask.shape[:2] != image.shape[:2]:
        raise SystemExit(f"mask {mask.shape[:2]} does not match image {image.shape[:2]}")
    mask = (mask > 127).astype(np.uint8)

    persons = []
    for x1, y1, x2, y2, conf in person_boxes(load_net(weights), image):
        area = max((x2 - x1) * (y2 - y1), 1)
        cover = int(mask[y1:y2, x1:x2].sum()) / float(area)
        persons.append({"box": [x1, y1, x2, y2], "conf": round(conf, 3), "cover": round(cover, 3)})
    max_cover = max((p["cover"] for p in persons), default=0.0)
    return {"fail": max_cover >= min_cover, "max_cover": round(max_cover, 3), "min_cover": min_cover, "persons": persons}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    print(json.dumps(check(sys.argv[1], sys.argv[2], threshold)))
