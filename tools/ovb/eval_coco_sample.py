import json, os, glob
import numpy as np
from PIL import Image
ROOT = '/ds'
s = json.load(open(f'{ROOT}/sample40.json'))
classes = s['classes']; BG = len(classes); K = BG + 1
conf = np.zeros((K, K), np.int64)
per_image = []
pred_bg_only = np.zeros((K, K), np.int64)
for iid, fn in zip(s['image_ids'], s['file_names']):
    stem = fn[:-4]
    gt = np.asarray(Image.open(f'{ROOT}/gt/{iid:012d}.png'), np.uint8)
    H, W = gt.shape
    masks = sorted(glob.glob(f'{ROOT}/ovb-seg/{stem}/**/*.png', recursive=True))
    masks = [m for m in masks if 'done' not in m]
    assert len(masks) == len(classes), (stem, len(masks))
    pred = np.full((H, W), BG, np.uint8)
    for c in reversed(range(len(classes))):   # lowest class index wins an overlap
        m = Image.open(masks[c]).convert('L')
        if m.size != (W, H):
            m = m.resize((W, H), Image.NEAREST)
        pred[np.asarray(m) > 127] = c
    valid = gt != 255
    g = gt[valid].astype(np.int64); p = pred[valid].astype(np.int64)
    c_img = np.bincount(g * K + p, minlength=K * K).reshape(K, K)
    conf += c_img
    pred_bg_only += np.bincount(g * K + BG, minlength=K * K).reshape(K, K)
    per_image.append({'file': fn, 'pixels_valid': int(valid.sum()), 'pred_fraction': {classes[c]: float((pred[valid] == c).mean()) for c in range(len(classes))}})
def iou_table(cm):
    out = {}
    for c in range(K):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        union = tp + fp + fn
        out[(classes + ['background'])[c]] = float(tp / union) if union > 0 else None
    return out
iou = iou_table(conf)
obj = [iou[c] for c in classes if iou[c] is not None]
allv = [v for v in iou.values() if v is not None]
gt_px = {c: int(conf[i, :].sum()) for i, c in enumerate(classes)}
res = {
    'sample_images': len(s['image_ids']),
    'classes': classes,
    'iou': iou,
    'mIoU_5_objects': float(np.mean(obj)) if obj else None,
    'mIoU_6_with_background': float(np.mean(allv)),
    'gt_pixels_per_class': gt_px,
    'pixel_accuracy': float(np.trace(conf) / conf.sum()),
    'baseline_all_background_mIoU_6': float(np.mean([v for v in iou_table(pred_bg_only).values() if v is not None])),
    'pixels_evaluated': int(conf.sum()),
}
json.dump({'summary': res, 'per_image': per_image}, open(f'{ROOT}/ovb-seg-metrics.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
