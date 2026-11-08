"""Offline reference dump for the C# HairDeck preprocessing port (never imported by engine code).

Runs hairdeck.prep.preprocess.Preprocessor (preset "best": SAM 3 text grounding and BiRefNet-matting) on one image, in
float32 on the CPU, and writes reference.safetensors plus a reference.json sidecar. The parity test reads both.

Setup: a HairDeck source checkout at the pinned commit cb20f7073419a33c9ab88f22f686fcb9060a99f0 (its hairdeck/ package
directory is given with --code-dir), a SAM 3 source checkout at 0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc (--sam3-repo),
sam3.pt, and the BiRefNet-matting code files of revision eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9 next to model.safetensors
(--weights-dir, laid out as hairdeck.paths expects: sam3/sam3.pt and birefnet_matting/{birefnet.py, BiRefNet_config.py,
config.json, model.safetensors}).

Run isolated, with every GPU hidden:

    set CUDA_VISIBLE_DEVICES=-1
    python -I dump_hairdeck_prep.py --code-dir <dir that contains hairdeck/> --sam3-repo <sam3 checkout> \
        --weights-dir <weights> --image <png> --out <dir>

The float32 CPU run differs from the GPU release, which uses bfloat16 autocast in SAM 3 and half precision in BiRefNet;
the parity test reports the differences rather than assuming them away.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-dir', required=True, help='directory that holds the hairdeck/ package')
    parser.add_argument('--sam3-repo', required=True, help='SAM 3 source checkout (the sam3/ package directory)')
    parser.add_argument('--weights-dir', required=True, help='HAIRDECK_WEIGHTS layout: sam3/sam3.pt, birefnet_matting/')
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise SystemExit('CPU-only dump: set CUDA_VISIBLE_DEVICES=-1 so no GPU is touched')
    os.environ['HAIRDECK_DEVICE'] = 'cpu'
    os.environ['HAIRDECK_WEIGHTS'] = args.weights_dir
    os.environ['HAIRDECK_SAM3_REPO'] = args.sam3_repo
    os.environ.setdefault('HAIRDECK_ROOT', args.out)
    os.environ.setdefault('HAIRDECK_CONFIG', os.path.join(args.out, 'no_config.json'))
    sys.path.insert(0, args.code_dir)
    sys.path.insert(0, args.sam3_repo)
    os.makedirs(args.out, exist_ok=True)

    import kornia.filters  # noqa: F401  (TorchScript compiles at import; the patches below would break it)

    # Upstream CUDA-only paths, replaced on this CPU run with the same math in float32.
    import sam3.perflib.fused as fused

    def addmm_plain(activation, linear, mat1):
        return activation()(linear(mat1))

    fused.addmm_act = addmm_plain
    for name, module in list(sys.modules.items()):
        if name.startswith('sam3') and getattr(module, 'addmm_act', None) is not None and module is not fused:
            module.addmm_act = addmm_plain
    for fn_name in ('zeros', 'ones', 'empty', 'full', 'arange', 'randn'):
        def wrap(fn):
            def inner(*a, **k):
                if k.get('device') in ('cuda', 'cuda:0'):
                    k['device'] = 'cpu'
                return fn(*a, **k)
            return inner
        setattr(torch, fn_name, wrap(getattr(torch, fn_name)))
    torch.Tensor.pin_memory = lambda self, *a, **k: self

    import hairdeck.prep.preprocess as pp  # noqa: E402
    from hairdeck.data.crop import apply_crop, composite  # noqa: E402
    from hairdeck.data.illus import crop_white  # noqa: E402
    from hairdeck.prep import matte as matte_mod  # noqa: E402

    captured = {'sam': [], 'matte': []}
    original_ground = pp.ground

    def ground_hook(model, state, tf, idx, device=None):
        probs, boxes, masks = original_ground(model, state, tf, idx, device=device)
        captured['sam'].append((probs.detach().float().cpu().numpy().copy(), masks.detach().float().cpu().numpy().copy()))
        return probs, boxes, masks

    pp.ground = ground_hook

    def matte_float(self, rgb):
        # the upstream call without the half-precision cast, so the reference matte is float32 as the port's is
        h0, w0 = rgb.shape[:2]
        t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float() / 255
        t = torch.nn.functional.interpolate(t, (self.size, self.size), mode='bilinear', align_corners=False)
        t = (t - self.mean) / self.std
        a = self.net(t)[-1].sigmoid().float()
        out = torch.nn.functional.interpolate(a, (h0, w0), mode='bilinear', align_corners=False)[0, 0]
        captured['matte'].append(out.detach().float().cpu().numpy().copy())
        return out

    matte_mod.BiRefNet.__call__ = torch.no_grad()(matte_float)

    im0 = Image.open(args.image)
    if im0.getexif().get(0x0112, 1) not in (1, None):
        raise SystemExit('EXIF orientation is applied by the C# decoder; use an upright input for the dump')
    rgba_src = np.asarray(im0.convert('RGBA'))
    has_alpha = im0.mode in ('RGBA', 'LA', 'PA', 'RGBa', 'La') or 'transparency' in im0.info
    if has_alpha:
        rgb = np.ascontiguousarray(rgba_src[..., :3])
        alpha = np.ascontiguousarray(rgba_src[..., 3])
    else:
        rgb = np.asarray(im0.convert('RGB'))
        alpha = None

    pre = pp.Preprocessor('best')
    pre.prep.model.float()
    pre.prep.matte.net.float()
    p = pre(rgb, alpha, args.image)

    height, width = rgb.shape[:2]
    alpha_m = p['person'].copy()
    if alpha_m.max() <= 127:
        alpha_m[:] = 255
    full = np.asarray(composite(Image.fromarray(np.concatenate([p['rgb'], alpha_m[..., None]], -1), 'RGBA')))
    model_input = crop_white(full, tuple(p['box']), 448)
    matte_crop = np.asarray(apply_crop(Image.fromarray(p['person']), tuple(p['box']), 448,
                                       resample=Image.Resampling.NEAREST)) > 127
    if not matte_crop.any():
        matte_crop = np.ones_like(matte_crop)

    hair = p['hair']
    if not hair.any():
        hair_color = [0.4, 0.3, 0.25]
    else:
        px = p['rgb'][hair].astype(np.float64)
        lum = px @ np.array([0.299, 0.587, 0.114])
        hair_color = (np.median(px[lum <= np.median(lum)], 0) / 255).tolist()

    tensors = {
        'rgba_src': rgba_src.astype(np.uint8),
        'person': p['person'].astype(np.uint8),
        'hair': p['hair'].astype(np.uint8),
        'face': p['face'].astype(np.uint8),
        'box': np.array(p['box'] if p['box'] is not None else [np.nan] * 4, np.float32),
        'model_input': np.ascontiguousarray(model_input).astype(np.uint8),
        'matte_crop': matte_crop.astype(np.uint8) * 255,
        'hair_color': np.array(hair_color, np.float32),
        'alpha_full': captured['matte'][0].astype(np.float32),
    }
    scores, masks = captured['sam'][0]
    kept = {}
    for j, name in enumerate(pre.prep.prompts):
        keep = np.nonzero(scores[j] > pre.prep.t_min)[0]
        keep = keep[np.argsort(scores[j][keep])[::-1]]
        kept[name] = [float(scores[j][k]) for k in keep]
        tensors[f'sam_scores_{j}'] = scores[j][keep].astype(np.float32)
        tensors[f'sam_logits_{j}'] = masks[j][keep].astype(np.float32)

    save_file(tensors, os.path.join(args.out, 'reference.safetensors'))
    with open(os.path.join(args.out, 'reference.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(image=args.image, size=[width, height], status=p['status'], notes=p['notes'],
                       info=p['info'], prompts=list(pre.prep.prompts), t_min=pre.prep.t_min, kept_scores=kept,
                       box=p['box'], backend=p['backend'], torch=torch.__version__, device=str(pre.prep.device)),
                  f, indent=1, default=str)
    print('status', p['status'], 'box', p['box'], 'notes', p['notes'])
    print('wrote', args.out)


if __name__ == '__main__':
    main()
