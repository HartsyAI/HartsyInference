"""LLaVA-NeXT/1.6 (anyres) vision-pipeline reference.

Ground truth for two pieces NOT already validated by dump_llava_vision_ref.py (which proved the
CLIP tower + mm.0/mm.2 projector at corr=1.0 for LLaVA-1.5, and is byte-identical here since the
llava-v1.6-vicuna-7b mmproj tower/projector tensors match 1.5's shapes exactly):
  1. anyres tiling (select_best_resolution / resize+pad / divide_to_patches / base-image resize)
  2. pack_image_features merge (reshape/permute -> unpad -> image_newline insert -> base-first concat)

Per docs/Checklists/PARITY_VERIFICATION.md ("prefer running the real upstream library as an oracle"):
both pieces are produced by the REAL installed `transformers` library (image processor + the actual
`LlavaNextModel.pack_image_features`/`get_anyres_image_grid_shape`/`unpad_image` functions), not a
hand-reimplementation from memory -- this repo's own research flagged llama.cpp's own LLaVA-NeXT merge
as buggy (base placed last, no unpad, image_newline loaded but unused; ggml-org/llama.cpp#8457), and HF's
own docstrings in modeling_llava_next.py contradict each other on the image_size (H,W) vs (W,H)
convention -- so this script resolves that ambiguity empirically by reading select_best_resolution's
actual body, not by trusting either docstring.

The per-tile CLIP tower + projector math (clip_tower_and_project) is copy-identical to
dump_llava_vision_ref.py's validated single-tile code -- deliberately not re-derived.

Usage: run after the C# side dumps cs_tile{0..4}_embeds.f32 and cs_packed.f32 to DUMP (via
HARTSY_VLM_DUMP), matching the tags this script writes/compares.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gguf import GGUFReader
from transformers.models.llava_next.image_processing_llava_next import LlavaNextImageProcessor
from transformers.models.llava_next.modeling_llava_next import LlavaNextModel
from transformers.image_utils import PILImageResampling

MMPROJ = '/home/hartsy/Desktop/HartsyInference/Models/LLM/llava-next-vicuna-7b/llava-v1.6-vicuna-7b-mmproj-model-f16.gguf'
IMG_PATH = os.environ.get('HARTSY_TEST_IMAGE', '/home/hartsy/Desktop/HartsyInference/tests/HartsyInference.Cuda.Tests/TestData/bus.png')
DUMP = os.environ.get('HARTSY_VLM_DUMP', '/tmp/claude-1000/-home-hartsy/653b4ecd-7040-4d22-9749-94356f3a7c72/scratchpad/llavanextdump')
PREFIX = os.environ.get('HARTSY_DUMP_PREFIX', 'py')   # output files: {PREFIX}_pixel_values.f32, {PREFIX}_meta.txt, {PREFIX}_packed.f32;
# compare tags cs_{tag}.f32 stay unprefixed since the C# side dump name doesn't vary per test image.
GRID_PINPOINTS = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]  # (height, width) pairs -- matches
# clip.vision.image_grid_pinpoints GGUF metadata flat array [336,672, 672,336, 672,672, 1008,336, 336,1008] paired
# in file order, and matches select_best_resolution's actual (not docstring-claimed) (height, width) convention.

r = GGUFReader(MMPROJ)
W = {t.name: torch.from_numpy(t.data.astype(np.float32).copy()) for t in r.tensors}
HID, LAYERS, HEADS, INTER, PATCH, IMGSZ = 1024, 23, 16, 4096, 14, 336
GRID = (IMGSZ - PATCH) // PATCH + 1
NP = GRID * GRID
SEQ = NP + 1
D = HID // HEADS
EPS = 1e-5


def ln(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b, EPS)


def linT(x, name):
    y = x @ W[name + '.weight'].T
    if name + '.bias' in W:
        y = y + W[name + '.bias']
    return y


def qgelu(x):
    return x * torch.sigmoid(1.702 * x)


def clip_tower_and_project(pixel_tile):
    """pixel_tile: [1,3,336,336] normalized -> [1,576,4096] LLaVA-projected embedding (CLS dropped).
    Identical math to dump_llava_vision_ref.py's validated (corr=1.0) single-tile LLaVA-1.5 path."""
    conv = F.conv2d(pixel_tile, W['v.patch_embd.weight'].reshape(HID, 3, PATCH, PATCH), None, stride=PATCH)
    patches = conv.reshape(1, HID, NP).transpose(1, 2)
    seq = torch.cat([W['v.class_embd'].view(1, 1, HID), patches], dim=1)
    seq = seq + W['v.position_embd.weight'].reshape(1, SEQ, HID)
    seq = ln(seq, W['v.pre_ln.weight'], W['v.pre_ln.bias'])
    h = seq
    for i in range(LAYERS):
        p = f'v.blk.{i}'
        x = ln(h, W[f'{p}.ln1.weight'], W[f'{p}.ln1.bias'])
        q = linT(x, f'{p}.attn_q').reshape(1, SEQ, HEADS, D).transpose(1, 2)
        k = linT(x, f'{p}.attn_k').reshape(1, SEQ, HEADS, D).transpose(1, 2)
        v = linT(x, f'{p}.attn_v').reshape(1, SEQ, HEADS, D).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, SEQ, HID)
        h = h + linT(a, f'{p}.attn_out')
        x = ln(h, W[f'{p}.ln2.weight'], W[f'{p}.ln2.bias'])
        up = qgelu(linT(x, f'{p}.ffn_down'))   # clip swaps names: ffn_down = fc1 (up)
        h = h + linT(up, f'{p}.ffn_up')        # ffn_up = fc2 (down)
    # No post-LN (penultimate-layer output, same as LLaVA-1.5).
    pf = h[:, 1:, :]  # drop CLS
    mid = linT(pf, 'mm.0')
    act = F.gelu(mid, approximate='tanh')
    emb = linT(act, 'mm.2')
    return emb  # [1,576,4096]


def load(tag, shape):
    return torch.from_numpy(np.fromfile(f'{DUMP}/cs_{tag}.f32', dtype=np.float32).reshape(shape))


def cmp(tag, py):
    path = f'{DUMP}/cs_{tag}.f32'
    if not os.path.exists(path):
        print(f"{tag:14s} SKIPPED (no {path})")
        return
    cs = load(tag, tuple(py.shape))
    d = (py - cs).abs()
    cor = np.corrcoef(py.flatten().numpy(), cs.flatten().numpy())[0, 1]
    print(f"{tag:14s} py[mean={py.mean():.4f} max={py.abs().max():.3f}] cs[mean={cs.mean():.4f} max={cs.abs().max():.3f}] "
          f"maxdiff={d.max():.4f} corr={cor:.5f}")


# --- anyres tiling via the REAL HF image processor (not a memory reimplementation) ---
img = Image.open(IMG_PATH).convert('RGB')
proc = LlavaNextImageProcessor(
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
    size={"height": IMGSZ, "width": IMGSZ},
    crop_size={"height": IMGSZ, "width": IMGSZ},
    image_grid_pinpoints=GRID_PINPOINTS,
    resample=PILImageResampling.BICUBIC,
    do_center_crop=False,
)
out = proc.preprocess([img], return_tensors='pt')
pixel_values = out['pixel_values'][0]      # [num_patches, 3, 336, 336]
image_size = out['image_sizes'][0].tolist()  # [height, width], real HF (h,w) convention
num_patches = pixel_values.shape[0]
print(f"anyres tiling: num_patches={num_patches} image_size(h,w)={image_size}")

os.makedirs(DUMP, exist_ok=True)
np.asarray(pixel_values, dtype=np.float32).tofile(f'{DUMP}/{PREFIX}_pixel_values.f32')
with open(f'{DUMP}/{PREFIX}_meta.txt', 'w') as f:
    f.write(f"num_patches={num_patches}\nimage_size_h={image_size[0]}\nimage_size_w={image_size[1]}\n"
            f"pixel_values_shape={list(pixel_values.shape)}\n")


def dump_tag(name):
    return name if PREFIX == 'py' else f'{PREFIX}_{name}'


# --- per-tile CLIP tower + projector (validated math, reused verbatim) ---
tile_embeds = []
for i in range(num_patches):
    emb = clip_tower_and_project(pixel_values[i:i + 1])   # [1,576,4096]
    tile_embeds.append(emb[0])
    cmp(dump_tag(f'tile{i}_embeds'), emb[0])
image_feature = torch.stack(tile_embeds, dim=0)  # [num_patches, 576, 4096]

# --- pack_image_features: call the REAL bound method (not a reimplementation) with a config stub ---
class _VisionConfigStub:
    image_size = IMGSZ
    patch_size = PATCH


class _ConfigStub:
    vision_config = _VisionConfigStub()
    image_grid_pinpoints = GRID_PINPOINTS


class _ModelStub:
    config = _ConfigStub()


image_newline = W['model.image_newline']
packed_list, feature_lens = LlavaNextModel.pack_image_features(
    _ModelStub(), [image_feature], [image_size], vision_feature_select_strategy="default", image_newline=image_newline,
)
packed = packed_list[0]
print(f"pack_image_features: packed shape={list(packed.shape)} feature_lens={feature_lens.tolist()}")
np.asarray(packed, dtype=np.float32).tofile(f'{DUMP}/{PREFIX}_packed.f32')
cmp(dump_tag('packed'), packed)

print(f"\nDump written to {DUMP} ({PREFIX}_pixel_values.f32, {PREFIX}_meta.txt, {PREFIX}_packed.f32).")
print("Re-run after the C# side writes matching cs_*.f32 dumps to the same dir to compare.")
