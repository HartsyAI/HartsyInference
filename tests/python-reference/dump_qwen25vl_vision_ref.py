"""Qwen2.5-VL vision tower reference (from the mmproj GGUF weights), used to validate the C# Qwen25VlEncoder.

Builds a deterministic test image, patchifies it in Qwen's merge-block order, and runs the full windowed
2D-RoPE ViT + patch-merger. Saves the [1,3,H,W] image (ref_image.f32) so C# uses the identical input, and dumps
per-stage tensors (ref_*.f32). Run the C# encoder with HARTSY_VLM_DUMP pointed at the same dir to diff.
"""
import numpy as np, torch, torch.nn.functional as F
from gguf import GGUFReader

DUMP = '/tmp/claude-1000/-home-kalebbroo-Desktop-Projects-SharpInference/eb7ffcf1-a730-4648-9a49-2e6f0df16099/scratchpad/qwendump'
import os; os.makedirs(DUMP, exist_ok=True)
r = GGUFReader('/tmp/qwenvl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf')
W = {t.name: torch.from_numpy(t.data.astype(np.float32).copy()) for t in r.tensors}

HID, LAYERS, HEADS, FFN, PATCH = 1280, 32, 16, 3420, 14
D = HID // HEADS                # 80
MERGE = 2
WINDOW = 112                    # pixels
EPS = 1e-6
FULLATT = {7, 15, 23, 31}       # n_wa_pattern=8
MEAN = [0.48145467, 0.45782750, 0.40821072]; STD = [0.26862955, 0.26130259, 0.27577710]
IMG = 224                       # test image (grid 16x16, merged 8x8, 4x4-merged windows -> 2x2=4 windows)
GRID = IMG // PATCH             # 16
T = 1

def save(tag, t): t.detach().cpu().numpy().astype(np.float32).tofile(f'{DUMP}/ref_{tag}.f32')

# --- deterministic test image: red disk on white, normalized [1,3,IMG,IMG] ---
img = np.full((3, IMG, IMG), 255.0, np.float32)
cy = cx = IMG // 2; rad = IMG // 3
for y in range(IMG):
    for x in range(IMG):
        if (x - cx) ** 2 + (y - cy) ** 2 <= rad * rad:
            img[:, y, x] = [220, 30, 30]
for c in range(3):
    img[c] = (img[c] / 255.0 - MEAN[c]) / STD[c]
img.tofile(f'{DUMP}/ref_image.f32')             # [3,IMG,IMG] for C# to load
px = torch.from_numpy(img)

# --- patchify in Qwen merge-block order: [grid_t, gh//m, gw//m, m, m] flattened, each patch [3*2*14*14] ---
# temporal_patch=2 -> duplicate the single image across 2 frames.
gh = gw = GRID
patches = []
for bh in range(gh // MERGE):
    for bw in range(gw // MERGE):
        for mh in range(MERGE):
            for mw in range(MERGE):
                ph, pw = (bh * MERGE + mh) * PATCH, (bw * MERGE + mw) * PATCH
                patch = px[:, ph:ph + PATCH, pw:pw + PATCH]          # [3,14,14]
                # flatten as [temporal(2), channel(3), 14,14] -> two identical frames
                flat = torch.cat([patch.reshape(-1), patch.reshape(-1)])  # frame0 then frame1? check below
                patches.append(flat)
NP = len(patches)                                                    # gh*gw
flat_patches = torch.stack(patches)                                  # [NP, 1176]
save('patches', flat_patches)

# --- patch embed: two temporal conv weights, each [1280,3,14,14] = [1280,588]; concat over temporal ---
w0 = W['v.patch_embd.weight'].reshape(HID, -1)                       # [1280, 588]
w1 = W['v.patch_embd.weight.1'].reshape(HID, -1)                     # [1280, 588]
# input patch layout per frame = [channel,14,14]=588; our flat = [frame0(588), frame1(588)]
hidden = flat_patches[:, :588] @ w0.T + flat_patches[:, 588:] @ w1.T  # [NP, 1280]
save('embed', hidden)

# --- 2D rope position ids (merge-permuted), then rotary table ---
def rope_pos_ids(gh, gw):
    hpos = torch.arange(gh).unsqueeze(1).expand(-1, gw)
    hpos = hpos.reshape(gh // MERGE, MERGE, gw // MERGE, MERGE).permute(0, 2, 1, 3).flatten()
    wpos = torch.arange(gw).unsqueeze(0).expand(gh, -1)
    wpos = wpos.reshape(gh // MERGE, MERGE, gw // MERGE, MERGE).permute(0, 2, 1, 3).flatten()
    return torch.stack([hpos, wpos], dim=-1)                         # [NP,2]
pos = rope_pos_ids(gh, gw)
rope_dim = D // 2                                                    # 40
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))   # [20]
maxg = max(gh, gw)
freqs_full = torch.outer(torch.arange(maxg).float(), inv_freq)      # [maxg,20]
rot = freqs_full[pos].flatten(1)                                    # [NP, 40]  (h:20 ++ w:20)
emb = torch.cat([rot, rot], dim=-1)                                 # [NP, 80]
cos, sin = emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)
def apply_rope(x):  # x: [NP, heads, D]
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]

# --- window index (in merge-units) ---
def window_index(gh, gw):
    vit_win = WINDOW // PATCH // MERGE                               # 112//14//2 = 4 (merged units)
    lh, lw = gh // MERGE, gw // MERGE
    idx = torch.arange(lh * lw).reshape(1, lh, lw)
    pad_h = (vit_win - lh % vit_win) % vit_win
    pad_w = (vit_win - lw % vit_win) % vit_win
    nwh, nww = (lh + pad_h) // vit_win, (lw + pad_w) // vit_win
    ip = F.pad(idx, (0, pad_w, 0, pad_h), value=-100)
    ip = ip.reshape(1, nwh, vit_win, nww, vit_win).permute(0, 1, 3, 2, 4).reshape(1, nwh * nww, vit_win, vit_win)
    seqlens = (ip != -100).sum([2, 3]).reshape(-1)
    ipf = ip.reshape(-1)
    new = ipf[ipf != -100]
    cu = (seqlens.cumsum(0) * MERGE * MERGE)
    cu = torch.cat([torch.zeros(1, dtype=cu.dtype), cu])
    return new, cu
win_idx, cu_win = window_index(gh, gw)                              # win_idx in merge-units
# reorder hidden + rope into window order (merge-units of 4)
def reorder(x):  # x [NP, ...]
    x = x.reshape(NP // (MERGE * MERGE), MERGE * MERGE, *x.shape[1:])
    return x[win_idx].reshape(NP, *x.shape[2:])
hidden = reorder(hidden)
cos_w = reorder(cos); sin_w = reorder(sin)
save('embed_win', hidden)

def block(h, i, cos_, sin_):
    p = f'v.blk.{i}'
    def rms(x, w): return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
    def lin(x, n):
        y = x @ W[n + '.weight'].T
        return y + W[n + '.bias'] if n + '.bias' in W else y
    x = rms(h, W[f'{p}.ln1.weight'])
    q = lin(x, f'{p}.attn_q').reshape(NP, HEADS, D)
    k = lin(x, f'{p}.attn_k').reshape(NP, HEADS, D)
    v = lin(x, f'{p}.attn_v').reshape(NP, HEADS, D)
    # apply 2D rope (with window-reordered cos/sin)
    qr = (q * cos_[:, None] + rotate_half(q) * sin_[:, None])
    kr = (k * cos_[:, None] + rotate_half(k) * sin_[:, None])
    # attention mask: full for FULLATT layers, else block-diagonal over windows (cu_win)
    mask = torch.full((NP, NP), float('-inf'))
    if i in FULLATT:
        mask[:] = 0.0
    else:
        for a in range(len(cu_win) - 1):
            s, e = int(cu_win[a]), int(cu_win[a + 1])
            mask[s:e, s:e] = 0.0
    qh = qr.permute(1, 0, 2); kh = kr.permute(1, 0, 2); vh = v.permute(1, 0, 2)   # [heads,NP,D]
    attn = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[None])
    attn = attn.permute(1, 0, 2).reshape(NP, HID)
    h = h + lin(attn, f'{p}.attn_out')
    x = rms(h, W[f'{p}.ln2.weight'])
    gate = F.silu(lin(x, f'{p}.ffn_gate'))
    up = lin(x, f'{p}.ffn_up')
    h = h + lin(gate * up, f'{p}.ffn_down')
    return h

h = hidden
for i in range(LAYERS):
    h = block(h, i, cos_w, sin_w)
    if i == 0: save('blk0', h)
# post norm (RMSNorm)
h = h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * W['v.post_ln.weight']
save('postln', h)
# merger: group 4 (merge units already consecutive) -> [NP/4, 5120] -> mm.0 -> gelu -> mm.2
m = h.reshape(NP // (MERGE * MERGE), HID * MERGE * MERGE)
m = F.gelu(m @ W['mm.0.weight'].T + W['mm.0.bias'])
m = m @ W['mm.2.weight'].T + W['mm.2.bias']                         # [NP/4, 2048]
# un-reorder to spatial order
rev = torch.argsort(win_idx)
m = m[rev]
save('embeds', m)
print(f"NP={NP} tokens={m.shape[0]} dim={m.shape[1]} embeds mean={m.mean():.4f} max={m.abs().max():.3f} finite={torch.isfinite(m).all().item()}")
print(f"cu_win={cu_win.tolist()}  win_idx[:8]={win_idx[:8].tolist()}")
