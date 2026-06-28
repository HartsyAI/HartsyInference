"""Reference for the Llama-3.2-Vision (mllama) vision tower, computed directly from the mmproj GGUF weights (the
weights are gated on HF, so we cannot run HF's MllamaVisionModel — instead we reimplement its forward from the
documented math and the Ollama converter's gate handling, which this GGUF targets).

Key facts established from transformers `modeling_mllama.py` + ollama `convert/convert_mllama.go`:
  * Gates are PRE-TANH'd at conversion: every `*.gate` in the GGUF already stores tanh(g); `position_embd.gate`
    additionally stores 1 - tanh(g). So the forward multiplies by the stored gate DIRECTLY (no tanh at inference).
    Position combine: hidden += position_embd.gate * base_pos + tile_position_embd.gate * tile_pos.
  * Aspect/tile embeds: hidden += gate * embedding[aspect_id][tile]  (pre-tile before transformer, post-tile after
    local + layernorm_post, before global).
  * Local: 32 plain pre-norm ViT blocks (LN→MHA(no bias)→+res→LN→MLP(gelu, biased)→+res).
  * Global: 8 blocks with extra direct-multiply gates: h = res + attn_gate*attn; h = res + ffn_gate*mlp.
  * Output: concat([global_output, stack(local_hidden_states[i] for i in [3,7,15,23,30])]) → 7680, then mm.0.
  * hidden_states[0] = input to local layer 0 (after layernorm_pre + pos); hidden_states[k] = output of local
    layer k-1. So index i collects the output of local layer (i-1).
  * Single 560x560 tile (num_tiles=1, aspect_ratio_id=1 = the (1,1) ratio). Sequence = 1601 (1600 patches + CLS).
    HF pads to a multiple of 8 with a masked-out tail → identical to not padding for the real tokens, so we skip it.

Usage:  python dump_mllama_vision_ref.py /path/to/mmproj.f16.gguf [out_dir]
Saves ref_*.npy stage dumps + the fixed pixel input (ref_pixels.f32) for the C# diff.
"""
import sys, os
import numpy as np
from gguf import GGUFReader

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mllama_vis"
os.makedirs(out, exist_ok=True)
r = GGUFReader(path)
W = {t.name: np.array(t.data).astype(np.float32) for t in r.tensors}
def shp(n): return [int(x) for x in {t.name: t for t in r.tensors}[n].shape]  # ggml ne

HID, HEADS, FFN, NL, NG = 1280, 16, 5120, 32, 8
PATCH, IMG = 14, 560
GRID = IMG // PATCH          # 40
NP = GRID * GRID             # 1600
SEQ = NP + 1                 # 1601 (+CLS)
D = HID // HEADS             # 80
EPS = 1e-5
ASPECT = 1                   # (1,1) single-tile aspect-ratio id
INTERMEDIATE = [3, 7, 15, 23, 30]

def ln(x, w, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + EPS) * w + b
def lin(x, wname, bname=None):       # GGUF data is HF row-major [out,in]; in = x's last dim → y = x @ Wᵀ
    indim = x.shape[-1]
    outdim = W[wname].size // indim
    y = x @ W[wname].reshape(outdim, indim).T
    return y + W[bname] if bname else y
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Fixed pixel input [3, 560, 560] (deterministic, no real image needed for tower parity).
rng = np.random.RandomState(1234)
px = (rng.rand(3, IMG, IMG).astype(np.float32) - 0.5) * 2.0
px.tofile(os.path.join(out, "ref_pixels.f32"))

# --- patch embed: conv kernel=stride=14, no bias. weight ne=[14,14,3,1280] → [out=1280,in=3,kh,kw]. ---
pw = W["v.patch_embd.weight"].reshape(1280, 3, 14, 14)
patches = np.zeros((NP, HID), np.float32)
for py in range(GRID):
    for pxi in range(GRID):
        blk = px[:, py*PATCH:(py+1)*PATCH, pxi*PATCH:(pxi+1)*PATCH]   # [3,14,14]
        patches[py*GRID + pxi] = (pw * blk[None]).sum(axis=(1, 2, 3))
# pre-tile positional embedding (gate * embedding[aspect][tile0]); tile 0 of a single tile.
pre = W["v.pre_tile_position_embd.weight"].reshape(9, 4, HID)[ASPECT, 0]   # [1280]
patches = patches + W["v.pre_tile_position_embd.gate"][0] * pre[None]
# class embedding prepend → [1601,1280]
h = np.concatenate([W["v.class_embd"][None], patches], axis=0)
# gated positional embedding: base (gated) + tile (gated)
base = W["v.position_embd.weight"].reshape(SEQ, HID)                       # [1601,1280]
tile = W["v.tile_position_embd.weight"].reshape(9, 4, SEQ, HID)[ASPECT, 0] # [1601,1280]
h = h + W["v.position_embd.gate"][0] * base + W["v.tile_position_embd.gate"][0] * tile
# layernorm_pre
h = ln(h, W["v.pre_ln.weight"], W["v.pre_ln.bias"])
np.save(os.path.join(out, "ref_seq.npy"), h)

def block(h, p, gated):
    res = h
    x = ln(h, W[f"{p}.ln1.weight"], W[f"{p}.ln1.bias"])
    q = lin(x, f"{p}.attn_q.weight").reshape(SEQ, HEADS, D).transpose(1, 0, 2)
    k = lin(x, f"{p}.attn_k.weight").reshape(SEQ, HEADS, D).transpose(1, 0, 2)
    v = lin(x, f"{p}.attn_v.weight").reshape(SEQ, HEADS, D).transpose(1, 0, 2)
    sc = (q @ k.transpose(0, 2, 1)) / np.sqrt(D)
    sc = sc - sc.max(-1, keepdims=True)
    a = np.exp(sc); a = a / a.sum(-1, keepdims=True)
    o = (a @ v).transpose(1, 0, 2).reshape(SEQ, HID)
    o = lin(o, f"{p}.attn_out.weight")
    if gated: o = W[f"{p}.attn_gate"][0] * o
    h = res + o
    res = h
    x = ln(h, W[f"{p}.ln2.weight"], W[f"{p}.ln2.bias"])
    # clip swaps the MLP names: ffn_down is fc1 (up, 1280→5120), ffn_up is fc2 (down, 5120→1280).
    x = gelu(lin(x, f"{p}.ffn_down.weight", f"{p}.ffn_down.bias"))
    x = lin(x, f"{p}.ffn_up.weight", f"{p}.ffn_up.bias")
    if gated: x = W[f"{p}.ffn_gate"][0] * x
    return res + x

# local transformer, collecting hidden_states. hs[0]=input, hs[k]=output of layer k-1.
hs = [h.copy()]
for i in range(NL):
    h = block(h, f"v.blk.{i}", gated=False)
    hs.append(h.copy())
    if i == 0: np.save(os.path.join(out, "ref_blk0.npy"), h)
h = ln(h, W["v.post_ln.weight"], W["v.post_ln.bias"])    # layernorm_post
np.save(os.path.join(out, "ref_postln.npy"), h)
# post-tile positional embedding
post = W["v.post_tile_position_embd.weight"].reshape(9, 4, HID)[ASPECT, 0]
h = h + W["v.post_tile_position_embd.gate"][0] * post[None]
# global transformer
for i in range(NG):
    h = block(h, f"v.global.blk.{i}", gated=True)
# concat global output + intermediate local hidden states
inter = [hs[idx] for idx in INTERMEDIATE]                # each [1601,1280]
cat = np.concatenate([h] + inter, axis=-1)               # [1601, 7680]
np.save(os.path.join(out, "ref_cat.npy"), cat)
# projector mm.0 (nn.Linear [7680->4096] + bias)
emb = lin(cat, "mm.0.weight", "mm.0.bias")               # [1601, 4096]
np.save(os.path.join(out, "ref_embeds.npy"), emb)
print("ref embeds shape", emb.shape, "first6", [round(float(v), 4) for v in emb[0, :6]])
print("saved to", out)
