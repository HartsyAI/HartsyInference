# Phase 6 — Deviations from Design Plan

This document tracks every case where the Vision package implementation diverged from the reference Python (Ultralytics / HuggingFace transformers) behavior, how the bug was found, and how it was fixed.

Format mirrors [PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md) and [PHASE_3_5_DEVIATIONS.md](PHASE_3_5_DEVIATIONS.md): one entry per issue with **Design assumption → Deviation → How it was found → Fix → Impact**.

---

## YOLO11 — Block library deviations

These bugs were specific to the YOLO11 implementation. YOLOv8 worked end-to-end on the first pass because its block library (C2f + SPPF + DetectHead) doesn't hit any of these cases — the C3k / C3k2 / DetectHeadV11 / DwConvBnSilu / PsaAttention additions in v11 are where they surfaced.

### 1. `C3k` was implemented as a C2f variant; actually it's a C3 variant (3 convs, parallel branches)

**Design assumption**: From a casual read of the Ultralytics source — `class C3k(C2f): ...` — I assumed `C3k` inherited C2f's "expand → split → chain → concat-all → project" structure with only the inner unit changed.

**Deviation**: `C3k` actually inherits from **`C3`**, not `C2f`. The class hierarchy line in Ultralytics' source is:
```python
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)    # left branch
        self.cv2 = Conv(c1, c_, 1, 1)    # RIGHT BRANCH (parallel, takes same input)
        self.cv3 = Conv(2 * c_, c2, 1)   # final project
        self.m = nn.Sequential(*(Bottleneck(...) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```

So C3 (and therefore C3k) has **three** convs (`cv1`, `cv2`, `cv3`) where `cv1` and `cv2` are *parallel* branches — both take the original input. Bottlenecks chain from `cv1`'s output only. The concat is `[m(cv1(x)), cv2(x)]`, then `cv3` projects to `c_out`.

My first implementation copied C2f's pattern: single `cv1` expanding to `2*c`, split into halves, chain bottlenecks on the second half, concat all accumulated tensors, final `cv2`. Wrong both structurally and weight-key-wise — the `cv3` key wouldn't load and the weight shapes were off.

**How it was found**: After implementing the full YOLO11 pipeline, the end-to-end test on `bus.png` returned 0 detections. A diagnostic dump of the converted safetensors revealed three convs per inner C3k unit (`model.6.m.0.cv1`, `cv2`, `cv3`) — not the two my C2f-style code expected. Reading the Ultralytics source then made the inheritance chain obvious.

**Fix**: Rewrote [`C3k.Forward`](../../src/SharpInference.Vision/Detection/Blocks/C3k.cs):
```csharp
public Tensor Forward(IBackend backend, Tensor input)
{
    // Left branch: cv1 then sequential bottlenecks.
    Tensor left = _cv1.Forward(backend, input);
    for (int i = 0; i < _numBottlenecks; i++)
    {
        Tensor next = _bottlenecks[i].Forward(backend, left);
        left.Dispose();
        left = next;
    }
    // Right branch: cv2 directly on input (parallel to cv1).
    Tensor right = _cv2.Forward(backend, input);
    Tensor concatenated = ConcatChannel(left, right);
    return _cv3.Forward(backend, concatenated);
}
```

**Impact**: Necessary but not sufficient. After this fix the max class score went from 0.013 → 0.034 (still way below the reference's 0.94). Needed bug #2 to actually fix detections.

**Lesson**: Don't trust the class name. `C3k` looks like a "C3-with-k-parameter" name (and is — k is the kernel size param), but the inheritance is C3 not C2f. Always grep the Python source for `class XYZ(` before assuming the parent.

---

### 2. `Bottleneck` hardcoded `expansion=1.0`; YOLO11's C3k2(c3k=False) needs `expansion=0.5`

**Design assumption**: Ultralytics' `Bottleneck(c_, c_, ...)` always uses the default `e=0.5`, so its hidden channels are `c_/2`. But our internal call sites — `C2f`'s inner Bottlenecks, `C3k`'s inner Bottlenecks — all use `e=1.0` (no compression). So I baked `expansion=1.0` into the C# `Bottleneck` class.

**Deviation**: Both `C2f` (in v8) and `C3k` (used by v11's c3k=True path) explicitly override the default by passing `e=1.0`:
```python
# C2f:
self.m = nn.ModuleList(Bottleneck(self.c, self.c, ..., e=1.0) for _ in range(n))
# C3k:
self.m = nn.Sequential(*(Bottleneck(c_, c_, ..., k=(k, k), e=1.0) for _ in range(n)))
```

But `C3k2(c3k=False)` does NOT override the default:
```python
class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
```

`Bottleneck(self.c, self.c, shortcut, g)` uses Ultralytics' default `e=0.5`. So the inner Bottleneck in YOLO11's `C3k2(c3k=False)` blocks **compresses by half**: cv1: `c → c/2`, cv2: `c/2 → c`. The actual checkpoint weights confirm this — at YOLO11n layer 2 (C3k2, c3k=False, hidden=16):
```
model.2.m.0.cv1.conv.weight: [8, 16, 3, 3]   ← out=8 (NOT 16)
model.2.m.0.cv2.conv.weight: [16, 8, 3, 3]   ← in=8, out=16
```

My `Bottleneck(16, 16, shortcut)` with hardcoded `e=1.0` would have produced `[16, 16, 3, 3]` for both convs — but it didn't error at LoadWeights because the loader uses the *actual* tensor's `[8, 16, 3, 3]` shape directly. The mismatch only surfaced at runtime when `cv2`'s `Conv2D` tried to do `[B, 16, H, W] @ [16, 8, 3, 3]` and silently produced wrong (but not garbage) outputs.

Wait, actually — `Conv2D` would have errored on channel mismatch. The reason it didn't is that `ConvBnSilu.OutputSpatial` reads the *output* channel from the constructor (`_outChannels`), so it allocated an output tensor with the WRONG channel count (16 instead of 8 for cv1), and the conv kernel itself ran `[8, 16, 3, 3]` weight against `[1, 16, H, W]` input writing into `[1, 16, H, W]` output — only writing the first 8 channels of each spatial position, leaving the remaining 8 as zero/uninitialized. Then `cv2` read those partially-uninitialized values...

Hmm — actually let me not over-explain the exact failure mode; the relevant thing is: shape mismatch produced silent wrong output, not a clean exception.

**How it was found**:
1. End-to-end YOLO11n still failed after fix #1 (max class score 0.034 vs reference 0.94).
2. Wrote a Python script ([`/tmp/verify_folded_v11.py`](../../tests/python-reference/)) that emulated the C# forward path using the *same* BN-folded safetensors weights. Python reproduced the reference detections (max class score 0.94), confirming the conversion was correct.
3. Added a C# layer-by-layer diagnostic that loaded the Python-preprocessed input from disk and printed per-layer mean/std after each backbone stage. Layers 0 and 1 matched Python *bit-exactly*. Layer 2 (the first C3k2) diverged.
4. Wrote a focused unit test that ran just `C3k2(c3k=False, e=0.25)` on Python's `x1` and compared against Python's `x2`. **Max abs diff 11.65** — clearly broken.
5. Dumped the actual checkpoint weight shapes for `model.2.m.0.*` — they were `[8, 16, 3, 3]` and `[16, 8, 3, 3]`, revealing the half-compression.

**Fix**: Added an `expansion` parameter to [`Bottleneck`](../../src/SharpInference.Vision/Detection/Blocks/Bottleneck.cs) with default `1.0` (preserves v8 / C3k inner behavior), and have `C3k2` pass `expansion: 0.5f` explicitly when c3k=False:
```csharp
// Bottleneck.cs:
public Bottleneck(int inChannels, int outChannels, bool shortcut, float expansion = 1.0f)
{
    int hidden = (int)(outChannels * expansion);
    _cv1 = new ConvBnSilu(hidden, ...);
    _cv2 = new ConvBnSilu(outChannels, ...);
    ...
}

// C3k2.cs, c3k=False branch:
_bottlenecks[i] = new Bottleneck(_hiddenChannels, _hiddenChannels, shortcut, expansion: 0.5f);
```

**Impact**: After this fix, end-to-end YOLO11n produced 5 detections matching Ultralytics exactly: bus (0.940), person × 4 (0.902, 0.849, 0.833, 0.396).

**Lesson**: Don't trust a default parameter just because "every other call uses the same default" — Ultralytics' API has subtle inheritance interactions where the *same default* in two related classes means different things in context (C2f overrides Bottleneck's default explicitly; C3k2 does not). When porting a class library, dump the actual checkpoint tensor shapes for each block type and verify they match the assumed channel widths *before* spending hours debugging numerics.

---

### 3. YOLO11's classification branch uses depthwise-separable convs (different from YOLOv8)

**Design assumption**: YOLO11's `Detect` head has the same structure as YOLOv8 — three plain conv stages per branch (`cv2[s].0`, `cv2[s].1`, `cv2[s].2` for box; same for cv3 class).

**Deviation**: YOLOv8 has the symmetric structure but YOLO11 split the class branch into a depthwise-separable form to cut parameters:
```python
# Ultralytics Detect.__init__ — non-legacy (v11) path:
self.cv3 = nn.ModuleList(
    nn.Sequential(
        nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),   # depthwise 3×3 + pointwise 1×1
        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
        nn.Conv2d(c3, self.nc, 1),                        # final plain 1×1
    )
    for x in ch
)
```

The keys in the converted safetensors reflect this nesting:
```
model.23.cv3.{s}.0.0.conv.weight   ← depthwise 3×3 (c_in channels)
model.23.cv3.{s}.0.1.conv.weight   ← pointwise 1×1 (c_in → c3)
model.23.cv3.{s}.1.0.conv.weight   ← depthwise 3×3 (c3 channels)
model.23.cv3.{s}.1.1.conv.weight   ← pointwise 1×1 (c3 → c3)
model.23.cv3.{s}.2.weight          ← final plain 1×1 (c3 → nc, no BN/SiLU)
```

YOLOv8's cv3 was three `Conv` stages (`.0.conv`, `.1.conv`, `.2.weight`) — different layout, different layer count.

**How it was found**: Inspecting `model.23.*` keys in the converted YOLO11n checkpoint via the Python header-parsing diagnostic. The cv3 path had 8 BN-folded conv keys per scale (4 depthwise+pointwise) plus 1 plain conv key = 10 tensors per scale, instead of v8's 6.

**Fix**: Three coordinated changes:
1. New [`Conv2dDepthwise`](../../src/SharpInference.Core/Backends/IBackend.cs) op on `IBackend` with a CPU-loop default fallback. Backends override for performance.
2. New [`DwConvBnSilu`](../../src/SharpInference.Vision/Detection/Blocks/DwConvBnSilu.cs) helper — depthwise variant of `ConvBnSilu`. Calls `backend.Conv2dDepthwise` instead of `Conv2D` and expects weight shape `[C, 1, kH, kW]`.
3. New [`DetectHeadV11`](../../src/SharpInference.Vision/Detection/Blocks/DetectHeadV11.cs) — same box branch (cv2) as v8 but cv3 is the dw+pw chain. `YoloV11Model` uses `DetectHeadV11`; the original `DetectHead` is retained for YOLOv8.

**Impact**: Necessary to handle YOLO11 at all — the v8 `DetectHead` would have errored on missing keys during `LoadWeights`. Also unlocked a generally-useful `Conv2dDepthwise` op for future depthwise-using models (MobileNet variants, DINOv2 patch embed, etc.).

---

### 4. `PsaAttention` doesn't fit standard `ScaledDotProductAttention` — Q/K have `key_dim`, V has `head_dim`

**Design assumption**: YOLO11's C2PSA spatial attention is "just" multi-head self-attention over flattened spatial tokens, so it would fit `IBackend.ScaledDotProductAttention(Q, K, V, mask, scale)`.

**Deviation**: Standard SDPA expects Q, K, V to share the same shape `[B, H, S, D]` where `D` is the per-head dimension. PSA-Attention diverges:
- **Q and K** have channel dim `key_dim = head_dim * attn_ratio` (default attn_ratio=0.5 → key_dim = head_dim / 2)
- **V** has channel dim `head_dim`

So Q is `[B, nh, kd, N]`, K is `[B, nh, kd, N]`, V is `[B, nh, head_dim, N]` — different ranks of "channel" across Q/K vs V. And the spatial axis is the *second-to-last* (N), not the last as in standard attention.

The forward also includes a depthwise 3×3 positional encoding added to V before the output projection:
```python
attn = (q.transpose(-2, -1) @ k) * scale     # [B, nh, N, N]
attn = attn.softmax(dim=-1)
x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
x = self.proj(x)
```

`backend.ScaledDotProductAttention`'s standard contract wouldn't produce this output without contortions.

**How it was found**: While reading Ultralytics' `Attention.__init__` — the channel counts in the qkv projection (`dim + 2*nh*kd` instead of `3*nh*head_dim`) made the asymmetry obvious. Confirmed by checking the actual qkv weight shape in the checkpoint: `[256, 128, 1, 1]` for layer 10's qkv (with dim=128, nh=2, kd=32: 128 + 2*2*32 = 256 ✓).

**Fix**: Hand-coded the matmuls and softmax inline in [`PsaAttention.ComputeAttention`](../../src/SharpInference.Vision/Detection/Blocks/PsaAttention.cs). The spatial sizes are small (N ≤ 1600 at 640×640 input) so the manual loop performs adequately on CPU. The depthwise PE is folded into the same per-head loop. A future GPU port can fuse these into a single kernel.

**Impact**: This block worked on the first try (the C2PSA block diagnostic test matched Python with max abs diff = 7e-6). The lesson is that "this looks like standard attention" can be misleading — check the QKV projection output channels against `3 * total_head_channels` before assuming standard SDPA applies.

---

## YOLO11 Converter Deviations

### 5. Hard-coding the detect-head layer index in the Python conversion script

**Design assumption**: The conversion script's fallback for plain (non-BN) Conv2d weights — the final 1×1 projections inside detection heads — only needed to handle YOLOv8's detect head at layer 22.

**Deviation**: Initial script:
```python
for key, tensor in state.items():
    if "model.22." in key and (".2.weight" in key or ".2.bias" in key):
        folded[key] = tensor.contiguous().clone()
```

YOLO11's detect head is at **layer 23** (extra C2PSA pushed it back by one). The hardcoded `model.22.` check missed every YOLO11 final projection conv. Additionally, YOLO11's cv3 has a `.2` plain conv that doesn't fit the original v8 pattern (it's part of a 5-stage class branch rather than v8's 3-stage one).

The converter still produced a safetensors file, but it was missing the `model.23.cv2.{s}.2.{weight,bias}` and `model.23.cv3.{s}.2.{weight,bias}` keys — 12 tensors per detect head. Loading would fail at runtime with `KeyNotFoundException`.

**How it was found**: First conversion of `yolo11n.pt` succeeded but the converter's debug print showed only 163 folded tensors (vs the 175 we'd expect). Listing all keys in the output revealed missing `model.23.cv2.*.2.*` and `model.23.cv3.*.2.*`.

**Fix**: Refactored the converter to a two-pass approach in [`convert_yolov8_pt_to_safetensors.py`](../../tests/python-reference/convert_yolov8_pt_to_safetensors.py):
1. **Pass 1**: Walk `*.conv.weight` keys. Fold BN siblings if present, otherwise copy the conv directly. Mark all participating keys (conv weight/bias + BN params) as consumed.
2. **Pass 2**: Walk every remaining `*.weight` / `*.bias` key that wasn't consumed and copy as-is. This is generic — it picks up YOLOv8's detect head, YOLO11's detect head, and any future variant's plain Conv2d weights without code changes.

**Impact**: Two-pass approach is now architecture-agnostic. The converter handles v8 (175 → 127 tensors) and v11 (499 → 175 tensors) identically. Future YOLO variants (12, 13, segmentation, OBB) should just work as long as they follow the Ultralytics `Conv` wrapper / `nn.Conv2d` distinction for BN vs no-BN convs.

**Lesson**: Conversion scripts should drive off-of-pattern, not off-of-named-layer-indices. The "what was consumed" tracking is a small amount of bookkeeping that pays back the first time a new architecture lands.

---

## Diagnostic methodology (worth keeping)

The bisect approach that found bug #2 is general — when a deep network produces wrong end-to-end output but the individual blocks look correct in isolation:

1. **Write a Python script that uses the SAME folded weights** to do the forward pass. If Python produces the right answer (it did), the bug is in C# implementation, not in the weights or conversion. If Python ALSO produces the wrong answer, the bug is upstream — in the converter or the assumed forward algorithm.
2. **Dump the preprocessed input from Python to disk** so C# can read it. Eliminates preprocessor differences (PIL bilinear vs. our bilinear) as a variable.
3. **Run C# layer-by-layer with stats logging** (mean, std, min, max). Compare against Python at each layer. The first layer where stats diverge is where the bug lives.
4. **For the divergent layer, write a focused unit test** that loads Python's reference input/output binary dumps and compares element-wise. Quantify max abs diff — if it's beyond float-precision noise (say > 1e-3), you have a real bug.
5. **Check the actual checkpoint weight shapes against your assumed shapes**. Channel mismatches in Bottleneck-style blocks are easy to overlook because the surrounding cv1/cv2 weights "look right" — but the inner Bottleneck's hidden width is where assumptions break down.

The diagnostic tests written during this debug session were all deleted before merge (they had `/tmp/` paths and were single-use). The methodology is what's worth keeping — the bisect from "C2PSA matches" → "C3k2 c3k=True matches" → "DetectHeadV11 matches" → "C3k2 c3k=False on real input does NOT match" → bug in `Bottleneck` took about 90 minutes once the framework was in place.
