# Vision Models — status

Concise status for vision models: CLIP embeddings, object detection, and segmentation / face. Open work
is in the [Remaining work](#remaining-work) section below; bring-up debugging notes live in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). Vision-tower parity for VLM use is tracked in
[MODEL_STATUS_LLM.md](MODEL_STATUS_LLM.md) and [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend:
[MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Status | Notes |
|---|---|---|
| **CLIP** (ViT-L/14, H/14, bigG/14) | ✅ | Real `openai/clip-vit-large-patch14` loads; L2 norm == 1.0, self-sim == 1.0, semantic ranking matches published OpenAI numbers. Bit-exact HF diff is the one remaining diagnostic. |
| **YOLOv8** (n/s/m/l/x) | ✅ | End-to-end on real `yolov8n-folded.safetensors`; bus.png detects 4 persons + 1 bus, matching Ultralytics one-for-one. Output shape `[1,84,8400]`. |
| **YOLO11** (n/s/m/l/x) | ✅ | End-to-end on real `yolo11n.pt` → safetensors; bus.png output exactly matches Ultralytics YOLO11n (bus 0.940 + 4 persons). |
| **Depth-Anything-V2** (ViT-S / ViT-L, relative depth) | ✅ | Real-weight parity vs the OFFICIAL DepthAnything/Depth-Anything-V2 implementation on a fixed 280×378 input (non-square → exercises pos-embed interpolation), CPU backend (2026-07-16): **ViT-S depth avg err 2.1e-6, ViT-L 4.5e-5 (rel 2.9e-7)**; all 17 tapped stages ≤ 4.5e-5 (test `DepthAnythingParityTests`, oracle `dump_depth_anything.py` + `diff_depth_anything.py`). ([details](#depth-anything-v2)) |
| **HED / SoftEdge** (ControlNetHED, softedge_hed + scribble_hed) | ✅ | Real-weight parity vs controlnet_aux `HEDdetector` on a fixed 320×256 image, CPU backend (2026-07-16): **fused edge avg err 3.9e-8**, all 5 side logits ≤ 1.1e-5; e2e uint8 forms code-exact (softedge mismatch 1.2e-5, safe 0, scribble 5.9e-4 — the scribble path re-implements cv2 GaussianBlur (auto-ksize 25/19, reflect-101) + 4-direction line NMS + binarize in C#). ([details](#hed--softedge)) |
| **Lineart** (sk_model realistic + sk_model2 coarse) | ✅ | Real-weight parity vs controlnet_aux `LineartDetector` on a fixed 320×256 image, CPU backend (2026-07-16): **line avg err 1.6e-8 (realistic) / 3.8e-8 (coarse)**, all stages ≤ 2.1e-6; e2e inverted uint8 conditioning map **bit-exact** (mismatch 0). ([details](#lineart)) |
| **NormalBAE** (surface normals, scannet NNET) | ✅ | Real-weight parity vs controlnet_aux `NormalBaeDetector` on a fixed 320×256 image, CPU backend (2026-07-16): **out_res1 avg err 7.9e-6**, all 14 stages ≤ 8.9e-6, e2e uint8 normal-RGB mismatch 1.5e-4. Test `NormalBaeParityTests`, oracle `dump_normalbae.py` + `diff_normalbae.py`. ([details](#normalbae)) |
| **UperNet-Seg** (semantic segmentation, ADE20K 150-class, ConvNeXt-Small) | ✅ | Real-weight parity vs transformers `UperNetForSemanticSegmentation` (`openmmlab/upernet-convnext-small` — the diffusers-documented annotator for `control_v11p_sd15_seg`) on a real 512² image (bus.png), CPU backend (2026-07-16): all 7 tapped stages at float noise (**feat_0..3 ≤ 6.8e-7, psp_out 7.0e-7, fpn_out 1.3e-6, logits_q 5.7e-6** vs ref mean abs 15.3), **per-pixel argmax class agreement 100.000%** (0/262144; ≥99% is the documented interpolation-boundary tolerance), ADE20K palette **byte-exact** vs controlnet_aux `ade_palette()`. ([details](#upernet-seg)) |
| **RMBG-1.4** (BriaRMBG / ISNet — background removal) | ✅ | Real-weight `briaai/RMBG-1.4` foreground mask on the RTX 4090 (2026-07-01). ([details](#rmbg-14)) |
| **BiRefNet-matting** (`ZhengPeng7/BiRefNet-matting`, Swin-L; trimap-free alpha matte) | ✅ (CPU) | Real-weight parity vs the upstream PyTorch code at revision `eccde0a8…`, sha256-verified weights, CPU backend (2026-11-07): `BiRefNetMattingParityTests` passes at logits maxAbs < 0.05 / corr > 0.9999, preprocess maxAbs < 0.1, and the 8-bit matte within 2 LSB with ≤ 0.1 % of pixels off by more than 1 LSB (exact figures not logged). Licence: MIT (`ZhengPeng7/BiRefNet-matting` Hugging Face metadata, 2026-11-08). Catalog stays ValidationPending until a GPU run. Files: `src/HartsyInference.Vision/Matting/BiRefNet/`. |
| **ArcFace IR-50** (InsightFace w600k_r50, face embedding) + **FaceAlignment** | ✅ | Real-weight parity vs onnxruntime on buffalo_l `w600k_r50.onnx` (2026-07-16): **512-d embedding cosine 1.000000, maxAbs ≤ 6.4e-6** on random + 2 real portrait crops; per-stage stem/layer1–4/emb all corr 1.0 (`ArcFaceParityTests` in `tests/HartsyInference.Vision.Tests/Parity/`, env `ARCFACE_WEIGHTS`/`ARCFACE_REF`; the per-stage debug test was removed 2026-08-06). ([details](#arcface-ir-50)) |
| **Open Vision Banana 4B** (FunAILab, FLUX.2 Klein Base 4B fine-tune, zero-shot averaged checkpoint, segmentation / depth / normals by editing) | ✅ (3060) | Real weights, fp32 shards narrowed to F16, 3060. Segmentation of 40 COCO val2017 images, five object classes, one legend, 28 steps, CFG 3.5: **mIoU 0.779** (person 0.878, car 0.655, bus 0.756, bicycle 0.712, dog 0.892; background 0.957; our protocol, not comparable to the paper's 73.9). Depth and normals qualitative on a 1024x472 street photo. Routed through `VisionService` (`vision --mode segment|depth|normal`). Licence: FLUX non-commercial. ([details](#open-vision-banana-generative-perception)) |
| **Depth Pro 2** (`depthpro2-s/-b/-l/-h`: metric depth in metres with a focal estimator; `hartsy vision -m depthpro2-s`, `--focal-px`) | ✅ S, B, L, H+ on CPU and CUDA; S also Vulkan | S and its focal estimator match the official ml-depth-pro-2 (`aa389cbb`) on CPU FP32, metric depth included (relRMS ≤ 1.1e-5, 2026-11-24). CLI and API run end to end on the real S weights: one image, views that share one focal length (the contract takes 2 to 64 and Depth Pro 2 at most 16; only two were run), and video frames (`--fps`, buffered `--window`/`--overlap` or `--unbuffered`). Views and frames attend to each other through the cross-frame blocks; the joint T=3 forward, buffered and unbuffered video match upstream `infer`/`infer_video` on CPU FP32 (relRMS at most 7.7e-6, 2026-11-24). Infinite depth (sky) is +inf with `ValidMask` 0. It runs at the image's own size, so images over 16384 patch tokens (2048x2048) are refused; at that cap S took 552 s and 6.4 GB on CPU. On the RTX 3060 (2026-11-24) image, joint views and video match the CPU lane: CUDA with `numerics.noTf32=true` and `numerics.sdpaNoTf32=true` within 1e-5 (views 3.3e-5), Vulkan within 3e-5; the engine's TF32 default was 2e-4 to 9e-4 off, so Depth Pro 2 now runs its own GEMMs and attention in plain F32 on CUDA (`ExactGemmScope`, A5). The 1024x472 street image took 1.5 s on CUDA (944 MiB), 4.5 s on Vulkan (its `ConvTranspose2d` runs on the host) and 9.7 s on CPU, load included; 24 frames at 384x240 took 3.0 s (buffered) and 3.5 s (unbuffered) on CUDA against about 51 s on CPU. B, L and H+ (2026-11-24, A5): the CPU F32 lane matches the official `ml-depth-pro-2` on a real photo for image, joint views, buffered and unbuffered video (inverse depth relRMS at most 5e-6, focal within 1.7e-6), and CUDA F32 (now the default) matches the CPU lane at most 7.4e-6 on the same modes at the 1024x576 / 3x768x432 / 12x384x216 inputs; bit-identical to `numerics.noTf32` + `numerics.sdpaNoTf32`. On the RTX 3060 the 1024x576 image takes 2.3 / 3.1 / 3.8 s (B / L / H+) against 34 / 145 / 197 s on CPU, 12 buffered frames 3.1 / 5.0 / 7.7 s; the inputs are shifted crops of one photo. All four catalog rows are Verified. On the RTX 4090 (2026-11-26) S passed the CUDA parity tests (image, joint views, video) at `934da1665`, after the 2026-11-25 device-path changes; Vulkan has not been rerun since them, and B, L and H+ have neither been rerun since them nor run on the 4090. F16/BF16, INT8 and Vulkan for B, L and H+ have not run. SwarmUI is deferred, as for every vision output ([ROADMAP.md](ROADMAP.md) §8). Weights are research-only (Apple Machine Learning Research Model License, non-commercial, revocable). See [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). |

## Detection / segmentation — real-weight verified (✅, 2026-07-17)

| Model | Status | Notes |
|---|---|---|
| **RT-DETR** (`PekingU/rtdetr_r18vd`) | ✅ | **Real-weight VERIFIED (2026-07-17)** vs transformers `RTDetrForObjectDetection` on bus.png: top-detection **box IoU 0.9996**, score **corr 1.0000**, all 5 detections matched (bus + 4 persons), exact labels. ([details](#rt-detr)) |
| **Grounding DINO** (open-vocab detection, `IDEA-Research/grounding-dino-tiny`) | ✅ | **Real-weight VERIFIED end-to-end (2026-07-17)** vs transformers `GroundingDinoForObjectDetection` on the COCO cats image + `"a cat. a remote control."` (CPU backend). ([details](#grounding-dino)) |
| **SAM 2 / MobileSAM** | ✅ | **Real-weight VERIFIED (2026-07-17, `facebook/sam2-hiera-tiny`):** Hiera windowed-attn encoder + two-way mask decoder built; best-mask **IoU 0.9972** vs HF `Sam2Model`, image-embed **corr 0.999**. ([details](#sam-2--mobilesam)) |
| **Face detection / landmarks** | ✅ | **Real-weight VERIFIED (2026-07-17, `akanametov/yolo-face` widerface-Pose):** YOLOv8-Face detector (v8 trunk + `cv4` 5-pt landmark branch) → box **IoU 0.9999**, landmark **mean 0.025 px / max 0.042 px** vs ultralytics; detector → `FaceAlignment` → `ArcFace` **512-d L2-norm 1.0**. ([details](#face-detection--landmarks)) |

**CLI catalog auto-download gap (2026-07-22):** `hartsy vision` wiring covers 13/17 real-weight-verified models
end-to-end (catalog `Assets` + auto-download + CLI-verified adversarially): CLIP, SigLIP, DINOv2, RT-DETR,
Grounding DINO, ClipSeg, SAM 2, Depth-Anything-V2, HED, Lineart, NormalBAE, UperNet-Seg, RMBG-1.4 — the last
two (NormalBAE, UperNet-Seg) required a real CUDA correctness bug fix along the way: `MaskRows`/`Transpose2D`
called with a `.Reshape` view as the write target orphans the CUDA activation-cache entry from the object the
caller reads back, producing silent all-zero output. Fixed in `NormalBaeModel.cs` (`SqueezeExcite`,
`RefineHead`) and `UperNetSegModel.cs` (`LayerNormChannelsFirst`, `MbConvBlock.Forward`) by allocating the
output already in the written shape and reshaping only on the way out.
**YOLO8, YOLO11 and YOLOv8-Face stay blocked** — not gated access, a distribution-format gap: HF hosts these only
as raw Ultralytics `.pt` (a full pickled `nn.Module` object graph the engine's safe-subset pickle VM can't resolve).
No ungated pre-folded-safetensors mirror was found (checked 2026-07-22). The architectures above are already
real-weight parity-verified — the gap is purely that the one-time offline `-folded.safetensors` conversion isn't
something the Assets auto-download system can perform; it needs a local Python step outside the engine. See
`src/HartsyInference.Cli/Infra/ModelCatalog.cs` for the per-model detail.

**ArcFace is not blocked on a loader.** It is verified in the table above, and the engine reads raw ONNX itself:
`OnnxWeightLoader` (`ModelAssets/Onnx`) parses initializers and the node list, with integer and float attributes,
without ONNX Runtime. ArcFace still goes through its offline conversion (`convert_arcface_onnx.py`); the audio side already
binds ONNX weights without it (CosyVoice2, g2pW, Silero, and Kim_Vocal_2 by walking the node list; see [MODEL_STATUS_AUDIO.md](MODEL_STATUS_AUDIO.md)).

## Runs end-to-end, parity pending (⚠️)

| Model | Status | Notes |
|---|---|---|
| **Real-ESRGAN** (x4plus, x2plus, x4plus-anime-6B — RRDBNet super-resolution) | ✅ x4plus / ⚠️ x2plus, anime-6B | x4plus matches BasicSR on real weights (2026-11-03, CPU: network max \|Δ\| 1.7e-6, every output byte identical, untiled). Seeded x4 and x2 fixtures pin the network, the x2 reflect pad and unshuffle in the CPU lane. x2plus and anime-6B have only the 2026-09-09 real-weight CPU smoke (expected sizes, sharper than bilinear); tiled output is not compared. ([details](#real-esrgan)) |
| **RMBG 3.0** (Bria; Fibo 2.0 edit that draws an alpha matte in 4 steps; `rmbg3`, `rmbg3-int8`) | ⚠️ CUDA cutouts ran; no parity, 12 GB not shown | Catalog, variant, matte step and vision modes are in; CPU unit tests pin the routing, defaults, shift and matte maths. A GPU run on an RTX 4090 at `2b32fc122` (2026-11-25) produced cutouts with `rmbg3-int8` on CUDA; one alpha is over-inclusive. A 12 GB fit is unlikely as built: a 0.5 s GPU-wide sampling of `main` `6548e968a` and of PR #1290 (RTX 4090, reported) peaked at 13,759–13,933 MiB against a 777–803 MiB baseline, about 12,956–13,156 MiB above it, and the earlier 2 s sampling is superseded. The memory is not yet attributed to a stage. No diffusers or ComfyUI reference comparison has run, so there is no parity claim. The BF16-bias error that failed the CPU path at `2b32fc122` is gone on `main` (`69e66e1b2`): the CPU run loads, but a CPU cutout has not finished within the 30-minute cap; the estimated time is 15–30 minutes, so no slowdown is shown. The Fibo 2.0 edit path has no real-weight parity yet. The int8 builds run float attention by default where ComfyUI runs `comfy_kitchen_int8`; INT8 attention is an opt-in CUDA knob (`numerics.int8AttnV2`), numerically unverified against Comfy (the loader warns). The int8 encoder must be the pinned Comfy-Org file; a wrong-hash file is refused. CC BY-NC 4.0. ([details](#rmbg-30)) |
| **DINOv3 ViT-H+/16** (1024 px encoder: RoPE with float32 periods, 4 storage tokens, LayerScale, SwiGLU MLP 5120 wide, 32 layers, 20 heads) | 🔬 encoder only | Real-weight encoder parity on the pinned Comfy-Org repack `dino_v3_vit_h_bf16.safetensors` against Hugging Face `DINOv3ViTModel` (2026-11-17, CPU FP32, knight input at 1024 px, 4101 tokens): after the final norm relRMS 1.2e-5, peak-relative 5.0e-5. The consumer is Idol (image to 3D Gaussians, in review). Weights are under the DINOv3 License (Meta, custom; not MIT or Apache). See [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). |
| **DINOv3 ViT-L/16** (448 px encoder: RoPE, 4 storage tokens, LayerScale, GELU MLP) | 🔬 encoder only | Real-weight encoder parity on the pinned `facebook/dinov3-vitl16-pretrain-lvd1689m` (2026-11-07, CPU FP32): peak-relative error ≤1.4e-6 at every stage (final norm: max abs 2.2e-5, rms 5.6e-7). No consumer in this change (HairDeck conditioning and GPU paths are not built). ViT-S/B are config-only. Weights are under the DINOv3 License (Meta, custom; not MIT or Apache). See [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). |

## Deferred (❌)

**EVA-CLIP** (EVA-02 vision tower: RoPE + SwiGLU + sub-LN — preset added, blocked on tower support), AM-RADIO.
*(Done 2026-07-17, moved out of deferred: YOLO-seg head validated; SigLIP 2 + MetaCLIP presets are drop-ins;
GPU-native MaxPool2D + depthwise-Conv kernels shipped with CUDA PTX + GPU-parity gate.)*
See the [Remaining work](#remaining-work) section for the stretch list. **Segmentation ControlNet
preprocessing** shipped 2026-07-16 via UperNet-ConvNeXt-Small (see ✅ table) — UniFormer / OneFormer
variants remain deferred (custom-mmcv fork / 1GB+ checkpoints) with no quality need while UperNet matches
the diffusers reference exactly.

## Notes

CLIP and SigLIP **vision towers used inside diffusion and VLMs** are validated separately (SigLIP tower
corr 1.0 for Gemma-3 / SmolVLM); see [MODEL_STATUS_LLM.md](MODEL_STATUS_LLM.md). The standalone Vision
package wraps the existing `ClipVisionEncoder` rather than duplicating the math.

## Remaining work

Distilled from the retired PHASE_6_VISION plan. Items now ✅ above (SAM 2 / MobileSAM, RT-DETR,
Grounding DINO, YOLO-Face, YOLO-seg, SigLIP 2) are omitted.
See [ROADMAP.md](ROADMAP.md) for cross-cutting infra (multi-GPU, kernel perf, quant, serving).

### Open Vision Banana (generative perception)

Built as an image-recipe variant (see [MODEL_STATUS_IMAGE.md](MODEL_STATUS_IMAGE.md#open-vision-banana)). `OvbPerception` runs it through the image service and maps the painted answers onto `VisionResult`: segmentation (one mask per class plus the painted picture), relative inverse depth, and unit normals. `hartsy vision -m open-vision-banana-4b --mode segment|depth|normal` routes there. The 4B has one real-weight run on the RTX 3060 (2026-11-24, details in MODEL_STATUS_IMAGE.md).

### Deferred stretch models
- [x] DINOv3 ViT-L/16 encoder with RoPE at 448 px (encoder parity in the table above; consumers still open).
- [ ] EVA-CLIP / EVA-02 vision tower (RoPE + SwiGLU + sub-LN).
- [ ] AM-RADIO; MetaCLIP presets already exist (recorded above).

### Decoders
- [ ] JPEG / WebP decoders.

### Testing gaps
- [ ] CLIP bit-exact Python parity.
- [ ] Additional YOLO variants beyond the recorded validated paths.

## Details

Verification evidence, bugs found, and caveats for the rows above. Moved out of the status
tables on 2026-08-06 so the tables stay scannable — no content was dropped.

### Depth-Anything-V2

Real-weight parity vs the OFFICIAL DepthAnything/Depth-Anything-V2 implementation on a fixed 280×378 input (non-square → exercises pos-embed interpolation), CPU backend (2026-07-16): **ViT-S depth avg err 2.1e-6, ViT-L 4.5e-5 (rel 2.9e-7)**; all 17 tapped stages ≤ 4.5e-5 (test `DepthAnythingParityTests`, oracle `dump_depth_anything.py` + `diff_depth_anything.py`). Reuses the shared `Dinov2VisionEncoder` (new `EncodeIntermediates` taps 4 blocks + final norm; dinov2-exact bicubic pos-embed interpolation for non-native grids) + new `DptHead` (reassemble → fusion w/ residual conv units → depth convs). Root-cause fix along the way: DINOv2 MLPs are **exact-erf GELU** — the backend tanh `Gelu` drifted feats ~5e-3 rel over 24 layers → new `IBackend.GeluErf` (host default + `gelu_erf_f32` PTX, CUDA-verified 4.4e-7). Preprocessor matches the official cv2 INTER_CUBIC lower-bound-518/multiple-of-14 transform (avg err 1.1e-5); postprocess = align-corners bilinear back-resize + min-max → [0,1] grayscale (the ComfyUI/ControlNet conditioning form). ViT-B fits via `DepthAnythingPreset.Base` (config-only, unverified). Ckpts: HF `depth-anything/Depth-Anything-V2-{Small,Large}` `.pth`, loaded natively via `PytorchPickleLoader` + in-loader hub→HF key remap w/ fused-qkv split. Files: `src/HartsyInference.Vision/DepthAnything/`.

### HED / SoftEdge

Real-weight parity vs controlnet_aux `HEDdetector` on a fixed 320×256 image, CPU backend (2026-07-16): **fused edge avg err 3.9e-8**, all 5 side logits ≤ 1.1e-5; e2e uint8 forms code-exact (softedge mismatch 1.2e-5, safe 0, scribble 5.9e-4 — the scribble path re-implements cv2 GaussianBlur (auto-ksize 25/19, reflect-101) + 4-direction line NMS + binarize in C#). Test `HedParityTests`, oracle `dump_hed.py` + `diff_hed.py`. VGG16-style 5-block trunk w/ 1×1 side projections, bilinear fuse + mean + sigmoid; input is raw [0,255] RGB minus a learned shift. Ckpt: HF `lllyasviel/Annotators/ControlNetHED.pth` (sha256 `5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa`), loaded via `PytorchPickleLoader`. Files: `src/HartsyInference.Vision/Annotators/Hed*`.

### Lineart

Real-weight parity vs controlnet_aux `LineartDetector` on a fixed 320×256 image, CPU backend (2026-07-16): **line avg err 1.6e-8 (realistic) / 3.8e-8 (coarse)**, all stages ≤ 2.1e-6; e2e inverted uint8 conditioning map **bit-exact** (mismatch 0). Test `LineartParityTests`, oracle `dump_lineart.py` + `diff_lineart.py`. Compact ResNet UNet (reflection-pad 7×7 stem, 2× stride-2 down, 3 res blocks, 2× ConvTranspose up, sigmoid head); affine-free InstanceNorm realized as per-channel `GroupNorm`; the detector's `255 − x` inversion (white lines on black — the CN conditioning form) lives in `LineartPreprocessor`. Ckpts: HF `lllyasviel/Annotators/sk_model.pth` (sha256 `c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6`) / `sk_model2.pth` (sha256 `30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0`). Files: `src/HartsyInference.Vision/Annotators/Lineart*`.

### NormalBAE

Real-weight parity vs controlnet_aux `NormalBaeDetector` on a fixed 320×256 image, CPU backend (2026-07-16): **out_res1 avg err 7.9e-6**, all 14 stages ≤ 8.9e-6, e2e uint8 normal-RGB mismatch 1.5e-4. Test `NormalBaeParityTests`, oracle `dump_normalbae.py` + `diff_normalbae.py`. Full tf_efficientnet_b5_ap encoder port (39 MBConv/DS blocks w/ squeeze-excitation, TF SAME asymmetric padding on stride-2 depthwise convs, BatchNorm eps 1e-3 folded into convs at load) + NNET decoder (conv/BN up pyramid + dense test-mode per-pixel 1×1-conv MLP refinement at 1/4, 1/2, 1/1 + `norm_normalize`). Training-time uncertainty-guided sparse sampling intentionally not ported. Ckpt: HF `lllyasviel/Annotators/scannet.pt` (sha256 `03dbf1600c51ee3d45c29f77b77bf1a3b7a24c3452dba62a4ae658f37330c209`). Files: `src/HartsyInference.Vision/Annotators/NormalBae*`.

### UperNet-Seg

Real-weight parity vs transformers `UperNetForSemanticSegmentation` (`openmmlab/upernet-convnext-small` — the diffusers-documented annotator for `control_v11p_sd15_seg`) on a real 512² image (bus.png), CPU backend (2026-07-16): all 7 tapped stages at float noise (**feat_0..3 ≤ 6.8e-7, psp_out 7.0e-7, fpn_out 1.3e-6, logits_q 5.7e-6** vs ref mean abs 15.3), **per-pixel argmax class agreement 100.000%** (0/262144; ≥99% is the documented interpolation-boundary tolerance), ADE20K palette **byte-exact** vs controlnet_aux `ade_palette()`. Test `UperNetSegParityTests`, oracle `dump_upernet_seg.py` + `ade20k_palette.json`. ConvNeXt-Small trunk (4×4/4 stem, depths 3/3/27/3, dims 96/192/384/768; 7×7 depthwise + channels-last LN + 4× erf-GELU MLP, layer-scale folded into pwconv2 at load) + UperNet head (PSP scales 1/2/3/6 + FPN @512ch, BatchNorm eps 1e-5 folded via the shared `ConvBnFold`), logits bilinearly upsampled align_corners=False. Chosen over comfyui_controlnet_aux's UniFormer (needs the bundled custom-mmcv fork — not runnable in a plain venv) and OneFormer (1GB+ ckpts, deformable-attn transformer decoder). Ckpt: HF `openmmlab/upernet-convnext-small/pytorch_model.bin` (327MB F32, sha256 `76c163aa531ab7edfb3a77bbcc039e340645aa0ffe2b0ffcfc68755f550c76ea`), loaded via `PytorchPickleLoader`; auxiliary FCN head (training-only) skipped. Files: `src/HartsyInference.Vision/Annotators/UperNetSeg*` + `Ade20kPalette`.

### RMBG-1.4

Real-weight `briaai/RMBG-1.4` foreground mask on the RTX 4090 (2026-07-01). Full U²-Net-style nested-U (`conv_in` + 6 encoder + 5 decoder RSU blocks + `side1`) at 1024²; mask **maxAbs 2.9e-6, corr 1.00000000** vs the upstream model (test `RmbgParityTests`, oracle `dump_rmbg.py`). Clean chair segmentation confirmed visually. **BatchNorm folded into the conv at load; dilated convs realized as zero-inflated 3×3 kernels** (no dilation kernel needed); host bilinear-2× upsample (no backend op exists for it). `RmbgBackgroundRemover` (preprocess + alpha + gray-0.5 composite) is the pure-C# replacement for the Python `rembg` step the **image→3D pipelines** (TripoSR / Hunyuan3D) need — see PHASE_11 §6. Since alpha.56 `VisionMode.BackgroundRemoval` also returns the matte as `ImageData.Alpha` (8-bit, straight), so consumers that want a cutout rather than the gray composite get one from the same forward pass. Since alpha.65 `ImageRequest.RemoveBackground` drives the same net as an image-generation post step (`BackgroundRemovalStage`, after the refiner and segment passes) and returns the matte as `ImageResult.Alpha` over untouched RGB — that is what Swarm's Remove Background param maps to. Files: `src/HartsyInference.Vision/Rmbg/`.

### RMBG 3.0

CUDA cutouts ran (2026-11-25 and the 2026-11-26 rerun on `main`, records below); no parity claim. `rmbg3` (bf16) and `rmbg3-int8` are Vision catalog entries on Bria's ungated `briaai/RMBG-3.0-Comfy` repack; the Fibo 2.0 encoder and VAE are shared. `VisionMode.BackgroundRemoval` and `VisionMode.Matting` for these ids run the Fibo 2.0 edit path through the image service (the fixed instruction, 4 steps, no guidance, the trained size for the source aspect, the diffusers dynamic sigma shift), then turn the picture into 8-bit alpha at the source size with PIL's luma and Lanczos (`Rmbg3Matte`). Files: `src/HartsyInference.Engine/Features/Rmbg3Matte.cs`, `Recipes/Image/Fibo2Variants.cs` and `Fibo2Recipe.cs`. Pins, licence and the reference differences: [RMBG3_ARCHITECTURE.md](../Research/RMBG3_ARCHITECTURE.md).

**Run record, 2026-11-24 (before the fix):** all runs failed with no cutout. `94fa9b557` failed on the U8 `comfy_attention.config` tensor on CUDA and CPU; `#1132` head `e5194f015` failed on CUDA with an I8→F32 conversion in the Qwen3-VL vision tower.

**Run record, 2026-11-25 (GPU test session, RTX 4090, commit `2b32fc122`, Release CLI, reported by that session; evidence on the server under `/home/hartsy/evidence/gpu-validation-2026-11-25/v10/3-rmbg/`, not re-verified here):** `vision <photo> --model rmbg3-int8 --mode removebg --backend cuda`, rc 0 on all four photos. 000000032887 (640×480): 207 s including the text-encoder download, alpha > 0 on 9.5 % of pixels, peak 12,746 MiB. 000000054931 (427×640): 26 s, 20.8 %, 12,873 MiB. 000000065350 (640×427): 48 s, 9.0 %, 13,382 MiB. 000000084362 (640×480): 52 s, 39.8 %, 12,769 MiB; the alpha is over-inclusive, covering the sofa and cushions around the child. Matting on 000000032887 (`vision <photo> --model rmbg3-int8 --mode matting --backend cuda`): 24 s, peak 13,761 MiB. The 2026-11-25 runs used default knobs (as reported). Memory is GPU-wide, sampled every 2 s by `nvidia-smi`, with 795 MiB in use before load. Two runs peaked above 12 GiB after that baseline was subtracted: the matting run (13,761 − 795 = 12,966 MiB) and the cutout on 000000065350 (13,382 − 795 = 12,587 MiB). Weights: `diffusion_models/rmbg3_int8_convrot.safetensors` (8,408,562,474 B) and `text_encoders/fibo2/comfy-org/qwen3vl_4b_int8_convrot.safetensors` (4,858,559,240 B); the VAE was not logged. CPU at the same commit: rc 143 (SIGTERM, likely the memory cap) at a 28 GB cap, then rc 1 at 48 GB with `LinearTransB: the CPU backend is F32-only (bias=BF16)`.

**Run record, 2026-11-26 (rerun on `main` `69e66e1b2`, Release net10.0 CLI, RTX 4090; reported by the GPU test session; outputs on the server under `/home/hartsy/evidence/gpu-validation-2026-11-26-rmbg/`, not re-verified here):** `vision <photo> --model rmbg3-int8 --mode removebg --backend cuda|cpu -o <outdir>`. CUDA, rc 0 on all four photos: 000000032887 64 s, peak 12,748 MiB; 000000054931 19 s, 12,858 MiB; 000000065350 20 s, 12,861 MiB; 000000084362 18 s, 12,782 MiB (GPU-wide, including the 795 MiB baseline). These peaks were sampled every 2 s and are superseded: they read just under 12 GiB only after subtracting the baseline, and the 0.5 s sampling in the row above does not support that. The matting run was not rerun. CPU on 000000032887 loaded, then hit the 30-minute cap (rc 124) with no output; the other CPU photos were not run. No reference comparison and no 12 GB test.

**CPU cutout: not finished; no slowdown shown.** The 30-minute cap was reached with no output. The expected time is an estimate, not a measurement. The edit path runs at about 1 MP, which gives about 4,096 noisy latent tokens, about 4,096 clean source-image tokens and the prompt: about 8,000–8,500 tokens in all. For the 8B denoiser over 4 steps that is about 5×10¹⁴ FLOP before attention. At a practical 0.3–0.6 TFLOPS on a CPU of unknown size, that is about 15–30 minutes, so the cap is within plain compute time. Two code facts could add time, not measured: the CPU `LinearTransB` converts each non-F32 weight to F32 on every call (`MatMulKernels.cs:123`), and the int8 DiT is widened to F16 on the CPU (`Fibo2Recipe.cs:122`). A rerun with a longer cap (about 90 minutes) and per-stage timing (text encoder, each denoiser step, VAE) is needed. Until then CPU is not a supported backend for RMBG 3.0.

**Open:** real-weight parity against diffusers on saved noise (no reference run yet); a 12 GB fit, untested (whether 12 GB holds the int8 transformer after the encoder is released); a CPU cutout that finishes, with the per-stage timing above; the `ImageRequest.RemoveBackgroundModel` choice (`rmbg3` as the post-generation matter, RMBG-1.4 stays the default; CLI `--remove-background-model`) releases the generating pipeline first and is untested on a GPU. Not included: `briaai/VRMBG-3.0`, the video model, whose weights need a commercial agreement.

### Real-ESRGAN

Reachable since alpha.56 through `VisionMode.Upscale` (`VisionService.Upscale`) and the catalog ids
`real-esrgan-x4plus` / `real-esrgan-x2plus` / `real-esrgan-anime6b`, which fetch the official BasicSR `.pth`
files from HuggingFace mirrors (hashes cross-checked across independent uploads; x2plus is a single mirror whose
size matches the v0.2.1 release). `RealEsrganConverter` strips the `params_ema` envelope and infers the RRDBNet
geometry from the keys. **Format trap (alpha.57):** BasicSR's 2× model is the ×4 network with a 2× `pixel_unshuffle`
in front (`conv_first` takes 12 channels) — every checkpoint has both `conv_up` stages, so the factor must be read
from `conv_first`'s input width, not from the presence of `conv_up2`. `UpscalePipeline` unshuffles on the host
(torch channel order `c·r² + i·r + j`) before tiling, reflect-padding an odd side to even as RealESRGANer does (until
2026-11-03 it repeated the edge pixel). Tiled at 256 px input (`VisionService.UpscaleTileSize`); a target size runs at most two
passes then resizes down (`UpscalePlan`).

**Parity (2026-11-03).** `RealEsrganParityTests` against `tests/python-reference/real_esrgan_reference.py`, a float64
numpy transcription of BasicSR's `RRDBNet` and RealESRGANer's pre/post-processing at the official CLI defaults (pre_pad 0,
untiled). Real weights (Comfy-Org/Real-ESRGAN @5fd49b7b27, `RealESRGAN_x4plus.safetensors`, 16×12 crop): network max
\|Δ\| 1.7e-6, pipeline bytes all identical. Seeded tiny x4 and x2 networks committed under `TestData/RealEsrgan` run in the
CPU lane (max \|Δ\| 1.0e-7, bytes identical; the x2 case has an odd width, so it checks the pad and unshuffle). **Open:**
x2plus and anime-6B on real weights; tiled output (the engine ramp-blends overlapping tiles, RealESRGANer crops `tile_pad`
margins, so the two differ near seams by design).

### ArcFace IR-50

Real-weight parity vs onnxruntime on buffalo_l `w600k_r50.onnx` (2026-07-16): **512-d embedding cosine 1.000000, maxAbs ≤ 6.4e-6** on random + 2 real portrait crops; per-stage stem/layer1–4/emb all corr 1.0 (`ArcFaceParityTests` in `tests/HartsyInference.Vision.Tests/Parity/`, env `ARCFACE_WEIGHTS`/`ARCFACE_REF`; the per-stage debug test was removed 2026-08-06). Converter `tests/python-reference/convert_arcface_onnx.py` (ONNX ships block bn2/bn3/downsample-BN pre-folded; it folds pre-fc `bn2` + post-fc `features` BN1d into `fc`, emits per-block `bn1` as scale/shift applied via depthwise-1×1 `Conv2dDepthwise`). Block counts derived from keys so IR-100 loads too. `FaceAlignment` = ArcFace 112×112 template + least-squares similarity (Umeyama, non-reflective) from YOLO11-pose eyes+nose (3-point; SCRFD's mouth corners unavailable from COCO-17 — documented deviation) + bilinear `WarpAffine`; unit-tested (`FaceAlignmentTests`). Consumed by IP-Adapter FaceID (see PARITY_VERIFICATION image section). Files: `src/HartsyInference.Vision/Face/`.

### RT-DETR

**Real-weight VERIFIED (2026-07-17)** vs transformers `RTDetrForObjectDetection` on bus.png: top-detection **box IoU 0.9996**, score **corr 1.0000**, all 5 detections matched (bus + 4 persons), exact labels. Full faithful port replacing the earlier invented-key stand-in (0/526 keys had matched): **ResNet-18vd** backbone (deep 3-conv stem + vd avg-pool downsample), **hybrid encoder** (AIFI with separate q/k/v + 2D sincos pos on Q/K + float64 freq arithmetic + GELU-erf FFN; CCFM CSPRepLayer/RepVGG fpn/pan), **two-stage query selection**, **HF multi-scale deformable decoder** (bilinear grid_sample, align_corners=False, iterative box refine, per-layer heads). `RtDetrCheckpointConverter` folds all three unfolded-BN naming conventions into conv-with-bias; all 526 keys map. Env-gated `RealWeights_Parity_BusImage` (`RTDETR_PATH`) passes. CPU-parity port (host-loop deformable sampler); GPU-kernel routing is a follow-up. Files: `src/HartsyInference.Vision/Detection/RtDetr*`.

### Grounding DINO

**Real-weight VERIFIED end-to-end (2026-07-17)** vs transformers `GroundingDinoForObjectDetection` on the COCO cats image + `"a cat. a remote control."` (CPU backend). Full faithful port replacing the earlier structural stand-in: **Swin-T backbone** (shifted-window MSA + relative-position bias + patch merging; 3 feature maps corr **1.000000**), **neck** (input_proj 1×1/3×3 convs + GroupNorm + DETR sine positions + level embed; corr **1.0**), **BERT-base text tower** (special-token block self-attention mask + reset position ids + text_projection; `text_features` corr **1.000000**), **cross-modality enhancer encoder** (6× MS-deformable image self-attn with bilinear grid_sample + bi-directional image↔text fusion + text self-attn enhancer; `encoder_vision` corr **0.99948**, `encoder_text` **0.99899**), **two-stage query selection**, **6-layer decoder** (self / text-cross / deformable-cross attn + iterative box refinement), contrastive class head + shared box head, and the sigmoid/threshold/phrase-decode post-process. End-to-end from raw `pixel_values`+`input_ids`: all 3 oracle detections reproduced — **a cat IoU 0.9990, a cat IoU 0.9992, a remote control IoU 0.9941**, exact labels, scores within ~0.03. Along the way **BERT was hoisted** from `HartsyInference.Audio` into `HartsyInference.ModelAssets.TextEncoders.Bert` (shared by Audio + Vision; self-contained, IBackend-only, with an added position-id + additive-mask overload); Audio consumers updated. Env-gated `GroundingDinoRealWeightParityTests` (`GROUNDING_DINO_PATH` = `model.safetensors`, `GROUNDING_DINO_REF` = `oracle.py` dumps) validates every stage boundary. Encoder/backbone run as host loops (CPU e2e ≈ 11 min); GPU-kernel routing is a follow-up. Files: `src/HartsyInference.Vision/Detection/GroundingDino/`.

### SAM 2 / MobileSAM

**Real-weight VERIFIED (2026-07-17, `facebook/sam2-hiera-tiny`):** Hiera windowed-attn encoder + two-way mask decoder built; best-mask **IoU 0.9972** vs HF `Sam2Model`, image-embed **corr 0.999**. Bringup fixed 7 real arch bugs (mask-decoder `mlp.layers` keys, `global_att_blocks`, bicubic pos-embed, FPN nearest top-down, high-res feature path, `obj_score_token` off-by-one, dense no-mask embed + IoU sigmoid). `Sam2RealWeightParityTests` (env-gated `SAM2_CHECKPOINT_PATH`+`SAM2_REF_DIR`) passes; seg-overlay artifact produced.

### Face detection / landmarks

**Real-weight VERIFIED (2026-07-17, `akanametov/yolo-face` widerface-Pose):** YOLOv8-Face detector (v8 trunk + `cv4` 5-pt landmark branch) → box **IoU 0.9999**, landmark **mean 0.025 px / max 0.042 px** vs ultralytics; detector → `FaceAlignment` → `ArcFace` **512-d L2-norm 1.0**. Reuses the verified YOLO stack; guessed key layout was correct. Env-gated `YOLOV8_FACE_PATH`+`ARCFACE_WEIGHTS_PATH` integration test passes.
