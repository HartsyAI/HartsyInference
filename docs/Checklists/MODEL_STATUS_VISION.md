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
| **Depth-Anything-V2** (ViT-S / ViT-L, relative depth) | ✅ | Real-weight parity vs the OFFICIAL DepthAnything/Depth-Anything-V2 implementation on a fixed 280×378 input (non-square → exercises pos-embed interpolation), CPU backend (2026-07-16): **ViT-S depth avg err 2.1e-6, ViT-L 4.5e-5 (rel 2.9e-7)**; all 17 tapped stages ≤ 4.5e-5 (test `DepthAnythingParityTests`, oracle `dump_depth_anything.py` + `diff_depth_anything.py`). Reuses the shared `Dinov2VisionEncoder` (new `EncodeIntermediates` taps 4 blocks + final norm; dinov2-exact bicubic pos-embed interpolation for non-native grids) + new `DptHead` (reassemble → fusion w/ residual conv units → depth convs). Root-cause fix along the way: DINOv2 MLPs are **exact-erf GELU** — the backend tanh `Gelu` drifted feats ~5e-3 rel over 24 layers → new `IBackend.GeluErf` (host default + `gelu_erf_f32` PTX, CUDA-verified 4.4e-7). Preprocessor matches the official cv2 INTER_CUBIC lower-bound-518/multiple-of-14 transform (avg err 1.1e-5); postprocess = align-corners bilinear back-resize + min-max → [0,1] grayscale (the ComfyUI/ControlNet conditioning form). ViT-B fits via `DepthAnythingPreset.Base` (config-only, unverified). Ckpts: HF `depth-anything/Depth-Anything-V2-{Small,Large}` `.pth`, loaded natively via `PytorchPickleLoader` + in-loader hub→HF key remap w/ fused-qkv split. Files: `src/HartsyInference.Vision/DepthAnything/`. |
| **HED / SoftEdge** (ControlNetHED, softedge_hed + scribble_hed) | ✅ | Real-weight parity vs controlnet_aux `HEDdetector` on a fixed 320×256 image, CPU backend (2026-07-16): **fused edge avg err 3.9e-8**, all 5 side logits ≤ 1.1e-5; e2e uint8 forms code-exact (softedge mismatch 1.2e-5, safe 0, scribble 5.9e-4 — the scribble path re-implements cv2 GaussianBlur (auto-ksize 25/19, reflect-101) + 4-direction line NMS + binarize in C#). Test `HedParityTests`, oracle `dump_hed.py` + `diff_hed.py`. VGG16-style 5-block trunk w/ 1×1 side projections, bilinear fuse + mean + sigmoid; input is raw [0,255] RGB minus a learned shift. Ckpt: HF `lllyasviel/Annotators/ControlNetHED.pth` (sha256 `5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa`), loaded via `PytorchPickleLoader`. Files: `src/HartsyInference.Vision/Annotators/Hed*`. |
| **Lineart** (sk_model realistic + sk_model2 coarse) | ✅ | Real-weight parity vs controlnet_aux `LineartDetector` on a fixed 320×256 image, CPU backend (2026-07-16): **line avg err 1.6e-8 (realistic) / 3.8e-8 (coarse)**, all stages ≤ 2.1e-6; e2e inverted uint8 conditioning map **bit-exact** (mismatch 0). Test `LineartParityTests`, oracle `dump_lineart.py` + `diff_lineart.py`. Compact ResNet UNet (reflection-pad 7×7 stem, 2× stride-2 down, 3 res blocks, 2× ConvTranspose up, sigmoid head); affine-free InstanceNorm realized as per-channel `GroupNorm`; the detector's `255 − x` inversion (white lines on black — the CN conditioning form) lives in `LineartPreprocessor`. Ckpts: HF `lllyasviel/Annotators/sk_model.pth` (sha256 `c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6`) / `sk_model2.pth` (sha256 `30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0`). Files: `src/HartsyInference.Vision/Annotators/Lineart*`. |
| **NormalBAE** (surface normals, scannet NNET) | ✅ | Real-weight parity vs controlnet_aux `NormalBaeDetector` on a fixed 320×256 image, CPU backend (2026-07-16): **out_res1 avg err 7.9e-6**, all 14 stages ≤ 8.9e-6, e2e uint8 normal-RGB mismatch 1.5e-4. Test `NormalBaeParityTests`, oracle `dump_normalbae.py` + `diff_normalbae.py`. Full tf_efficientnet_b5_ap encoder port (39 MBConv/DS blocks w/ squeeze-excitation, TF SAME asymmetric padding on stride-2 depthwise convs, BatchNorm eps 1e-3 folded into convs at load) + NNET decoder (conv/BN up pyramid + dense test-mode per-pixel 1×1-conv MLP refinement at 1/4, 1/2, 1/1 + `norm_normalize`). Training-time uncertainty-guided sparse sampling intentionally not ported. Ckpt: HF `lllyasviel/Annotators/scannet.pt` (sha256 `03dbf1600c51ee3d45c29f77b77bf1a3b7a24c3452dba62a4ae658f37330c209`). Files: `src/HartsyInference.Vision/Annotators/NormalBae*`. |
| **UperNet-Seg** (semantic segmentation, ADE20K 150-class, ConvNeXt-Small) | ✅ | Real-weight parity vs transformers `UperNetForSemanticSegmentation` (`openmmlab/upernet-convnext-small` — the diffusers-documented annotator for `control_v11p_sd15_seg`) on a real 512² image (bus.png), CPU backend (2026-07-16): all 7 tapped stages at float noise (**feat_0..3 ≤ 6.8e-7, psp_out 7.0e-7, fpn_out 1.3e-6, logits_q 5.7e-6** vs ref mean abs 15.3), **per-pixel argmax class agreement 100.000%** (0/262144; ≥99% is the documented interpolation-boundary tolerance), ADE20K palette **byte-exact** vs controlnet_aux `ade_palette()`. Test `UperNetSegParityTests`, oracle `dump_upernet_seg.py` + `ade20k_palette.json`. ConvNeXt-Small trunk (4×4/4 stem, depths 3/3/27/3, dims 96/192/384/768; 7×7 depthwise + channels-last LN + 4× erf-GELU MLP, layer-scale folded into pwconv2 at load) + UperNet head (PSP scales 1/2/3/6 + FPN @512ch, BatchNorm eps 1e-5 folded via the shared `ConvBnFold`), logits bilinearly upsampled align_corners=False. Chosen over comfyui_controlnet_aux's UniFormer (needs the bundled custom-mmcv fork — not runnable in a plain venv) and OneFormer (1GB+ ckpts, deformable-attn transformer decoder). Ckpt: HF `openmmlab/upernet-convnext-small/pytorch_model.bin` (327MB F32, sha256 `76c163aa531ab7edfb3a77bbcc039e340645aa0ffe2b0ffcfc68755f550c76ea`), loaded via `PytorchPickleLoader`; auxiliary FCN head (training-only) skipped. Files: `src/HartsyInference.Vision/Annotators/UperNetSeg*` + `Ade20kPalette`. |
| **RMBG-1.4** (BriaRMBG / ISNet — background removal) | ✅ | Real-weight `briaai/RMBG-1.4` foreground mask on the RTX 4090 (2026-07-01). Full U²-Net-style nested-U (`conv_in` + 6 encoder + 5 decoder RSU blocks + `side1`) at 1024²; mask **maxAbs 2.9e-6, corr 1.00000000** vs the upstream model (test `RmbgParityTests`, oracle `dump_rmbg.py`). Clean chair segmentation confirmed visually. **BatchNorm folded into the conv at load; dilated convs realized as zero-inflated 3×3 kernels** (no dilation kernel needed); host bilinear-2× upsample (CUDA has no `UpsampleBilinear2D`). `RmbgBackgroundRemover` (preprocess + alpha + gray-0.5 composite) is the pure-C# replacement for the Python `rembg` step the **image→3D pipelines** (TripoSR / Hunyuan3D) need — see PHASE_11 §6. Files: `src/HartsyInference.Vision/Rmbg/`. |
| **ArcFace IR-50** (InsightFace w600k_r50, face embedding) + **FaceAlignment** | ✅ | Real-weight parity vs onnxruntime on buffalo_l `w600k_r50.onnx` (2026-07-16): **512-d embedding cosine 1.000000, maxAbs ≤ 6.4e-6** on random + 2 real portrait crops; per-stage stem/layer1–4/emb all corr 1.0 (`ArcFaceParityTests` in `tests/HartsyInference.Vision.Tests/Parity/`, env `ARCFACE_WEIGHTS`/`ARCFACE_REF`; the per-stage debug test was removed 2026-08-06). Converter `tests/python-reference/convert_arcface_onnx.py` (ONNX ships block bn2/bn3/downsample-BN pre-folded; it folds pre-fc `bn2` + post-fc `features` BN1d into `fc`, emits per-block `bn1` as scale/shift applied via depthwise-1×1 `Conv2dDepthwise`). Block counts derived from keys so IR-100 loads too. `FaceAlignment` = ArcFace 112×112 template + least-squares similarity (Umeyama, non-reflective) from YOLO11-pose eyes+nose (3-point; SCRFD's mouth corners unavailable from COCO-17 — documented deviation) + bilinear `WarpAffine`; unit-tested (`FaceAlignmentTests`). Consumed by IP-Adapter FaceID (see PARITY_VERIFICATION image section). Files: `src/HartsyInference.Vision/Face/`. |

## Detection / segmentation — real-weight verified (✅, 2026-07-17)

| Model | Status | Notes |
|---|---|---|
| **RT-DETR** (`PekingU/rtdetr_r18vd`) | ✅ | **Real-weight VERIFIED (2026-07-17)** vs transformers `RTDetrForObjectDetection` on bus.png: top-detection **box IoU 0.9996**, score **corr 1.0000**, all 5 detections matched (bus + 4 persons), exact labels. Full faithful port replacing the earlier invented-key stand-in (0/526 keys had matched): **ResNet-18vd** backbone (deep 3-conv stem + vd avg-pool downsample), **hybrid encoder** (AIFI with separate q/k/v + 2D sincos pos on Q/K + float64 freq arithmetic + GELU-erf FFN; CCFM CSPRepLayer/RepVGG fpn/pan), **two-stage query selection**, **HF multi-scale deformable decoder** (bilinear grid_sample, align_corners=False, iterative box refine, per-layer heads). `RtDetrCheckpointConverter` folds all three unfolded-BN naming conventions into conv-with-bias; all 526 keys map. Env-gated `RealWeights_Parity_BusImage` (`RTDETR_PATH`) passes. CPU-parity port (host-loop deformable sampler); GPU-kernel routing is a follow-up. Files: `src/HartsyInference.Vision/Detection/RtDetr*`. |
| **Grounding DINO** (open-vocab detection, `IDEA-Research/grounding-dino-tiny`) | ✅ | **Real-weight VERIFIED end-to-end (2026-07-17)** vs transformers `GroundingDinoForObjectDetection` on the COCO cats image + `"a cat. a remote control."` (CPU backend). Full faithful port replacing the earlier structural stand-in: **Swin-T backbone** (shifted-window MSA + relative-position bias + patch merging; 3 feature maps corr **1.000000**), **neck** (input_proj 1×1/3×3 convs + GroupNorm + DETR sine positions + level embed; corr **1.0**), **BERT-base text tower** (special-token block self-attention mask + reset position ids + text_projection; `text_features` corr **1.000000**), **cross-modality enhancer encoder** (6× MS-deformable image self-attn with bilinear grid_sample + bi-directional image↔text fusion + text self-attn enhancer; `encoder_vision` corr **0.99948**, `encoder_text` **0.99899**), **two-stage query selection**, **6-layer decoder** (self / text-cross / deformable-cross attn + iterative box refinement), contrastive class head + shared box head, and the sigmoid/threshold/phrase-decode post-process. End-to-end from raw `pixel_values`+`input_ids`: all 3 oracle detections reproduced — **a cat IoU 0.9990, a cat IoU 0.9992, a remote control IoU 0.9941**, exact labels, scores within ~0.03. Along the way **BERT was hoisted** from `HartsyInference.Audio` into `HartsyInference.ModelAssets.TextEncoders.Bert` (shared by Audio + Vision; self-contained, IBackend-only, with an added position-id + additive-mask overload); Audio consumers updated. Env-gated `GroundingDinoRealWeightParityTests` (`GROUNDING_DINO_PATH` = `model.safetensors`, `GROUNDING_DINO_REF` = `oracle.py` dumps) validates every stage boundary. Encoder/backbone run as host loops (CPU e2e ≈ 11 min); GPU-kernel routing is a follow-up. Files: `src/HartsyInference.Vision/Detection/GroundingDino/`. |
| **SAM 2 / MobileSAM** | ✅ | **Real-weight VERIFIED (2026-07-17, `facebook/sam2-hiera-tiny`):** Hiera windowed-attn encoder + two-way mask decoder built; best-mask **IoU 0.9972** vs HF `Sam2Model`, image-embed **corr 0.999**. Bringup fixed 7 real arch bugs (mask-decoder `mlp.layers` keys, `global_att_blocks`, bicubic pos-embed, FPN nearest top-down, high-res feature path, `obj_score_token` off-by-one, dense no-mask embed + IoU sigmoid). `Sam2RealWeightParityTests` (env-gated `SAM2_CHECKPOINT_PATH`+`SAM2_REF_DIR`) passes; seg-overlay artifact produced. |
| **Face detection / landmarks** | ✅ | **Real-weight VERIFIED (2026-07-17, `akanametov/yolo-face` widerface-Pose):** YOLOv8-Face detector (v8 trunk + `cv4` 5-pt landmark branch) → box **IoU 0.9999**, landmark **mean 0.025 px / max 0.042 px** vs ultralytics; detector → `FaceAlignment` → `ArcFace` **512-d L2-norm 1.0**. Reuses the verified YOLO stack; guessed key layout was correct. Env-gated `YOLOV8_FACE_PATH`+`ARCFACE_WEIGHTS_PATH` integration test passes. |

**CLI catalog auto-download gap (2026-07-22):** `hartsy vision` wiring covers 13/17 real-weight-verified models
end-to-end (catalog `Assets` + auto-download + CLI-verified adversarially): CLIP, SigLIP, DINOv2, RT-DETR,
Grounding DINO, ClipSeg, SAM 2, Depth-Anything-V2, HED, Lineart, NormalBAE, UperNet-Seg, RMBG-1.4 — the last
two (NormalBAE, UperNet-Seg) required a real CUDA correctness bug fix along the way: `MaskRows`/`Transpose2D`
called with a `.Reshape` view as the write target orphans the CUDA activation-cache entry from the object the
caller reads back, producing silent all-zero output. Fixed in `NormalBaeModel.cs` (`SqueezeExcite`,
`RefineHead`) and `UperNetSegModel.cs` (`LayerNormChannelsFirst`, `MbConvBlock.Forward`) by allocating the
output already in the written shape and reshaping only on the way out.
**YOLO8, YOLO11, YOLOv8-Face, and
ArcFace stay blocked** — not gated access, a distribution-format gap: HF hosts these only as raw Ultralytics
`.pt` (a full pickled `nn.Module` object graph the engine's safe-subset pickle VM can't resolve) or, for
ArcFace, raw ONNX (a graph format neither loader reads). No ungated pre-folded-safetensors mirror was found
for any of the four (checked 2026-07-22). The architectures above are already real-weight parity-verified —
the gap is purely that the one-time offline conversion (`convert_arcface_onnx.py`, and the analogous YOLO
`-folded.safetensors` step) isn't something the Assets auto-download system can perform; it needs a local
Python step outside the engine. See `src/HartsyInference.Cli/Infra/ModelCatalog.cs` for the per-model detail.

## Deferred (❌)

**DINOv3** (needs encoder RoPE support — preset added but dimensionally-only), **EVA-CLIP** (EVA-02 vision
tower: RoPE + SwiGLU + sub-LN — preset added, blocked on tower support), MetaCLIP / AM-RADIO.
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

### Deferred stretch models
- [ ] DINOv3 (encoder RoPE support).
- [ ] EVA-CLIP / EVA-02 vision tower (RoPE + SwiGLU + sub-LN).
- [ ] MetaCLIP, AM-RADIO.

### Decoders
- [ ] JPEG / WebP decoders.

### Testing gaps
- [ ] CLIP bit-exact Python parity.
- [ ] All YOLO validation tests.
- [ ] CI.
