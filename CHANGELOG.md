# Changelog

## Unreleased

The SANA-WM stack (stage 1, its performance pass, and streaming) is written up here once. The `## alpha.N` section and the
`VersionSuffix` are set when the stack lands on `main`.

- **Added: SANA-WM stage 1 (`sana-wm`).** `hartsy world -m sana-wm --image <png> --prompt-file <txt> --intrinsics fx,fy,cx,cy
  (--actions "w-80,dw-40,aw-40" | --camera poses.npy)` renders up to 161 frames at 704x1280 and 16 fps from one image, a prompt
  and a camera path (per-frame poses, or an action script over `wasdijkl`). It matches NVIDIA's stage-1 reference on an RTX 4090
  and on a 12 GB RTX 3060, where the upstream code does not fit: a 161-frame clip takes about 3 minutes on the 4090 and 14.5
  minutes on the 3060. The refiner, FP8/FP4 and Pi3X intrinsics are refused when the request is planned. The weights are
  non-commercial (the LTX-2 VAE is under the LTX-2 community licence).
- **Added: SANA-WM streaming (`sana-wm-streaming`).** The same model distilled to be chunk-causal. It generates 24 video frames
  at a time and decodes each chunk as soon as it is done, and the camera can change while it runs: `--actions "w-24,d-24,s-24"`
  drives it, `--interactive` steers it from the keyboard (`wasdijkl`, space to release, `q` to stop), and `--unbounded` runs
  until stopped. Each frame is written to disk as it arrives, and `q` or Ctrl+C saves what was generated and muxes the MP4. A
  chunk takes 1.75 s on the 4090 and 7.7 s on the 12 GB 3060; every frame matches upstream's streaming sampler at PSNR 44.9 dB
  or better on both. The refiner (`--refiner`) is written and off by default; it has not run on a GPU.
- **Added: new backend ops for SANA-WM.** `FrameDeltaScan` (with an optional carried state), `DepthwiseConvGlu`,
  `UcpeTransform`, `DepthwiseConvFrames` and `RmsNormRelu`, each with a host reference and a CUDA kernel, plus `Exp` and
  `Reciprocal`. A SANA-WM step takes 2.59 s on the 4090 against 3.05 s for upstream PyTorch. Existing ops are unchanged.
- **Added: streaming building blocks.** `ChunkPlanner`, `RollingTokenWindow`, `FrameTailCache`, `CausalStreamCache` and
  `SelfForcingFlowSchedule` (Diffusion), `LiveCameraPath` and `ICameraIntegrator` (World), `ChunkedWorldSession` and
  `StreamingFrameWriter` (Engine). Another chunk-causal world model can reuse them. `vram.sanaWmStreamState` (default `auto`)
  holds the carried state in BF16 on cards of 16 GiB or less, as upstream does, and in F32 on larger ones.
- **Added: shared pieces for SANA-WM.** `VideoTemporalTiling` for decoding a long video latent in overlapping tiles, `CameraRayFrame`,
  `CoverCropMapping` and a `PluckerLayout` option on `PluckerEmbedding` (the existing overload is byte-identical), and a
  `World.Tests` project.
- **Fixed: CUDA `BatchedMatMul` wrote F32 into a BF16 output buffer.** The cuBLAS output type now follows the output tensor's
  dtype. `Conv2dDepthwise` also takes a BF16 output on CUDA.
- **Changed: Gemma-2 text encoders apply the attention-logit softcap (50).** Lumina-2 uses Gemma-2-2B, so its images change; this
  was a deliberate fix. Gemma-2 also encodes right-padded batches and can return the post-final-norm hidden state.
- **Changed: `CausalConv3d` holds one weight form** instead of both: the per-tap slices are cut on first use, and a convolution
  the native 3-D op keeps declining releases its 5-D form. SANA-WM's decoder drops from 1762 to 999 MB of weights after its first
  decode with byte-identical frames, and an LTX-2.5 clip is byte-identical too. The first decode in a process pays the slicing,
  about 1.7 s.
- **Changed: world planning.** `hartsy world --frames` below 2 is refused, random seeds are drawn and recorded, and an unknown
  world model id throws instead of loading Oasis.

## alpha.405

- Kimi K3.1 Pro (`kimi_k31`): CPU reference for the KDA forget gate, the KDA recurrence and output, the latent MoE
  order, the routing rule, and attention residuals, with the SiTU activation shared with the GLU reference.
- Expert-parallel dispatch host reference (`ExpertParallelDispatch`): contiguous expert ownership, route planning,
  send counts, and a combine that matches a single-device evaluation.
- MXFP4 checkpoint branch in `HfQuantCheckpoint` for the `mxfp4-pack-quantized` routed-expert tensors.
