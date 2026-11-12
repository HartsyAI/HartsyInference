# Iris-Z architecture

Iris-Z (Sperid Labs, released 2026-11-11) is a FLUX.2 fine-tune that generates an image and its depth map with one set
of weights. This file records what the release contains and how the engine maps it. Status and evidence live in
[MODEL_STATUS_IMAGE.md](../Checklists/MODEL_STATUS_IMAGE.md#iris-z).

Sources, pinned:

- Weights: `speridlabs/iris-z` at `16239e1a5444c484b11a04b61d633eceb768574c`.
- Reference code: `speridlabs/iris-z-sdk` at `6c8925f42a2a3581127609d771aa5bfece557467` (Apache-2.0).
- Paper: <https://speridlabs.com/assets/research/iris-z/iris-z.pdf>.

## Release inventory

| | Iris-Z-small | Iris-Z-big |
|---|---|---|
| Base model | FLUX.2 Klein base 9B | FLUX.2 Dev |
| Parameters | 9.1B | 32.2B |
| Transformer files | 4 shards, 18.16 GB, BF16 | 14 shards, 64.3 GB, BF16 |
| Topology | hidden 4096, 32 heads, 8 double + 24 single | hidden 6144, 48 heads, 8 double + 48 single |
| Text encoder | Qwen3-8B, taps 9/18/27, 512 tokens | Mistral-Small-3, taps 10/20/30, 512 tokens |
| Guidance | true CFG 4.0 | true CFG 3.5; the guidance embedding is removed |
| Sky ceiling | 1.0 (no sky class) | 0.95 |
| Reference GPU memory, all tasks | about 33 GB | about 112 GB |

Both folders also hold `config.json`, `transformer.safetensors.index.json` (tensor name to shard) and
`empty_prompt.safetensors` (`empty_prompt`, BF16 `[1, 512, context]`, the encoder's output for an empty prompt). The VAE
is the FLUX.2 autoencoder, which the release does not ship. The weights carry the FLUX non-commercial licence of their
base models. No Comfy-Org, Kijai or city96 repack and no quantized build existed on 2026-11-12, and neither ComfyUI nor
SwarmUI core recognizes the model.

Shard sha256 values are the Hugging Face LFS object ids. Iris-Z-small:

| File | Bytes | sha256 |
|---|---|---|
| `transformer-00001-of-00004.safetensors` | 4932515680 | `4b1fec0714cad8e0a76b2512b91fd21b7937879f80be11dc518817a8dfcbae2b` |
| `transformer-00002-of-00004.safetensors` | 4699760848 | `b402401fcf7dbe5e26d55d7a52abf72291636ed0be6b0f61fb95c0c085ffd2b1` |
| `transformer-00003-of-00004.safetensors` | 4798294288 | `148f4d1d65765f02ad0e2f4a1157ed3c749d798d5c789d10272c69fb2d55a7a1` |
| `transformer-00004-of-00004.safetensors` | 3726647136 | `5b93e5f0069dff96a5ffa23cdee666ad220fa8ef06d22142aaa54e5bfae51064` |
| `empty_prompt.safetensors` | 12583040 | `3e39a39a9444db5b63b376470deeff230f4428945f2cc510a4ef83534ca11925` |

## What differs from FLUX.2

The checkpoint uses BFL's own key names (`img_in`, `double_blocks.N.img_attn.qkv`, `single_blocks.N.linear1`, ...), the
layout `Flux2CheckpointConverter` already reads. Read from the real headers, the only new tensor is
`modality_in.weight`, `[4, hidden]`. Iris-Z-big has no `guidance_in` tensors.

1. **Two grids.** The image sequence is `[RGB tokens, depth tokens]`, both on the same patch positions. RoPE ids are
   `(0, row, col, 0)` for RGB and `(1, row, col, 0)` for depth; text stays at `(0, 0, 0, i)`.
2. **Modality embedding.** After `img_in`, each token adds one row of `modality_in`: 0 RGB, 1 depth, 2 given RGB,
   3 given depth. A given grid is clean conditioning and is never denoised.
3. **A timestep per grid.** Text tokens are modulated at the sampling time `t`. Each grid is modulated at its own level:
   `t` when it is being denoised, 0 when it is given, 1 when it is left as noise. The double-stream image modulation,
   the single-stream modulation and the final layer all take the per-grid time vector.
4. **True CFG.** The unconditional branch is the empty prompt. Guidance is `uncond + g·(cond − uncond)`.

The four tasks are four settings of the two grids:

| Task | RGB grid | Depth grid | Denoised | Steps | Guidance |
|---|---|---|---|---|---|
| Prompt to image | noise, level `t` | noise, level 1 | RGB | 50 | CFG |
| Prompt to image + depth | noise, level `t` | noise, level `t` | both | 50 | CFG |
| Image to depth | the photo, given, level 0 | noise, level `t` | depth | 4 | none, empty prompt |
| Depth to image | noise, level `t` | the depth map, given, level 0 | RGB | 50 | CFG when a prompt is set |

Prompt to image + depth then estimates the depth again from the finished image, which is sharper than the joint depth;
the joint depth is available on request.

The schedule is FLUX.2's: `linspace(1, 0, steps + 1)` through the generalized SNR shift with the empirical `mu` of the
single-grid token count and the step count. Sampling is plain Euler.

## Depth encoding

Depth passes through the same VAE as images. A depth map is relative log depth scaled so its 2nd to 98th percentile spans
`[-1, sky_ceiling]`, with sky at +1, replicated to three channels. Decoding takes the mean of the three channels and
maps it back to `[-1, 1]`; values above `(sky_ceiling + 1) / 2` are sky.

Image to depth averages the last two clean-depth estimates (`x − t·v`) of its four steps: each is decoded, aligned to
the last by least squares, and the per-pixel median is kept. Several seeds are combined the same way. The working size is
about 512² for a photo under 0.5 MP and 1024² above, with sides that are multiples of 16, never larger than the photo.

## Engine mapping

| Need | Owner |
|---|---|
| Transformer blocks, RoPE, checkpoint conversion | `Flux2Transformer`, `Flux2DoubleBlock`, `Flux2SingleBlock`, `Flux2CheckpointConverter` |
| Per-grid modulation | `TokenSegmentLayout` + `SegmentModulation`, now accepted by the FLUX.2 blocks |
| Grid position ids | `Flux2PosEmbed.BuildGridPositionIds` |
| Multi-grid forward | `Flux2Transformer.ForwardGrids` |
| VAE and latent packing | `Flux2VaeComponents`, `Flux2LatentOps` |
| Text encoders | `LlamaStyleEncoder` with the FLUX.2 Klein 9B and Dev side models |
| Sampling | `FlowMatchEulerDiscreteScheduler` and the sampler seam |
| Depth encoding, ensembles, rendering | `IrisZDepth` |
| Tasks | `IrisZPipeline` |
| Recipe, variants, defaults | `IrisZRecipe`, `IrisZVariants` |

True CFG runs as two batch-1 forwards, which is the same arithmetic as the reference's batch of two.

## Open questions for a real-weight run

- Whether the encoder's output for an empty prompt equals the shipped `empty_prompt` tensor closely enough that either
  can serve as the unconditional branch.
- The padding side and pad token of the reference's Mistral processor for Iris-Z-big. The engine's FLUX.2 Dev path
  follows ComfyUI and does not pad.
- The reference VAE uses a BatchNorm epsilon of 1e-4; the engine's FLUX.2 pipeline passes 1e-5. Iris-Z uses 1e-4.
- Memory at 1024²: the sequence holds twice the image tokens of FLUX.2 at the same size.
- The reference draws noise on the GPU, so seeds do not reproduce its images.
