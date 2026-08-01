# MiniMax-H3 — pre-release research

> **Status as of 2026-08-01: weights are NOT released.** Everything architectural below is vendor
> marketing copy, not a spec. Nothing in this document is sufficient to write a forward pass. The
> engine seam is wired (`MiniMaxH3Recipe`) and fails loudly *when handed a checkpoint* — see
> [Engine state](#engine-state) for what actually surfaces. The bring-up checklist starts at
> [Unknowns](#unknowns--the-day-one-checklist).

## What H3 is

MiniMax-H3 (internally "Hailuo 03") is an omni-modal generative model announced **2026-07-31**. It
takes text, images, video, and audio as input and generates video with a **jointly generated native
stereo soundtrack** — the audio is not a separate TTS/foley pass. Output is up to **15 seconds at 2K,
24 fps**.

MiniMax state they "plan to open up the model weights in the coming days, subject to applicable laws
and regulations," and that a full H3 Technical Report is coming. Neither has shipped.

## Release status — verified, not assumed

Checked 2026-08-01:

| Source | Result |
|---|---|
| `modelscope.cn/api/v1/models/MiniMax/MiniMax-H3` file tree | `获取模型失败` — repo does not exist / not public |
| `huggingface.co/api/models?search=MiniMax-H3` | `[]` |
| `huggingface.co/api/models?author=MiniMaxAI` | newest is `MiniMax-M3` (2026-07-23); no H3 |
| H3 Technical Report | not published |
| `config.json` / any state-dict key list | does not exist publicly |

**ComfyUI's H3 support is API-only and contains zero architectural information.** The nodes in
`comfy_api_nodes/nodes_minimax.py` (`MinimaxHailuo03TextToVideoNode`,
`MinimaxHailuo03FirstLastFrameNode`, `MinimaxHailuo03ReferenceNode`) POST to
`/proxy/minimax/v2/video_generation` and poll `/proxy/minimax/v2/query/video_generation/{task_id}` —
they are a thin cloud client. ComfyUI's own announcement says "Native open-weight support is coming
soon." **There is nothing to port from Comfy.** Re-check that file when weights drop; the local
implementation will land somewhere else entirely (`comfy/ldm/…`).

## Capability contract (KNOWN — this part is not speculation)

Extracted from ComfyUI's node schemas and validators, which encode MiniMax's documented API limits.
This is the user-facing surface the engine must eventually satisfy, and it is safe to design against
now.

**Output**

| Field | Value |
|---|---|
| Resolution | `2K` only (no other option exposed) |
| Aspect ratio | `adaptive`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`, `21:9` — T2V only; frame/reference modes inherit AR from the inputs |
| Duration | integer seconds, **5–15** per the Comfy node (`min=5,max=15`). Some press reports say 4–15; trust the node until MiniMax's own docs are readable |
| Frame rate | 24 fps |
| Audio | native stereo on every clip; sample rate not documented |

**Inputs — three modes**

1. **Text-to-video** — prompt + resolution + ratio + duration + seed.
2. **First/last frame** — `first_frame` (required) + `last_frame` (optional), each tagged with a
   `role` field in the request `content` array. Output AR follows the images; no `ratio` input.
3. **Reference-to-video** — up to **9 reference images**, **3 reference videos**, **3 reference
   audio clips**. Referenced positionally from the prompt text as `"Image 1"`, `"Video 1"`,
   `"Audio 1"`, … At least one image or video is required; **audio references cannot be used
   alone**.

**Validation limits**

- Images: min 256×256, aspect ratio within **0.4–2.5**, ≤30 MB.
- Reference videos: **23.976–60 fps**, ≥2 s each, ≤15 s total across all clips, ≤50 MB.
- Reference audio: ≥2 s each, ≤15 s total, ≤15 MB.
- Mixed reference inputs capped at **12 files** total; prompt ≤7,000 characters.
- Seed: uint32. MiniMax explicitly document it as **not** guaranteeing identical results — same seed
  gives "similar, but not guaranteed identical" output. Do not build a bit-exactness gate on it.

## Architecture — the four named components (marketing-level only)

MiniMax name four pieces. Each is a label with a claimed benefit and **no disclosed structure** —
treat every one as an open research question, not a design input.

- **H3-VAE** — replaces the conventional video tokenizer. Claimed **4× gain in effective sequence
  length** from a higher compression ratio, and this is credited as the enabling technology for
  native 2K. Compression ratio, latent channel count, and whether video and audio share one latent
  space are all undisclosed.
- **H3-Omni Transformer** — one pre-training framework spanning T2I, T2V, I2V, and audio. Claimed to
  "separate understanding and generation workloads" for better hardware utilization, worth ~30%
  end-to-end training throughput. Whether that separation is a two-tower design, an MoE, or a
  routing scheme is undisclosed.
- **Contextual Omni Representation** — language used as the bridge between modalities. ~100K tokens
  of source material distilled to an average of ~4K tokens at inference.
- **In-Context Regeneration** — 2K output is produced by the base model **regenerating** a
  low-resolution result, explicitly *not* a bolt-on super-resolution module. Implies a second
  conditioned pass through the same backbone rather than a separate upscaler stage.

Unverified: a widely-reshared tweet claims H3 "can run on RTX 3060." That is tweet-level, contradicts
nothing but is supported by nothing. Do not size the implementation around it.

## Unknowns — the day-one checklist

When weights land, these are the questions to answer in order. Every one of them currently blocks
writing a single line of forward pass.

- [ ] **Parameter count** and checkpoint layout (single file vs sharded vs diffusers folder).
- [ ] **H3-VAE**: spatial and temporal compression factors, latent channel count, encoder/decoder
      block structure, whether it is causal 3D (Wan/Hunyuan-style) or something new.
- [ ] **Audio latent path**: does audio share the video latent space, or is there a separate audio
      VAE + vocoder like LTX-2.3? Sample rate, channel layout (true stereo latents vs mono + spatializer).
- [ ] **Text encoder identity** — which LLM/encoder, which layer is tapped, what prompt template.
      "Contextual Omni Representation" implies a real language model in the conditioning path.
- [ ] **DiT block structure**: single-stream vs dual-stream vs MMDiT, patch size, RoPE convention
      (interleaved vs split-half — this has burned us on LTX-2.3 and Qwen3.5 already), norm placement,
      QK-norm, AdaLN vs modulation table.
- [ ] **Scheduler**: flow-match vs ε-prediction, sigma shift, guidance convention (CFG vs embedded /
      distilled guidance).
- [ ] **In-Context Regeneration mechanics** — how the low-res pass is fed back in. This determines
      whether generation is one pipeline call or two, and it changes the pipeline's shape.
- [ ] **License** — MiniMax say "subject to applicable laws and regulations," which is not a license
      name. Confirm before shipping catalog assets.

## Engine state

What is wired today, and — measured, not assumed — what a user actually sees:

- `MiniMaxH3Recipe` (`src/HartsyInference.Engine/Recipes/Video/`) is registered in `VideoRecipeRegistry`
  and matches `minimax-h3`, `minimax-hailuo-03`, `hailuo-03`, `hailuo03`. `Construct()` always throws
  `UnsupportedModelException` with the release status plus a pointer to this doc.
- Short-form catalog entry `minimax-h3` — `ModelStatus.Structural`, not CLI-drivable, no `Assets`.
- **Verified on the CLI (net10.0, 2026-08-01):**
  `hartsy video -m minimax-h3 --model-path <any file> "…"` reaches `Construct` and prints the full
  explanation. Plain `hartsy video -m minimax-h3 "…"` does **not** — checkpoint resolution runs before
  recipe construction, so it stops at the generic *"No checkpoint found for this model"*, byte-identical
  to what an unregistered model id produces. The seam helps whoever supplies an H3 checkpoint; it does
  not improve the bare-model-id probe.
- No `ModelSupport` compat-class entry was added. That list is keyed on SwarmUI `T2IModelClassSorter`
  strings, and H3's class string will not exist until weights ship — a guessed entry would be dead code.

`VideoDefaults` on the recipe are **placeholders except `Fps = 24`**: MiniMax's API exposes the opaque
string `"2K"` plus an aspect ratio and never states pixel dimensions, steps/CFG depend on an undisclosed
sampler, and `Frames` is 5 s × 24 fps arithmetic. One concrete impedance mismatch for the implementer:
H3's API takes **duration in integer seconds (5–15)** while `VideoRequest` carries a **frame count**.

## Bring-up notes to save the next session time

**Naming and hosting.** Record every alias — the sniffer must not be keyed to one guess:

- Comfy calls the nodes `MinimaxHailuo03*`; **H3 == "Hailuo 03"** internally. The architecture string
  in the checkpoint could plausibly be `hailuo`, `Hailuo03`, `hailuo-03`, or `MiniMaxH3`.
- **Org slugs differ per host**: ModelScope is `MiniMax/MiniMax-H3`; HuggingFace uses `MiniMaxAI/*`
  (so expect `MiniMaxAI/MiniMax-H3`). Check both hosts — MiniMax has historically published to
  ModelScope first.

**The blocking engine gap: generated audio has nowhere to go.** `IVideoRecipePipeline.Generate`
returns `IReadOnlyList<VideoFrame>` — frames only. This is already a live TODO: `LtxVideo2RecipePipeline`
generates an LTX-2.3 soundtrack and **logs a warning and drops it** (`TODO(E-IMG-4/5)`), because the
frame-only contract cannot carry it. Wan2.2-S2V muxes audio only because its audio is an *input* that
gets passed through to `VideoOutputEncoder.AudioTrack`.

For LTX-2.3 that gap is a missing feature. **For H3 it is fatal** — every H3 clip has a jointly
generated soundtrack, and a video model that silently discards half its output is worse than one that
doesn't load. Widening the video result contract to carry a generated waveform (sample rate + channel
count + PCM, alongside the frames) is therefore a **prerequisite** for H3, not a follow-up. LTX-2.3
picks up the fix for free, which makes it independently verifiable against a model we already have
working before H3 ever arrives.

**Precedents worth reading first.** LTX-2.3 (`LtxVideo2Pipeline`, `Ltx2Result`) is the closest
existing architecture: dual-stream video+audio DiT with separate video and audio VAEs and a BigVGAN
vocoder. Whatever H3 turns out to be, that pipeline is the nearest template and its
`Ltx2Result`-shaped return is the natural model for the widened contract.

## Sources

- [MiniMax H3 announcement blog](https://www.minimax.io/blog/minimax-h3)
- [ComfyUI `comfy_api_nodes/nodes_minimax.py`](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_api_nodes/nodes_minimax.py) — API node schemas, the source of the capability contract
- [ComfyUI announcement (partner nodes, "native open-weight support coming soon")](https://x.com/ComfyUI/status/2083071877891682784)
- [MarkTechPost writeup](https://www.marktechpost.com/2026/08/01/minimax-releases-minimax-h3-an-omni-modal-video-model-that-generates-15-second-2k-clips-with-native-stereo-audio/) — reference-input limits, file size caps
- [Comfy docs — Hailuo MiniMax in ComfyUI](https://comfy.org/p/supported-models/hailuo-minimax/)
