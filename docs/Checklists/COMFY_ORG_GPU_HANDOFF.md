# Comfy-Org rollout: GPU end-to-end handoff

For the GPU agent on the Linux box (RTX 4090 24 GB + RTX 3060 12 GB). Each Comfy-Org rollout PR is built and tested on
the Windows dev box, on CPU only, with tiny-config tests. This file lists, per PR, what still needs real weights, the GPU or
Swarm, and what passing looks like.

- **Work through it top to bottom.** On each PR:
  1. Run the steps.
  2. Fix what breaks: commit to the PR's branch as kalebbroo, with no trailers, and push.
  3. Post one verification comment.
- **Merging:** the dev-box session merges a PR once your comment says it is complete, CI is green, no thread is open,
  and Codex was clean at the last code change. Codex's review quota ran out on 2026-11-03; until it resets, PRs merge
  on the other three.
- **Keep this file current:** add new rollout PRs here as they open, and delete it when every row is final (the same rule
  as [COMFY_ORG_ROLLOUT.md](COMFY_ORG_ROLLOUT.md)).

## Rules for every GPU run

The full text is in the [add-model skill](../Agents/skills/add-model/SKILL.md) and [the shipping steps](../Agents/AGENTS.md#shipping-a-change).

- **Hygiene:**
  - Take the device lock (`gpu4090.lock` / `vk_3060.lock`).
  - Wait for `tests/swarm-quiet-window.sh` to report a 10-minute quiet window, and sample `nvidia-smi` for foreign
    processes during the batch.
  - Afterwards run `tests/swarm-quiet-window.sh --verify-since <UTC batch start>`. Exit 4 means redo the batch.
  - Never restart or evict Swarm mid-generation, and run one GPU suite at a time.
- **Proof the tests ran:**
  - Run mandatory tests through `tests/require-executed.sh` with a non-empty manifest and `HARTSY_REQUIRE_REAL_WEIGHTS=1`.
  - A `SKIPPED` line, a backend guard or a missing manifest test fails the run.
  - Put the executed test names and the checkpoint sha256 in the PR comment.
- **Downloads:** pins are in the catalog (`ModelCatalog`, `SideModels`). Check every downloaded file's sha256 against
  its pin.
- **Identity checks (CLI against API):**

  | Output | What must match |
  |---|---|
  | Image | Decoded RGBA digest, every layer included |
  | Video | Per-frame digests |
  | Audio | Decoded PCM |
  | Vision | Raw float arrays (the `.npy` payload equals the API's base64 bytes) |
  | 3D | Vertex, face and splat arrays |
  | LLM | Token ids |
- **Shared code:** `tests/regression-ab.sh`, current `main` against the branch with interleaved seeds, with the table in
  the PR.
- **Never count a refusal as support.** In the comment, mark each claim verified, partial (naming the refusals) or
  blocked (naming what unblocks it).

## Open PRs

### #631 OneReward (FLUX.1 Fill fine-tune) — held

Download, pins, LoRA load (427 of 427 weights) and Fill conditioning are verified on the 4090. Two design points from
the review are open before it can merge:
1. Guidance 30 is chosen by sniffing the file name in `Flux1Recipe`, which ENGINE_PATTERNS forbids. Wire distilled
   guidance as its own request field (not CfgScale: Swarm sends CFG 1 for Flux), or add hint-refined variants to the
   resolver in a separate PR.
2. The CLI's mask creativity default of 1.0 applies to every family. Scope it to Fill.

After that:
- Re-run object removal (`bus.png`, 768×1024, the two front people masked, guidance 30, full denoise, with and without
  `--lora flux.1-fill-dev-object-removal-lora`).
- Pass: the masked area shows street, not people.
- Record bf16 and fp8.

### #638 Stable Audio Open 1.0 (P3)

v-prediction EDM with CFG over DPM++ 3M SDE; Comfy-Org's all-in-one file (4.85 GB, sha256 `7b20458a…`).

1. **Load.** `hartsy music -m stable-audio-open-1.0 "<prompt>"` must load every component from its stable-audio-tools prefix
   with no missing key:
   - the DiT (`model.model.`; `FromWeights` must pick Open 1.0);
   - the VAE (`pretransform.model.`);
   - T5 (`conditioner.conditioners.prompt.model.`);
   - both timing conditioners.

   On a missing key, fix the prefix map in `StableAudioMusicModel.LoadOpen1Async`.
2. **Parity.** Against stable-audio-tools at the model card's settings: 100 steps, CFG 7, sigma 0.3 → 500, `dpmpp-3m-sde`,
   30 s, a fixed seed.
   - The sampler noise differs, so compare the stages: the DiT `v` output on a dumped `(x, σ, cond)`; the conditioning
     tensors (cross tokens, global token); the decoded audio of a dumped latent.
   - Record the DiT-step cosine and the VAE PCM error.
3. **Listen.** A music prompt, an SFX prompt and a 45 s clip, each opened and listened to.
4. **Swarm.** `Kind.Music`, once the engine version is published.

### #639 ComfyUI `asym_w4a8_int8` loading (P11 infra, moved up)

The decode runs on CPU at load. A layer without a correction term (every layer of the MOSS Qwen3.5 w4a8 file) comes out
as I8 with `int8_tensorwise` metadata, `s_channel` as its row scale: CUDA runs it on the resident int8 chain and other
backends widen it. A layer with a correction term comes out as dense BF16.
1. **Decode against comfy-kitchen.** On a real layer of Comfy-Org/MOSS-Image `text_encoders/qwen3.5_9b_w4a8_gptq.safetensors`
   (`model.language_model.layers.0.mlp.down_proj`), compare `W4A8ConvRotCodec.DecodeToInt8` with the int8 tensor
   comfy-kitchen's `_dequant_int4_grouped_to_int8` gives (pip `comfy-kitchen`, eager backend), and `DequantToBf16` with
   `dequantize_w4a8_int8_weight`.
   - Pass: the int8 grids are equal, and the BF16 max |Δ| is at the rounding floor.
   - Also compare the same layer with the bf16 build (`qwen3.5_9b_bf16.safetensors`): relative L2 near comfy-kitchen's
     documented ~0.07.
2. **6-bit.** A w6a8 build (MiniMax-H3's `*_w6a8*` DiTs) through the same pass.
3. **End to end:** with #640, below.

### #640 MOSS-Image int8 ConvRot Qwen3.5 encoder (P3)

The Qwen3.5 encoder gathers rows from an `int8_tensorwise` embedding and widens the trunk linears its backend cannot run
packed. Comfy-Org/MOSS-Image `text_encoders/qwen3.5_9b_int8_convrot.safetensors` (9.7 GB) needs a download; ask first.
1. **Encoder parity.** On one prompt, dump the encoder's hidden states with `--text-encoder` set to the int8 file and to
   `qwen3.5_9b_bf16.safetensors`. Record the cosine and relative L2 per token on CUDA (int8 stays packed) and on Vulkan or
   CPU (widened); the log line says how many linears were widened.
2. **Image.** `hartsy image -m moss-image --text-encoder <int8 file>` against the bf16 encoder at the same seed. Open both
   images and record peak VRAM for each.
3. **Edit.** One reference-image edit with the int8 encoder. Its vision tower is int8 as well, and that path has not been
   checked. Fix it or refuse it by name.
4. **w4a8.** Once #639 is merged into this branch or main, repeat steps 1–2 with `qwen3.5_9b_w4a8_gptq.safetensors`
   (7.3 GB). Ask before downloading.

## Merged, with GPU items still open

| PR | What | Still to run on GPU |
|---|---|---|
| #621 | `tests/require-executed.sh` | One class through `gpu-suite.sh` and one through `run-multigpu-campaign.sh` |
| #622 | Flux Fill, Klein 4B and Wan 2.2 A14B catalog entries | A14B T2V and I2V through the catalog: the low-noise expert pairs automatically, which the log shows. Flux Fill and Klein 4B catalog runs |
| #627 | Depth-Anything and DINOv2 sizes from the checkpoint | DINOv2 Large/Giant; the Depth-Anything parity dumps (needs torch) |
| #630 | MOSS-Image catalog | `moss-image` (Base bf16) and `moss-image-int8` through the catalog |
| #632 | Qwen inpainting ControlNet refusal | None. Implementing InstantX Inpainting is its own task |
| #633 | Real-ESRGAN parity | x2plus and anime-6B on real weights; a CUDA upscale against CPU |
| #635 | Layered image output | Waits for Qwen-Image-Layered (P6): engine → service → CLI files → API JSON → Swarm outputs |
| #634 | Vision contracts | Smoke `/v1/native/vision/sequence` on the live API (views → 501 naming multi-view; unordered timestamps → 400). The real modes arrive with P5/P7 |
| #636 | ByT5 glyph stream (Hunyuan-Image 2.1) | Whether the catalog's QuantStack GGUF carries `byt5_in` / `context_embedder_2` (a header read); an ungated default GGUF, since the current one returns HTTP 401; Swarm once published |

## Blocked

- **Klein-9B (P1):** BFL's repo is gated, so its terms must be accepted and a token provided on the box. Then pin the DiT,
  header-check `Flux2Config.Klein9B` and run it through the catalog.
- **Chroma Radiance from the CLI (P0):** host conversion of the 19 GB bf16 file runs out of memory. Needs an fp8 file or a
  leaner conversion, plus a run.

## Upcoming PRs (P3–P11)

Each new PR gets a section above when it opens, with its exact steps. The general shape per model follows the add-model
skill's definition of done:
1. Pin and sha check.
2. Real-weight parity against the official code or pinned ComfyUI at the same precision, from the reference script the PR
   adds.
3. A `regression-cases.sh` row.
4. The full input set through the modality's consumer, with every output opened.
5. Speed and VRAM against ComfyUI.
6. CLI-vs-API identity.
7. Swarm, where the model has a route (image, video, music, AudioLab, LLMAssistant).
