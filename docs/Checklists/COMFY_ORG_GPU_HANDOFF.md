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

### #631 OneReward (FLUX.1 Fill fine-tune) — removal works for some seeds, not reliably

Download, pins, LoRA load (427 of 427 weights) and Fill conditioning are verified on the 4090. Both review design points
are addressed on `fix/gpu-check-followups`, and the removal recipe from the OneReward README is reachable from the CLI:

1. Guidance: the fine-tune is a Flux variant (`Flux1Variants.OneReward`) selected by its catalog-id hint. `FineTuneOf`
   lets the hint win over its base's structural match. `--flux-guidance` sets the distilled guidance; CfgScale is not read
   for Flux.
2. The mask default (creativity 1.0) applies only to the Fill catalog ids.
3. True CFG: `--true-cfg` runs the negative prompt as the unconditional pass. The pipeline already had the path; the recipe
   never passed the negative prompt through.

Recipe (the README's removal example): `--negative nsfw --true-cfg 4 --flux-guidance 1 --steps 50`, prompt `remove`, removal
LoRA `flux.1-fill-dev-object-removal-lora` at 1.0. Scene: `bus.png` (768×1024), a tight silhouette mask (15 px dilation) on
the two front people, seed 42 unless noted. Pass: the masked area shows street, not people.

| Precision | Seed | Settings | Result |
|---|---|---|---|
| fp8 | 1 | removal LoRA | pass: both removed, no person generated; a repeat is byte-identical |
| fp8 | 2 | removal LoRA | fail: a man generated |
| fp8 | 3 | removal LoRA | pass |
| fp8 | 4 | removal LoRA | pass |
| fp8 | 42 | removal LoRA | fail: a man generated |
| fp8 | 42 | no LoRA | fail: a man generated |
| fp8 | 42 | ComfyUI template LoRA (`removal_timestep_alpha-2-1740`) | fail: a man generated |
| bf16 | 1 | removal LoRA | fail: a man in a navy suit generated |
| bf16 | 3 | removal LoRA | pass |
| bf16 | 42 | removal LoRA | fail: a man generated |
| fp8 | 42 | removal LoRA, 5 px mask | pass for people: both removed, two bags left on the sidewalk, no person |
| fp8 | 2 | removal LoRA, 5 px mask | fail: both removed, but part of a leg and a shoe remain in the masked area |
| fp8 | 2 | removal LoRA, 0 px mask | fail: both men regenerated |
| fp8 | 42 | removal LoRA, 0 px mask | fail: both men regenerated |
| fp8 | 2 | removal LoRA, negative = people classes | fail: a man generated |
| fp8 | 42 | removal LoRA, negative = people classes | pass: both removed, no person |
| fp8 | 1 | removal LoRA, negative = people classes | fail: a man generated (passed with `nsfw`) |
| fp8 | 2 | removal LoRA, T5 pad mask removed (experiment, reverted) | fail: a child generated |
| fp8 | 42 | removal LoRA, T5 pad mask removed (experiment, reverted) | fail: a man generated |
| fp8 | 1 | removal LoRA, T5 pad mask removed (experiment, reverted) | pass: both removed, a dark post on the sidewalk |
| bf16 | 0 | README example (`assets/image.png`, 1024×1024), T5 pad mask removed (experiment, reverted) | pass: the leaf sprig is removed and the wall and table edge are filled (visual check) |
| bf16 | 0 | README example, shipped build (T5 pad mask applied) | fail (visual): the leaf sprig is removed, but the area is a flat grey-blue smudge, not the wall |

Single-person runs (fp8, removal LoRA, seed 42): the left man alone is removed cleanly, and so is the center man alone.
Two passes (left, then center with the first output as its init image): the first pass is clean, and the second generates
a man in the center area.

Earlier runs without true CFG (20 steps, prompt `Remove the people`, guidance 30 or 1, rectangle or tight masks, fp8 and
bf16, several LoRA settings) all generated a person in the masked area.

Conclusion: on the bus scene, the recipe removes both people with nothing generated in 4 of 8 removal-LoRA runs (fp8 seeds
1, 3 and 4; bf16 seed 3), and in neither LoRA-free nor template-LoRA run. The result depends on the seed and the precision,
so it is not a reliable fix. The engine does reproduce a clean removal on the README's own example (see the table), so the
failures are specific to this scene: the people stand in a crowd, and the model paints a person into the masked area.

Findings from this round:
- Matches the reference: Fill conditioning (`[noise 64 | masked latent 64 | mask 256]`, `c = sy·8+sx`), the true-CFG combine
  (`uncond + scale·(cond − uncond)`), the LoRA (427 A/B pairs, no alpha keys, scale 1.0), and the settings (guidance 1,
  true CFG 4, 50 steps). The per-step masked-inpaint blend does not run on Fill models (`FluxPipeline.cs`: `isMaskedInpaint`
  excludes them).
- Divergence, fixed in a separate draft PR: the engine passes a pad mask to the T5 encoder for every FLUX.1 generation
  (`Flux1RecipePipeline.cs:76/148/160/164` → `FluxPipeline.cs:288/310`), so pad positions are masked in self-attention.
  diffusers (`pipeline_flux.py:254`) and ComfyUI's default Flux T5 (`sd3_clip.py`, `attention_mask=False`) run the encoder
  unmasked. Removing the mask did not change the bus-scene outcome on seeds 1, 2 and 42, so it is not the cause of the
  people. It does change the reference example: with the mask, the README's removal leaves a flat grey-blue smudge where the
  sprig was; without it, the wall fills cleanly. The unmasked T5 is therefore the fix for the FLUX.1 path, in its own draft PR,
  https://github.com/HartsyAI/HartsyInference/pull/646. FLUX.1-Dev text-to-image on seed 7 renders the requested sign text
  ("OPEN 24 HOURS") unmasked and garbles it masked (one prompt, so directional only). It changes every FLUX.1 generation. The numeric check against ComfyUI's T5 hidden states, and the FLUX.1 SSIM gate
  (`FluxSsimTests`, which needs `dump_flux_reference_image.py` output; that script sets `max_sequence_length=256`, so check the
  Dev window too), are not run here.
- Mask margin: 0 px fails on both seeds; 5 px removes both people on seed 42 but leaves a leg fragment on seed 2. The 15 px
  mask stays the recipe mask.
- Object-class negative prompt: not monotonic (seed 42 passes; seeds 1 and 2 fail).
- Not run: true-CFG scale 7 with extra negative terms. It was queued, then stopped during its first run, so there is no result.

Best-of-N with a person gate (in this PR): `tests/onereward-removal-retry.sh` runs the recipe on consecutive seeds and keeps
the first output where `tests/person_check.py` finds no person inside the mask. The gate runs YOLOv8n-seg's COCO person class
from the RAID's BN-folded `yolov8n-seg-folded.safetensors`, loaded with `strict=True`. A person counts when at least 0.6 of
its box lies under the mask. Across 43 labeled outputs of this scene, every visual failure scores 0.80 or more and every
visual pass scores 0.45 or less. The passes' highest values come from a person at the left edge whose box touches the mask.
OpenCV's HOG people detector was tried first and dropped: of the 7 candidates it accepted in seeds 1–16, 5 had a person in
the mask.

Retry results (fp8, 15 px mask, the #645 build with the unmasked T5 call-site change from #646; the first gate pass is the
chosen output):
- Seeds 1–8: the gate passes seeds 1 and 3. Chosen seed 1 (no person; a dark post on the sidewalk). Seed 6 has a leg and a
  sneaker in the mask, which HOG passed and the gate fails (cover 0.80).
- Seeds 9–16: the gate passes seeds 12 and 13. Chosen seed 12 (no person; a dark bin on the sidewalk).
- Seeds 17–24 (fresh range): the gate passes seeds 17 and 18. Chosen seed 17 (no person; a grey-blue bag on the sidewalk).
  Seeds 19–24 each have a person in the mask and fail the gate.
- Per-seed clean removal over these 24 seeds: 6 of 24 (25%). Each range returned a clean output after 1, 4 and 1
  generations.

Limits: the gate checks for people, so a leftover object can stay in the mask (the post, bin and bag above). The retry is a
selection, not a model fix: the recipe still puts a person in the mask on about 75% of seeds. Only this scene was tested.
Decision: the retry stays a script under `tests/`, not part of the CLI. The gate needs torch and ultralytics, which the engine
does not link, and porting YOLO into the C# engine would be a separate project. `NEGATIVE` (default `nsfw`) and the mask
argument are the levers the script exposes.

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
