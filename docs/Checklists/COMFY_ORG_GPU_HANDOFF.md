# Comfy-Org rollout: GPU end-to-end handoff

For the GPU agent on the Linux box (RTX 4090 24 GB + RTX 3060 12 GB). Each Comfy-Org rollout PR is built and tested on
the Windows dev box, on CPU only, with tiny-config tests. This file lists, per PR, what still needs real weights, the GPU or
Swarm, and what passing looks like.

- **Work through it top to bottom.** On each PR:
  1. Run the steps.
  2. Fix what breaks: commit to the PR's branch as kalebbroo, with no trailers, and push.
  3. Post one verification comment.
- **Merging:** the dev-box session merges a PR once your comment says it is complete, Codex is clean at the head, CI is
  green and no thread is open.
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

### #634 Vision contracts (P2)

- No model implements the new modes yet; P5 and P7 add them. On the GPU box:
  - Run the CPU contract suites (`VisionContractTests`, `VisionContractEndpointsTests`, `VisionOutputWriterTests`).
  - Smoke `/v1/native/vision/sequence` against the live API: views → 501 naming multi-view; frames with a bad timestamp
    order → 400.
- Confirm that an existing mode (`depth`, `embed`) gives byte-identical output against `main` through the CLI and the API.
- Needs a `## Unreleased` changelog entry before merging.

### #636 ByT5 glyph stream for Hunyuan-Image 2.1 (P3)

Verified on the 4090 with the fp8 checkpoint: quoted text is legible and the unquoted run is garbled. Since then:
- The catalog download is skipped when `--model-path` names the transformer.
- The first-use glyph download now honours cancellation (`948c16ca9`).

Remaining:
1. Whether the catalog's QuantStack GGUF carries `byt5_in` / `context_embedder_2`. A header read is enough. Without it the
   recipe warns and skips glyph text, so record which.
2. The catalog's default GGUF returned HTTP 401 (gated or moved); propose an ungated default.
3. Swarm, once the engine version is published.

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
