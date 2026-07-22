# Handoff: verify the LLM model catalog + new CLI flags on real hardware

## Context

This session wired up `hartsy text -m <id>` so every LLM/text-generation architecture the engine supports is
at least *discoverable* and *drivable*, and added several CLI flags (system prompt, image attachment,
sampler knobs, thinking-mode toggle) that the underlying `TextRequest`/`TextService` already supported but
the CLI never exposed. None of it could be exercised end-to-end — **this dev machine has no GPU**. Your job:
download real checkpoints on a GPU box and actually run `hartsy text` against every new catalog entry and
every new flag, the same way a prior session did for image models (see `docs/Checklists/MODEL_STATUS_IMAGE.md`,
`PARITY_VERIFICATION.md`, and memory entries `image-cli-catalog-verify-0721` /
`adversarial-image-output-verification` for that methodology) and a later one did for video
(`docs/Checklists/VIDEO_CLI_CATALOG_HANDOFF.md`, which this doc's structure mirrors).

## Where LLM differs from image/video (read this before starting)

Image and video needed real construction code written per architecture family (an `IArchitectureRecipe` /
`IVideoRecipe`) because diffusion checkpoints vary structurally. **LLM text generation does not have that
problem** — `docs/Design/ADDING_A_MODEL.md` Section C says plainly that a new GGUF architecture is "usually
nothing to do": `TextService` (`src/HartsyInference.Engine/Services/TextService.cs`) is config-driven off the
GGUF's own `general.architecture` metadata, dispatching to the shared `GenericTransformer` spine (dense +
MoE) or an `ISsmModel` (Mamba/RWKV/Qwen3.5) via `GgufKeyMapperRegistry`. So the gap here was never "missing
construction code" — it was:

1. **Catalog breadth.** `ModelCatalog.cs`'s Text section only listed 5 entries (`qwen2`, `qwen3`, `llama3`,
   `mistral`, `gguf`) before this session. `docs/Checklists/MODEL_STATUS_LLM.md` documents ~25 engine-verified
   architectures that were invisible to `hartsy list` and reachable only via a raw `--model-path` the user had
   to already know to point at. This session added ~26 new catalog entries (below) covering all of them, plus
   a few (`glm4`, `gpt2`, `starcoder2`) that have a real `IGgufKeyMapper` but no documented bring-up run at all.
2. **CLI flag gaps.** `TextRequest` already had `SystemPrompt`, `TopK`, `MinP`, `RepetitionPenalty`,
   `LowVramQuant`, `AlwaysFreeMemory`, and image attachment via `TextMessage.Images`, but
   `GenerationDispatch.TextAsync` never set most of them and `TextCommand` exposed no flags. This session
   added the flags and the plumbing (see "New CLI flags" below).
3. **"Thinking" mode was genuinely unwired**, not just missing a flag: `JinjaChatTemplate.Render`'s context
   dict never set `enable_thinking`, so a Qwen3-family GGUF's own embedded chat template was branching on an
   undefined variable. This session added an `EnableThinking` field that threads
   `TextRequest.EnableThinking` → `GenerationRequest.EnableThinking` → `PromptBuilder.BuildPromptIds` →
   `IChatTemplate.Encode(..., enableThinking)` → `JinjaChatTemplate`'s context dict (only set when non-null,
   so unset stays genuinely undefined rather than being forced to `false`). Covered by a synthetic unit test
   in `tests/HartsyInference.LLM.Tests/JinjaAndTokenizerTests.cs` (`JinjaChatTemplate_EnableThinking*`), but
   **that test uses a fake template and fake tokenizer** — it proves the plumbing works, not that any real
   Qwen3 GGUF's actual template renders differently with `--thinking` vs `--no-thinking`. Verify that for
   real.
4. **No HF GGUF repo sources are reliably documented** for most of the new architectures. Per a hard rule for
   this project (never fabricate/guess HF URLs), every new catalog entry shipped **without** `Assets`
   (auto-download metadata) — same as the 5 pre-existing entries. Finding and confirming a real, currently-live
   quantized GGUF per family, then adding `Assets`, is most of the work left for you.
5. **T5/FLAN-T5 is catalogued but marked `CliDrivable = false`.** `T5Model` (`src/HartsyInference.LLM/Seq2Seq/
   T5Model.cs`) is a real, engine-verified encoder-decoder implementation (cosine = 1.0 vs HF), but
   `TextService.LoadInto` only ever routes a GGUF to the decoder-only `GgufLanguageModel.Load` path or (for
   SSM architectures) `SsmLanguageModel.Load` — there is no seq2seq branch that would call `T5Model.Load`.
   Confirmed by direct code read, not inference. Wiring this properly means adding a genuinely different
   generation loop (encode once, then decode cross-attending) to `TextService`, which is real design work, not
   a flag — out of scope for this pass. If you have time and appetite, it's a good follow-up; otherwise just
   confirm the gap is still there and leave it documented.

## Current state (as of this session)

`src/HartsyInference.Cli/Infra/ModelCatalog.cs`, Text section — new entries added this session, all
short-form `E(...)`, `Status = Structural` (not `Verified` — that's an engine-internal-parity claim from
`MODEL_STATUS_LLM.md`, not a CLI-verified one), `CliDrivable = true` unless noted, **no `Assets`**:

- **Dense**, engine-verified per `MODEL_STATUS_LLM.md`: `gemma` (Gemma 2/3 text), `phi` (Phi-3/3.5-mini/
  4-mini), `stablelm2`, `granite3`, `command-r`, `olmoe`, `granite-moe`, `gemma4` (E2B/E4B mobile).
- **MoE / large dense, build-defer** (architecture + key-mapper + slice tests pass, no e2e run at any size —
  every one is >12GB): `mixtral`, `qwen3-moe`, `deepseek-v2-lite`, `deepseek-v3`, `gpt-oss`, `gemma4-moe`.
- **Vision-language (VLM)**, engine-verified e2e, reachable with the new `--image` flag: `llama32-vision`
  (mllama, gated cross-attention, no token splice), `gemma3-vision`, `smolvlm2`, `llava15`, `qwen25-vl`
  (covers both the 3B and 7B checkpoints).
- **Non-transformer / hybrid** (`Ssm/*Model.cs`, not `GenericTransformer`): `mamba` (Mamba-1/2, Falcon-Mamba),
  `rwkv` (RWKV-6/7), `qwen35` (Qwen3.5 Gated DeltaNet hybrid — only the 0.8B size was actually run; 2B/4B/9B
  share the same code path but are untested).
- **Encoder-decoder**: `t5` — `CliDrivable = false`, see point 5 above. Don't just flip this to `true`; it
  will fail without the `TextService` seq2seq work.
- **Undocumented bring-up** (real `IGgufKeyMapper` exists — `Glm4KeyMapper`, `Gpt2KeyMapper`, and
  `starcoder2` via `LlamaKeyMapper`'s architecture list — but no verified run anywhere in the checklists):
  `glm4`, `gpt2` (covers GPT-2/BLOOM/GPT-NeoX), `starcoder2`. Treat these as genuinely unknown — expect the
  first real run to surface bugs the doc-verified families already had shaken out of them.

`qwen2`, `qwen3`, `llama3`, `mistral`, `gguf` are unchanged from before this session.

## New CLI flags (`hartsy text`) → `TextRequest` field

All added to `src/HartsyInference.Cli/Commands/TextCommand.cs` + threaded through
`src/HartsyInference.Cli/Dispatch/GenerationDispatch.cs`'s `TextAsync`:

| Flag | `TextRequest` field | Notes |
|---|---|---|
| `--system <text>` | `SystemPrompt` | |
| `-i\|--image <path>` (repeatable) | `Messages[0].Images` | PNG/BMP, decoded via the same `PngDecoder`/`BmpEncoder` idiom `ReplSession.RenderPreview` and `GenerationDispatch.LoadImage` already use. **Exercise this on every VLM entry above** — it's new code, unverified against a real model. |
| `--top-k <int>` | `TopK` | |
| `--min-p <float>` | `MinP` | |
| `--repetition-penalty <float>` | `RepetitionPenalty` | |
| `--thinking` / `--no-thinking` | `EnableThinking` (tri-state; unset when neither is passed) | See point 3 above — **verify against a real Qwen3/Qwen3.5 GGUF**, not just the synthetic unit test. Mutually exclusive; `TextCommand.Execute` rejects both being set. |
| `--low-vram-quant` | `LowVramQuant` (any non-empty string is a bool-shaped toggle server-side, see `TextService.LoadInto`) | |
| `--always-free-memory` | `AlwaysFreeMemory` | |

Out of scope, not built this session (noted so you don't assume they exist): tool-calling flags
(`TextRequest.Tools`/`ForceToolId`), `SpeculativeDecode`, and REPL multi-turn conversation history — the CLI
and `ReplSession` are still single-turn only (each call builds one fresh `TextRequest` with exactly one user
message).

## Known checkpoint sources — only 3 are documented, everything else needs real research

From `docs/Checklists/MODEL_STATUS_LLM.md` / `LLM_MODEL_COVERAGE.md` (do not re-derive these, they're already
named):

- **`qwen25-vl`** (7B specifically): "unsloth Q4_K_M + mmproj-F16" — `MODEL_STATUS_LLM.md` line ~75.
- **`gemma3-vision`**: `ggml-org/gemma-3-4b-it-GGUF`, `mmproj-model-f16.gguf` (812MB) —
  `LLM_MODEL_COVERAGE.md` line ~176.
- **`llama32-vision`**: "leafspark Q4_K_M + mmproj-F16, low-VRAM" — `LLM_MODEL_COVERAGE.md` line ~403.

These are the fastest path to a first real result, and exercise the new `--image` flag immediately. For every
other entry, you need to find a real, currently-live, ungated quantized GGUF yourself — search HF directly
(the usual publishers seen elsewhere in this repo's catalogs are `bartowski`, `unsloth`, `ggml-org`,
`QuantStack`, `Comfy-Org` repacks, but don't assume any specific one exists for a given family without
checking). Per the project's hard rule: never fabricate/guess a URL, and if the canonical repo is gated, find
an ungated community repack instead of asking to escalate (same rule the image/video passes followed —
e.g. chroma-radiance → `Comfy-Org/Chroma1-Radiance_Repackaged`).

Once you've confirmed a source downloads and loads, add it to `ModelCatalog.cs` as a `ModelAsset` — same
record shape every image/audio entry already uses (`Repo`/`RepoPath`/`TargetSubdir`/`Role`/`Sha256`, `Role =
"transformer"` for the GGUF itself). VLM entries additionally need their mmproj sidecar as a second asset
(`TextService.FindMmproj` auto-discovers any `*.gguf` containing "mmproj" next to the text model — see its
doc comment for the exact matching rule).

## The auto-download mechanism works the same as image/audio — no LLM-specific plumbing needed

`TextCommand.cs` → `CommandRunner.Run(Modality.Text, ...)` is the same code path every other modality command
uses. `ModelAcquisition.EnsurePresent` / `ModelDownloader` are modality-agnostic — populating `Assets` on a
Text `CatalogEntry` will work exactly like it does for image models. One real difference: `TextService.
LoadInto` requires a **local `.gguf` file path** (`spec.LocalPath`) — there is no safetensors-preset path for
LLMs the way there is for Diffusion, so whatever you download must resolve to a `.gguf` under
`<ModelsRoot>/LLM/<id>/` (check `RepoPaths.ModelsRoot()` / `ModelResolver.cs` for the exact resolution rule
before assuming a directory layout).

## Verification methodology — read the memory first, same rule as image/video

Read `adversarial-image-output-verification.md` in this project's memory before starting (same file the video
handoff pointed at). Short version: **never mark a model `CliDrivable` stays-true or `Status = Verified`
just because generation completed without crashing or throwing.** Actually read the output and judge it
against the prompt.

LLM-specific verification notes:
- For every dense/MoE/SSM entry: run `hartsy text -m <id> "<prompt>"`, read the actual completion, and judge
  coherence/relevance — a model that loads and emits grammatical-but-wrong-language garbage, or that never
  stops, is not verified just because the process exited 0.
- For every VLM entry (`llama32-vision`, `gemma3-vision`, `smolvlm2`, `llava15`, `qwen25-vl`): run with
  `--image <path>` against a real test image (a solid-color square, a simple photo — whatever's easiest to
  judge objectively) and confirm the answer actually describes what's in the image, not a generic/hallucinated
  response. The `--image` flag's decode-and-attach path is new code with zero real-model exercise so far.
- For `--thinking`/`--no-thinking` on Qwen3/Qwen3.5: confirm the rendered prompt (or the model's raw output)
  actually differs between the two — e.g. Qwen3's real template typically emits a `<think>` scaffold
  differently depending on `enable_thinking`. If output is identical either way, the flag isn't actually
  reaching the template for that model and needs debugging, not just noting.
- For MoE entries that fit (`olmoe`, `granite-moe` first — smallest): confirm coherent output and note
  latency; MoE-specific bugs (wrong router dispatch, garbled output at higher context) are exactly the class
  of thing a synthetic/slice test can't catch.
- Update `docs/Checklists/MODEL_STATUS_LLM.md` and `PARITY_VERIFICATION.md` per model as you go, same style
  as the image/video passes — "CLI catalog-path verified `<date>`" notes with the actual prompt/output
  observed, honest documentation of any real limitation found (context-length ceilings, prompting quirks,
  VRAM limits) rather than a blanket flip to `Verified`.

## Practical constraints

- The build-defer entries (`mixtral`, `qwen3-moe`, `deepseek-v2-lite`, `deepseek-v3`, `gpt-oss`, `gemma4-moe`)
  are all >12GB — `deepseek-v3`/Kimi-K2 in particular (671B/1T total params) will likely need more than one
  GPU even quantized. Don't burn time trying to force these onto a single mid-range card; confirm they at
  least *load* if you have the VRAM, and be honest in the status doc if you don't.
- GPU is shared with other concurrent sessions on this project's boxes — check `nvidia-smi` before every run,
  don't evict another session's work, same rule as every prior handoff.
- LLM host-RAM headroom matters more than for diffusion: `TextService.EnsureRamHeadroomFor` refuses to load
  a GGUF unless free RAM is ≥2.5× the file size (dequantization headroom) — a large model can fail here before
  ever touching the GPU. If you hit this, it's the loader protecting you from an OOM-kill, not a bug to route
  around.

## Suggested order of attack

1. The 3 entries with a documented-but-unconfirmed source first (`qwen25-vl`, `gemma3-vision`,
   `llama32-vision`) — cheapest path to a first working result, and immediately exercises the new `--image`
   flag on real VLM output.
2. Small dense entries next (`gemma`, `phi`, `stablelm2`) to validate the catalog + new sampler-flag wiring
   broadly across different chat-template shapes (Gemma's own template, Phi's, etc.).
3. `qwen2`/`qwen3` already work — use them first to verify `--thinking`/`--no-thinking` actually changes
   output on a real Qwen3 GGUF before trusting the flag on anything else.
4. Small MoE (`olmoe`, `granite-moe`) before attempting any build-defer MoE model.
5. SSM/hybrid (`mamba`, `rwkv`, `qwen35`) — note these use a materially different generation path
   (`SsmGenerationPipeline`, not `TextGenerationPipeline`), worth confirming streaming/stop-token behavior
   works the same as the transformer path.
6. `glm4`/`gpt2`/`starcoder2` — genuinely unverified, treat as "does this even load" experiments.
7. Build-defer large models only if VRAM allows.
8. Revisit the T5/seq2seq gap (point 5 above) only if everything above is done and you want to take on real
   design work, not a quick win.

## Directives that carry over unchanged

- Never fabricate or guess a Hugging Face URL — only use one you actually found via a live search; if the
  canonical repo is gated, find an ungated mirror instead of asking to escalate.
- Never mark a model verified without reading the actual output and judging it against the prompt.
- No `git commit`/`push`/`merge` unless explicitly asked — edit in place.
- GPU is shared — check `nvidia-smi`, don't evict another session's work.
- Follow `docs/CODE_STYLE.md` for any code changes (explicit types, no `var`, file-scoped namespaces, etc.).
