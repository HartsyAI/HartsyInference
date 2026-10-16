# Fish Audio Drama 3 — architecture and implementation

Source snapshot: 2026-10-16. Primary sources: fish-speech `fish_speech/models/text2semantic/fish_omni/*` at 5415669 (model,
config, chunking, generate and their tests), sglang-omni `fishaudio_drama3/*` and vllm-omni `fish_omni` tests for the serving
behavior. The released checkpoints are `fishaudio/drama-3` (block-FP8 experts) and `fishaudio/drama-3-int4` (GPTQ), under the Fish
Audio Research License (non-commercial). This note records what the engine implements and what is still unverified.

## Model

`config.json` has `model_type: fish_omni` and `speech_template_version: reference-bracket-v1`. The engine validates the same fixed
flags upstream's `FishOmniConfig` does and refuses a checkpoint that changes any of them.

**Backbone.** A Qwen3.5-MoE hybrid text model under `model.language_model.*`, with a separate `lm_head.weight`.

- Layers alternate Gated DeltaNet (`linear_attention`) and gated full attention (`full_attention`), as `layer_types` lists them.
- RMSNorm is zero-centered: the weight is applied as `1 + w`.
- Every layer's FFN is a mixture of experts: softmax top-k routing, renormalized over the chosen experts, plus a shared expert behind
  a sigmoid gate.
- Routed experts are stored per expert (`experts.{e}.{gate,up,down}_proj.weight`) or fused (`experts.gate_up_proj` `[E, 2I, H]`,
  `experts.down_proj` `[E, H, I]`).

**Backbone input.** Each prompt row is a text token plus, on audio rows, codes. The row's embedding is the text embedding plus the
embeddings of codebooks 1..K (`vq_backbone_codebooks`); on semantic rows the sum is divided by `sqrt(1 + K)`.

**Codebook tower** (`audio_decoder*`), which replaces S2's fast transformer. It has `shared_enrichment_layers` shared layers and
`num_codebooks - 1` residual layers. Each layer:

1. Embeds the previous code: the semantic embedding for the shared layers, the codebook embedding for the residual ones, zero on a
   non-semantic row.
2. Mixes it with the running hidden state: `eh_proj([enorm(e); hnorm(h)])`.
3. Runs one pre-norm transformer layer. Attention is gated: `wqkv` rows are `[Q | Gate | K | V]`, Q and K get a zero-centered norm,
   RoPE is interleaved and partial, and a sigmoid of the gate multiplies the attention output before `wo`. The FFN is SwiGLU.
4. Applies `final_layernorm`.

Each tower layer keeps its own KV cache across frames. Prefill teacher-forces the tower over the prompt with each row's next-frame
codes, so its caches hold the prompt before the first new frame.

**Sampling** (`decode_step`).

- The main token is drawn over eos plus the semantic band only.
- Every draw applies temperature first, then the top-k and top-p cut.
- Defaults: temperature 0.7, top-p 0.7, top-k 30.
- Repetition-aware sampling: a semantic draw that repeats one of the last 16 main tokens (prompt rows included) is redrawn at
  temperature 1.0, top-k 30, top-p 0.95.

**Prompt** (`reference-bracket-v1`).

- The system turn says `convert the provided text to speech`, plus ` reference to the following:` and one `[Reference N]` block per
  voice reference (`Text: <transcript>` / `Speech: <codes>`).
- History turns follow: user text, then the assistant's `<|voice|>` codes.
- Generation starts at `<|im_start|>assistant\n<|voice|>`.
- Each literal is tokenized on its own and NFC-normalized.
- Speaker tags are `<|speaker:0|>` .. `<|speaker:15|>`. Structural tokens in user text are refused.
- A prompt may be at most 8192 rows. A single pass generates at most 2048 frames.

**Chunking and history** (`chunking.py`, `generate_long_drama3`).

- Text is cut into chunks of about 300 UTF-8 bytes with at most 3 speakers each.
- Special tokens, paired tags and `[bracket cues]` are never split.
- Each chunk generates at most 600 frames.
- Each chunk carries the conversation so far: the first chunk pinned, plus the latest one. Older chunks are dropped first when the
  prompt would pass its row limit.

**Codec.** S2's ModifiedDAC (`codec.pth`): 10 codebooks, 2048 samples per frame, 44.1 kHz mono. Upstream decodes with
`fishaudio/s2-pro/codec.pth` when a Drama 3 repo does not ship its own. The decoder is causal, so decoding the frames so far ends
exactly where the full decode would. That is what streaming relies on.

## Checkpoint builds and how each loads

| Build | Stored as | Loads as (native) | Exactness |
|---|---|---|---|
| Unquantized original | BF16 | BF16, borrowed from the mapped file | exact |
| `fishaudio/drama-3` | Experts FP8 e4m3, F32 `weight_scale_inv` in 128x128 blocks (`bsgemm-moe`); the rest BF16 | Experts Q8_0 | error reported at load |
| `fishaudio/drama-3-int4` | Routed experts and attention projections GPTQ: symmetric 4-bit, group 128, no `g_idx`; the rest BF16 | Q4_0 | bit-exact |

GPTQ maps onto Q4_0 exactly. Each 128-wide group scale becomes four 32-wide Q4_0 blocks with the same scale, and the zero point is 8,
which Q4_0 also uses. A `g_idx` that is not the identity, or asymmetric zeros, is refused.

The `bf16`, `q8`, `q6`, `q5` and `q4` precisions re-quantize the experts and dense projections of any build. At `q5` and `q4` the
attention projections stay Q6_K. Norms, router, embeddings, head and tower stay as stored.

Any conversion is written once to a GGUF cache under `<audio cache>/tts/_gguf-cache/`, one tensor at a time, and mapped from there
afterwards. A weight row that is not a whole number of blocks falls back to Q8_0, then to its stored float type.

## Engine map

| Piece | Code |
|---|---|
| Config and fixed-flag checks | `Audio/Models/FishAudio/FishAudioDrama3Config.cs` |
| Backbone | `LLM/Ssm/Qwen35Model*.cs` (HF Qwen3.5-MoE tier), `LLM/Transformer/StackedExperts.cs` |
| Tower | `Audio/Models/FishAudio/FishAudioDrama3Tower.cs` (single-layer `GenericTransformer` blocks with an attention output gate) |
| Generate / score loop | `Audio/Models/FishAudio/FishAudioDrama3Lm.cs` |
| Prompt, text rules, chunking, history | `FishAudioDrama3Prompt.cs`, `FishAudioDrama3Text.cs`, `FishAudioDrama3Chunking.cs`, `FishAudioDrama3History.cs` |
| Sampling | `FishAudioDrama3Sampler.cs` over `FishAudioSampling` |
| Checkpoint reading | `ModelAssets/Checkpoints/HfQuantCheckpoint.cs`, `BlockScale/Fp8BlockF32Codec.cs`, `Quant/GptqInt4Codec.cs` |
| Precision policy and cache | `FishAudioDrama3Weights.cs`, `ModelAssets/Gguf/GgufQuantCache.cs` |
| Pipeline, streaming | `Audio/Pipelines/FishAudioDrama3Pipeline.cs` |
| Engine descriptor, variants, placement | `Engine/Audio/Tts/Models/FishAudioDrama3Model.cs` |
| Q4_0 routed experts | `Cuda/Kernels/moe/moe_id_q4_0.cu`, `Cpu/Moe/CpuExpertKernels*.cs` |

Engine-level behavior:

- **Variants.** `fishaudio-drama3[:tokens]`, with tokens joined by `+`: `int4` for the GPTQ build, `org/repo` for any other build of
  the same layout, plus at most one precision. A precision keys its own runner.
- **Placement.** On a GPU the model goes through `TextPlacementPlanner`: resident when it fits, otherwise the dense weights stay
  resident and the routed experts sit behind a CUDA expert cache, with misses running on the CPU (Q8_0 and Q4_0 have CPU expert
  kernels).
- **CPU.** A CPU backend reads no quantized weights, so it needs `:bf16`.
- **Cloning and revision.** `SpeechRequest.References` holds voice references, each a transcript with audio or codes. The single
  `Reference` / `RefText` pair is the one-reference form. `SpeechRequest.History` takes the `AudioResult.Turns` of an earlier take,
  so a take can be revised by sending it back with a new cue. On the CLI these are `speak --references`, `--history` and
  `--save-turns`.
- **Streaming.** Each chunk decodes progressively while it generates: at 16 frames, then at doubling spans of at most 256 frames.
  Every emission hands over only the samples past the previous one.

## Verification

Done, on CPU:

- `tools/fish_audio/drama3_reference.py` runs a tiny random checkpoint through upstream's `FishOmniBackbone` and
  `FishOmniAudioDecoder`. `FishAudioDrama3ParityTests` matches it:
  - prompt hidden states to 7.6e-6;
  - main-token logits to 1.6e-6;
  - teacher-forced tower logits to 7.2e-7;
  - the hidden state after a fed frame to 1.8e-6.
- Fused and per-expert layouts load identically. A chunked prefill is bit-identical to one pass.
- GPTQ to Q4_0 is bit-exact (`GptqInt4CodecTests`). The Q8 cache holds exactly the quantized weights.
- Streamed audio equals the one-shot audio across chunks. History and references change the take.
- The prompt and chunking rules have golden cases from upstream's tests.

Not done:

- The real checkpoints have not been run. Their shard names, the config numbers and whether each repo ships `codec.pth` are still to
  be recorded; the catalog entry lists only the shard index for now.
- No GPU run yet:
  - the Q4_0 MoE kernel (`moe_id_q4_0.ptx`, compiled with NVRTC against ISA 9.0) is covered by `CudaMoeIndexedTests` but has not
    been run on a device;
  - the expert offload path has not run;
  - the Vulkan path has not run.
- The Whisper audit of generated speech (gate sentences 01–05, seeds 0–3, every precision) has not been run.
