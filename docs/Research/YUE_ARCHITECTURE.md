# YuE — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (YuE pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

**YuE** (乐, "music/joy") is an open-source full-song music-generation foundation model from HKUST + M-A-P (Multimodal Art Projection), released January 2025 with the full technical report published in March 2025 ([arXiv:2503.08638](https://arxiv.org/abs/2503.08638)). It is the open-source answer to Suno/Udio: given **lyrics + a free-form style/genre tag** (and optionally a reference audio prompt), it generates a **multi-minute full song** with separately-modeled **vocal and accompaniment** tracks, in English, Mandarin, Cantonese, Japanese, or Korean.

Architecturally YuE is a **two-stage LLaMA-based autoregressive pipeline**:

1. **Stage-1 (S1) — 7B LLaMA-2 decoder, "track-decoupled next-token prediction":** consumes lyrics + genre/style text + (optional) reference-audio tokens, and emits an interleaved stream of **codebook-0** audio tokens for the **vocal track** and **accompaniment track** at the X-Codec frame rate (50 Hz). This is the slow, expensive stage and the one that holds long-context musical structure.
2. **Stage-2 (S2) — ~1.5B LLaMA-based decoder:** takes the codebook-0 (semantic) tokens from S1 and autoregressively predicts the remaining residual codebooks 1…7, "upsampling" the bottom-level semantic tokens to a full 8-codebook representation that can be decoded back to waveform.
3. **Codec decode:** X-Codec (`xcodec_mini_infer`) decodes 8-codebook tokens to 16 kHz waveform. An optional **YuE-upsampler** (a Vocos-style super-resolution vocoder) lifts the output to **44.1 kHz**.

YuE's key innovations are (a) **track-decoupled next-token prediction (Dual-NTP)** — vocal and accompaniment tokens are interleaved at each frame so the model never has to disentangle mixed signals; (b) **structural progressive conditioning** — lyrics are presented section-by-section with `[verse]/[chorus]/[bridge]` tags so the LM can maintain long-context lyrical alignment; (c) **CoT vs ICL recipes** — Chain-of-Thought variants emit explicit per-section structure plans before audio tokens, while In-Context Learning variants take an audio "demonstration" as the seed for style transfer and continuation.

This file covers the **YuE model architecture and pipeline**. The X-Codec audio tokenizer config is cross-referenced in [AUDIO_CODECS.md](AUDIO_CODECS.md). The LLaMA decoder is reused from the native `HartsyInference.LLM` package. The Vocos super-resolution vocoder shares ideas with [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md).

Sources: [YuE paper (arXiv:2503.08638)](https://arxiv.org/abs/2503.08638), [YuE GitHub](https://github.com/multimodal-art-projection/YuE), [YuE demo page](https://map-yue.github.io/), [HKUSTAudio HF collection](https://huggingface.co/collections/HKUSTAudio/yue-679a2dedc6bce3aaef2953e1), [m-a-p/YuE-s1-7B-anneal-en-cot](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot), [m-a-p/YuE-s2-1B-general](https://huggingface.co/m-a-p/YuE-s2-1B-general), [m-a-p/xcodec_mini_infer](https://huggingface.co/m-a-p/xcodec_mini_infer), [x-codec (Codec Does Matter, AAAI 2025)](https://github.com/zhenye234/xcodec), [X-Codec project page](https://x-codec-audio.github.io/), [Spheron deployment guide](https://www.spheron.network/blog/deploy-open-source-ai-music-generation-gpu-cloud-2026/), [WhiteFiber YuE write-up](https://www.whitefiber.com/blog/yue-ai-music-generator), [YuEGP (deepbeepmeep)](https://github.com/deepbeepmeep/YuEGP), [YuE-exllamav2](https://github.com/sgsdxzy/YuE-exllamav2), [YuE-extend (Mozer)](https://github.com/Mozer/YuE-extend), [ACE-Step paper](https://arxiv.org/html/2506.00045v1).

## Model Variants

All released YuE checkpoints are mirrored under both the `m-a-p/*` and `HKUSTAudio/*` HuggingFace organizations (identical weights, different namespace). All weights are **bf16 safetensors** under Apache 2.0.

| HF Path (m-a-p) | Stage | Params | Language | Mode | Use | Approx. file size (bf16) |
|---|---|---|---|---|---|---|
| `YuE-s1-7B-anneal-en-cot` | S1 | ~6.9 B | English | **CoT** | Lyrics-to-song with explicit chain-of-thought structure tokens | ~14 GB |
| `YuE-s1-7B-anneal-en-icl` | S1 | ~6.9 B | English | **ICL** | Lyrics-to-song with an audio "demonstration" prompt (style transfer, continuation) | ~14 GB |
| `YuE-s1-7B-anneal-zh-cot` | S1 | ~6.9 B | Mandarin / Cantonese | CoT | Chinese lyrics-to-song | ~14 GB |
| `YuE-s1-7B-anneal-zh-icl` | S1 | ~6.9 B | Mandarin / Cantonese | ICL | Chinese ICL / continuation | ~14 GB |
| `YuE-s1-7B-anneal-jp-kr-cot` | S1 | ~6.9 B | Japanese / Korean | CoT | JP/KR lyrics-to-song | ~14 GB |
| `YuE-s1-7B-anneal-jp-kr-icl` | S1 | ~6.9 B | Japanese / Korean | ICL | JP/KR ICL / continuation | ~14 GB |
| `YuE-s2-1B-general` | S2 | ~1.5 B | language-agnostic | — | Residual-codebook upsampler (shared across all S1 variants and languages) | ~3 GB |
| `YuE-upsampler` | post | small | — | — | Vocos-style 16 kHz → 44.1 kHz spectral super-resolution vocoder | ~0.3 GB |
| `xcodec_mini_infer` (`m-a-p/xcodec_mini_infer`) | codec | small | — | — | X-Codec encoder + decoder + (optional) RepCodec semantic adapter; ~600 MB | ~0.6 GB |

**Implementation note.** Although every S1 checkpoint is named "7B", the actual parameter count is closer to **6.9 B** (matching standard LLaMA-2-7B). Likewise S2's "1B" name corresponds to roughly **1.5 B** parameters (the HF card on `YuE-s2-1B-general` reports 2 B with safetensors metadata, but the reported architecture is the smaller variant — for VRAM planning, assume ~1.5 B). A single full pipeline needs **S1 + S2 + xcodec + (optional) upsampler** loaded together, ≈ **18 GB bf16** all-in.

There is a unified Hugging Face Space ([Nymbo/YuE](https://huggingface.co/spaces/Nymbo/YuE/blob/main/inference/infer.py)) and several third-party redistributions (GGUF quantizations, ExLlamaV2 8-bit, YuEGP, YuE-for-windows, ComfyUI_YuE), but the canonical weights remain the `m-a-p/HKUSTAudio` bf16 safetensors.

## Lyrics Format

YuE expects a single text file containing **section-tagged lyrics**, paired with a separate **genre/style tag** file. The reference example files are at [`prompt_egs/lyrics.txt`](https://github.com/multimodal-art-projection/YuE/tree/main/prompt_egs) and `prompt_egs/genre.txt`.

**Genre file (`genre.txt`)** — one line, free-form descriptors. The recommended five-component recipe (per the GitHub README):

```
<genre> <instrument> <mood> <gender> <timbre>
```

Example (from the README):

```
inspiring female uplifting pop airy vocal electronic bright vocal vocal
```

Other concrete examples from the demo page:

```
Bass Metalcore Thrash Metal Furious bright vocal male Angry aggressive vocal Guitar
emotional piano slow ballad sad female warm vocal
```

**Lyrics file (`lyrics.txt`)** — each section starts with a structure tag in square brackets, followed by the lyrics for that section. Sections are separated by **two newlines** (one blank line between sections):

```
[verse]
Step back cause I'll ignite
Won't quit without a fight
No room for the weak inside

[chorus]
Hot flame burns within
This is where my story begins
Won't back down won't give in

[verse]
Another wave another rise
Looking straight ahead with steel eyes
…

[chorus]
Hot flame burns within
…

[bridge]
Quiet now before the storm
…

[outro]
This is my fire
```

Rules from the README:

* Allowed section tags: `[verse]`, `[chorus]`, `[bridge]`, `[intro]`, `[outro]`. `[intro]` is *"less stable"* — prefer starting on `[verse]` or `[chorus]`.
* Sections are separated by exactly **two `\n` characters** (one blank line).
* Each section is one "session" — the model generates approximately **30 seconds** of audio per session.
* `--run_n_segments N` tells the inference script how many sections to actually render (so a 6-section lyrics file with `--run_n_segments 2` will only render the first two).
* Language must match the chosen S1 checkpoint (use `en-cot/en-icl` for English lyrics, `zh-*` for Chinese, `jp-kr-*` for Japanese or Korean).

## CoT vs ICL Variants

The "anneal" suffix on every S1 checkpoint indicates that these are the **annealing-stage** weights (the final, fine-tuned policy). Each language ships in two flavors:

#### CoT (Chain of Thought) — `*-cot`

Before emitting any audio tokens for a section, the model produces an **explicit textual plan** for that section (think: "a bright fast-tempo pop chorus, female vocals, layered synths, kick on 1-and-3"), then emits the audio tokens conditioned on that plan. CoT was the first paradigm released (January 2025) and gives the most controllable per-section musical structure with no extra inputs.

Use CoT when: you only have lyrics + a genre string and want maximum prompt-following. Trade-off: lower aesthetic ceiling — the model has to invent the musical idea from scratch.

#### ICL (In-Context Learning) — `*-icl`

The prompt includes a **demonstration audio clip** (typically 30 s of a real song), tokenized through X-Codec and inserted into the context as the "answer" to a phantom first prompt. The model then continues in the same style. The ICL variants additionally support **dual-track ICL** — instead of a single mixed reference, you supply two parallel files (`--vocal_track_prompt_path` and `--instrumental_track_prompt_path`) and the model picks up both track styles separately.

The paper's evaluation (Table 4 / §5) reports:

| Variant | Win rate vs baseline |
|---|---|
| CoT only | 0.21 |
| ICL only | 0.63 |
| **ICL + CFG (Classifier-Free Guidance)** | **0.79** |

So **ICL + CFG is the recommended mode** for best output quality. CFG here is the standard "drop the genre/lyrics conditioning at probability `p` during training, then at inference combine conditional & unconditional logits with guidance scale `w`" (cross-ref: [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md)). YuE's CFG implementation lives in the ExLlamaV2 fork; the upstream HF implementation does not enable CFG by default.

ICL also unlocks **music continuation** — feed an existing 30 s clip and let YuE extend it, and **voice cloning / style transfer** — feed an a-capella vocal as the prompt.

## Sampling

Defaults from the reference `infer.py` (per the YuE GitHub quick-start and Hugging Face Space mirror):

| Parameter | Default | Notes |
|---|---|---|
| `--max_new_tokens` | **3000** | Per S1 session; corresponds to ~30 s of audio (50 Hz × 30 s × 2 tracks = 3000 tokens). |
| `--repetition_penalty` | **1.1** | Standard HF `RepetitionPenaltyLogitsProcessor`. Critical for music — without it, the LM falls into instrumental loops. |
| `--stage2_batch_size` | **4** | How many ~30 s S2 chunks to batch in parallel. Increase for throughput, decrease for VRAM. |
| `--run_n_segments` | **2** | How many lyric sections to render (default is short for quick demos). |
| `temperature` | **1.0** (implicit) | Reference script uses HF `generate()` defaults; community recipes commonly run T=0.85–1.0. |
| `top_p` | **0.93** (community-validated) | Not exposed as a CLI flag in mainline; ExLlamaV2 fork uses `top_p=0.93`, `top_k=50`. |
| `top_k` | **50** | Community default in ExLlamaV2 fork. |
| `guidance_scale` (CFG) | **1.5** (in ExLlamaV2 fork) | Only meaningful when the model was trained with CFG dropout (ICL variants). |

The reference `infer.py` does **not** expose temperature/top_p/top_k as CLI flags — it relies on the HF `generation_config.json` shipped with each checkpoint. Implementers should read those values from disk rather than hardcoding.

**Stop conditions.** Generation stops on either (a) `[end_of_segment]` token, (b) `--max_new_tokens` reached, or (c) the LLaMA EOS token. For long-form (multi-section) generation, the script invokes S1 once per section in a loop, carrying the previous session's audio tokens as context.

## Comparison to ACE-Step

| Dimension | **YuE** | **ACE-Step** |
|---|---|---|
| Generative paradigm | Autoregressive LLaMA decoder (next-token over codec tokens) | Flow-matching diffusion (continuous-time FM) over latent audio |
| Backbone | LLaMA-2 7B (S1) + LLaMA 1.5B (S2) | ~3.5 B linear-transformer over Sana DCAE latents |
| Audio repr | X-Codec (8 cb @ 50 Hz, 16 kHz) | Deep Compression AutoEncoder latents |
| Lyric alignment | **Strong** — section tags, dual-track NTP, structural CoT | Weaker — diffusion produces less explicit structure |
| Track separation | **Built-in** — vocal & accompaniment generated as parallel streams | Single mixed stream |
| Style transfer | ICL with audio demonstration; CFG-able | Prompt-based; very direct style steering |
| Speed (5 min song, A100) | ~12–15 min | **< 30 s** (>10× faster) |
| Min VRAM | ~8 GB quantized, 18 GB bf16 | ~8 GB; only viable model on a 12 GB consumer GPU |
| Editability | Per-track stems can be remixed | Single stream; harder to edit |
| Failure modes | Repetition loops; long-form drift; expensive | Less structural coherence on 5+ min; "muddier" lyric pronunciation |
| License | Apache 2.0 | Apache 2.0 |

**When to pick YuE.** You care about *what the song actually says*, want **separate vocal and instrumental stems**, can afford 10+ minutes of GPU time per song, and want maximum control over per-section structure via lyrics tagging. YuE is the right choice for "make this exact song from these exact lyrics", and the only open model that natively gives you the vocal stem separately for re-mixing.

**When to pick ACE-Step.** You want fast iteration (try 50 variations of a song in the time YuE generates one), are willing to accept less precise lyrical adherence, and care more about overall vibe / production than fine structural control. ACE-Step is the right default for casual / interactive use, especially on consumer hardware.

In HartsyInference, both pipelines should coexist in `HartsyInference.Audio.Music` with a shared `IMusicGenerator` interface — they share the codec-decode → vocoder back-end but differ in the LM core.

## C# Implementation Notes (HartsyInference)

This section is the implementer's bridge.

**Reuse the native `HartsyInference.LLM` patterns aggressively.** Both S1 and S2 are stock LLaMA-2 decoder architectures with vocab/RoPE-scaling tweaks. The native `HartsyInference.LLM` package already implements LLaMA. The right plan is:

1. **`HartsyInference.Models.Llama`** — already needed for the native LLM modality. Build it once, parameterize for: hidden size, layers, heads, KV heads (GQA), FFN dim, RoPE base + scaling factor, vocab size.
2. **`HartsyInference.Audio.YuE.YuETokenizer`** — wraps the LLaMA BPE tokenizer plus the 1024 audio-cb0 IDs and ~64 control tokens. The audio IDs are pure integer offsets — no embedding fancy footwork; LLaMA's input-embedding table just has to be sized to the expanded vocab.
3. **`HartsyInference.Audio.YuE.YuES1Stage`** — owns the S1 model instance, the section parser, the per-section loop, the cb0-extraction post-processor.
4. **`HartsyInference.Audio.YuE.YuES2Stage`** — owns the S2 model instance, the cb0-prefix packing, batched chunk inference.
5. **`HartsyInference.Audio.XCodec.XCodecDecoder`** — 8-codebook RVQ table lookup (vector add) followed by a transposed-conv decoder. Cross-ref the GAN-vocoder code in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) — the architecture is closely related (DAC/EnCodec style). PTX kernel needs: transposed conv1d, leaky-ReLU/Snake activation, weight-normed conv1d.
6. **`HartsyInference.Audio.YuE.VocosUpsampler`** — optional ConvNeXt + ISTFT. Reuses STFT/ISTFT from `HartsyInference.Audio.Dsp` (already needed for Whisper/Kokoro per `MODEL_STATUS_AUDIO.md`).

**Key implementation gotchas:**

* **Long context with KV cache.** S1 needs to handle ~16 K tokens of context per song. At bf16, a per-layer KV cache for LLaMA-7B is `2 × 32 layers × 32 heads × 128 head_dim × seq_len × 2 bytes` = 524 288 × seq_len bytes ≈ **8 GB at 16 K**. Plan for this in `TensorRef` allocation. RoPE scaling beyond 4 K must be applied at *generation time* via a θ rescale (NTK-aware or LongRoPE-style — verify against the released `config.json`'s `rope_scaling` field).
* **Interleaved dual-track sampling.** S1 emits `v_t, a_t, v_{t+1}, a_{t+1}, …`. No special kernel is needed — it's just a single autoregressive stream with the convention that even positions (after the prompt) are vocal and odd are accompaniment. The vocab is *not* track-segregated; the model is trained to "know" which slot it is in via position and prior context. So in C#, the audio-token post-extraction is just: take every audio-cb0 token, even indices go to the vocal cb0 array, odd to the accompaniment cb0 array.
* **Two-stage orchestration.** S1 must run to completion before S2 starts. They can share the *same* GPU memory pool (`UnmanagedTensor` allocator) by loading both weight sets and never co-running. Or — for >24 GB cards — both can stay resident.
* **CFG (only for ICL variants).** Standard CFG: run a "conditional" forward and an "unconditional" forward (genre/lyric tokens replaced with the null prompt), combine: `logits = logits_uncond + w × (logits_cond - logits_uncond)` where `w ≈ 1.5`. This doubles per-step compute; only enable when targeting maximum quality. Cross-ref [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md).
* **Repetition penalty is mandatory.** Without `repetition_penalty=1.1`, S1 will fall into looping instrumental phrases — every implementer who has skipped it reports this. Implement it via the standard HF formula: divide logits of tokens that appear in the prior context by `1.1`.
* **Codec decode is per-track.** Decode the vocal stream and the accompaniment stream *separately* through X-Codec, then mix. Do not try to decode "interleaved as one waveform" — that's not what X-Codec expects.
* **GGUF / quantization path.** The community has already shipped GGUF S1 builds; once `HartsyInference.GGUF` supports LLaMA-2 7B (which it must for `HartsyInference.LLM`), YuE quantized inference is free. See [GGUF_BACKEND.md](GGUF_BACKEND.md) and [QUANTIZATION_DIFFUSION.md](QUANTIZATION_DIFFUSION.md) (note: diffusion is a different quantization story, but the GGUF format/loader is shared).
* **No Python deps.** The reference pipeline pulls in `omegaconf`, `descript-audio-codec`, `transformers`. None survive — we re-implement the codec decoder from the X-Codec architecture spec and load weights from the bf16 safetensors directly (`HartsyInference.Safetensors`).

**Validation plan** (in line with the project rule "validate against references"):

1. Ship the official `m-a-p/YuE-s1-7B-anneal-en-cot` and run a fixed-seed generation against the reference Python pipeline on the same prompt. Compare token sequences for the first ~100 generated tokens — they should be **bitwise identical** (or within fp atol if quantization is involved).
2. X-Codec decode: encode-decode a 16 kHz test waveform through the Python `xcodec_mini_infer` and the C# port. PSNR > 30 dB target.
3. Full pipeline: generate one full 30 s `[chorus]` from the README's example prompt in both implementations. Compare via STFT magnitude correlation > 0.95.

## TL;DR for Implementers

* **YuE = LLaMA-2 7B (S1) + LLaMA 1.5B (S2) + X-Codec (8 cb @ 50 Hz, 16 kHz) + optional 44.1 kHz Vocos upsampler.**
* **Inputs:** lyrics with `[verse]/[chorus]/...` tags + genre string + (optional) reference audio.
* **Outputs:** 16 kHz (or 44.1 kHz) WAV, optionally with separate vocal & accompaniment stems.
* **Dual-track trick:** at every codec frame, S1 emits a vocal token then an accompaniment token — a single interleaved stream, no special kernel.
* **Long songs:** explicit section-by-section autoregressive looping, not one giant forward pass.
* **Two flavors per language:** CoT (no extras) and ICL (with audio demo prompt; CFG-capable; best quality).
* **VRAM:** ~18 GB bf16 full pipeline; ~12 GB at 8-bit; ~8 GB at 4-bit.
* **Speed:** ~12 min on A100, ~25 min on 4090 for a 5-min song. Realtime is not on offer.
* **For C#:** build LLaMA-2 once (shared with `HartsyInference.LLM`), build X-Codec decoder once (shares the GAN-vocoder kernels with Kokoro), then YuE is glue code on top.
