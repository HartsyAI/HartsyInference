# YuE — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (YuE pipeline)

## Summary

**YuE** (乐, "music/joy") is an open-source full-song music-generation foundation model from HKUST + M-A-P (Multimodal Art Projection), released January 2025 with the full technical report published in March 2025 ([arXiv:2503.08638](https://arxiv.org/abs/2503.08638)). It is the open-source answer to Suno/Udio: given **lyrics + a free-form style/genre tag** (and optionally a reference audio prompt), it generates a **multi-minute full song** with separately-modeled **vocal and accompaniment** tracks, in English, Mandarin, Cantonese, Japanese, or Korean.

Architecturally YuE is a **two-stage LLaMA-based autoregressive pipeline**:

1. **Stage-1 (S1) — 7B LLaMA-2 decoder, "track-decoupled next-token prediction":** consumes lyrics + genre/style text + (optional) reference-audio tokens, and emits an interleaved stream of **codebook-0** audio tokens for the **vocal track** and **accompaniment track** at the X-Codec frame rate (50 Hz). This is the slow, expensive stage and the one that holds long-context musical structure.
2. **Stage-2 (S2) — ~1.5B LLaMA-based decoder:** takes the codebook-0 (semantic) tokens from S1 and autoregressively predicts the remaining residual codebooks 1…7, "upsampling" the bottom-level semantic tokens to a full 8-codebook representation that can be decoded back to waveform.
3. **Codec decode:** X-Codec (`xcodec_mini_infer`) decodes 8-codebook tokens to 16 kHz waveform. An optional **YuE-upsampler** (a Vocos-style super-resolution vocoder) lifts the output to **44.1 kHz**.

YuE's key innovations are (a) **track-decoupled next-token prediction (Dual-NTP)** — vocal and accompaniment tokens are interleaved at each frame so the model never has to disentangle mixed signals; (b) **structural progressive conditioning** — lyrics are presented section-by-section with `[verse]/[chorus]/[bridge]` tags so the LM can maintain long-context lyrical alignment; (c) **CoT vs ICL recipes** — Chain-of-Thought variants emit explicit per-section structure plans before audio tokens, while In-Context Learning variants take an audio "demonstration" as the seed for style transfer and continuation.

This file covers the **YuE model architecture and pipeline**. The X-Codec audio tokenizer config is cross-referenced in [AUDIO_CODECS.md](AUDIO_CODECS.md). The LLaMA decoder reuse pattern is in [DOTLLM_ARCHITECTURE.md](DOTLLM_ARCHITECTURE.md). The Vocos super-resolution vocoder shares ideas with [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md).

Sources: [YuE paper (arXiv:2503.08638)](https://arxiv.org/abs/2503.08638), [YuE GitHub](https://github.com/multimodal-art-projection/YuE), [YuE demo page](https://map-yue.github.io/), [HKUSTAudio HF collection](https://huggingface.co/collections/HKUSTAudio/yue-679a2dedc6bce3aaef2953e1), [m-a-p/YuE-s1-7B-anneal-en-cot](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot), [m-a-p/YuE-s2-1B-general](https://huggingface.co/m-a-p/YuE-s2-1B-general), [m-a-p/xcodec_mini_infer](https://huggingface.co/m-a-p/xcodec_mini_infer), [x-codec (Codec Does Matter, AAAI 2025)](https://github.com/zhenye234/xcodec), [X-Codec project page](https://x-codec-audio.github.io/), [Spheron deployment guide](https://www.spheron.network/blog/deploy-open-source-ai-music-generation-gpu-cloud-2026/), [WhiteFiber YuE write-up](https://www.whitefiber.com/blog/yue-ai-music-generator), [YuEGP (deepbeepmeep)](https://github.com/deepbeepmeep/YuEGP), [YuE-exllamav2](https://github.com/sgsdxzy/YuE-exllamav2), [YuE-extend (Mozer)](https://github.com/Mozer/YuE-extend), [ACE-Step paper](https://arxiv.org/html/2506.00045v1).

## Detailed Findings

### 1. Model Variants

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

### 2. Architecture Overview

#### 2.1 Four-component framework

From the paper (§3): *"YuE comprises four main components: an audio tokenizer (with a lightweight upsampler), a text tokenizer, and two language models."*

```
                ┌──────────────────────────────────────────────────────────┐
                │  Lyrics text  +  genre/style tag  +  (opt) ref-audio    │
                └───────────────────────────┬──────────────────────────────┘
                                            │  BPE / X-Codec tokenize
                                            ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │  Stage-1 LM  (LLaMA-2 7B decoder, track-decoupled)             │
       │  Emits interleaved [vocal_cb0, accomp_cb0]  @ 50 Hz             │
       └───────────────────────────┬─────────────────────────────────────┘
                                   │  codebook-0 tokens (semantic level)
                                   ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │  Stage-2 LM  (LLaMA 1.5B decoder)                              │
       │  Takes cb0; emits residual codebooks 1..7 for each frame       │
       │  Operates in ~30-second windowed chunks (stage2_batch_size N)  │
       └───────────────────────────┬─────────────────────────────────────┘
                                   │  full 8-codebook stream  @ 50 Hz
                                   ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │  X-Codec decoder  →  16 kHz vocal wav  +  16 kHz accomp wav     │
       │  Mix (or keep separate)                                         │
       └───────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │  YuE-upsampler (Vocos)  →  44.1 kHz stereo song wav             │
       └─────────────────────────────────────────────────────────────────┘
```

#### 2.2 LLaMA base

Per the paper (§3 and Appendix), both stages start from a **LLaMA-2 architecture backbone** — same MLP, same RMSNorm, same RoPE, same GQA — re-initialized for music. S1 is the standard LLaMA-2-7B shape; S2 is a smaller LLaMA-shaped decoder (the paper calls it "a smaller LM"). Concrete shape parameters (verify in `config.json` at load time):

| Component | S1 (≈7B) | S2 (≈1.5B) |
|---|---|---|
| Hidden size | 4096 | ≈2048 |
| Layers | 32 | ≈24 |
| Attention heads | 32 | ≈16 |
| KV heads (GQA) | 32 (MHA, LLaMA-2 7B style) | ≈16 |
| FFN dim (SwiGLU) | 11008 | ≈5504 |
| RoPE base θ | 10000 (with LongRoPE-style scaling for >4 K context) | 10000 |
| Tied embeddings | no | no |
| Activation | SiLU/SwiGLU | SiLU/SwiGLU |

**Vocabulary expansion.** S1 starts from LLaMA-2's **32 000 BPE** tokens and **expands the vocabulary to include audio tokens and structural markers**. Concretely:

* Text BPE: 32 000 tokens (LLaMA tokenizer is reused unchanged).
* Audio tokens for codebook-0: **1024 entries**, added as 1024 new vocabulary IDs (the dual-track scheme uses the same 1024-entry set for vocal and accompaniment; the *position* in the interleave determines which track).
* Structural / control tokens: `[verse]`, `[chorus]`, `[bridge]`, `[intro]`, `[outro]`, `[start_of_segment]`, `[end_of_segment]`, plus dual-track markers `<SOA>` (start-of-accompaniment) and `<EOA>` (end-of-accompaniment), and the CoT-mode structure markers.

After expansion the effective S1 vocab is roughly **32 000 + 1024 + ≈64 control = ~33 100 entries** (verify exact count from `tokenizer.json` at load).

**Context length.** S1 is trained / annealed at **8K – 16K tokens**, with RoPE-scaled inference up to ~30 K tokens for ~5-minute generations (one minute ≈ 50 Hz × 60 s × 2 tracks = 6000 audio tokens plus lyrics/structure overhead). The official inference script chunks the song into ~30-second segments to stay within S1's effective context.

S2 has a much shorter operating context — it is run in **~30-second windowed chunks** (configurable via `--stage2_batch_size`) over the S1 output stream, not over the whole song at once.

#### 2.3 Audio tokenizer (X-Codec)

YuE uses **X-Codec** ([AAAI 2025, "Codec Does Matter"](https://github.com/zhenye234/xcodec)) via the `xcodec_mini_infer` HF repository. X-Codec is a semantic-aware neural audio codec: it concatenates **acoustic** features (from a DAC-style encoder) with **semantic** features (from a self-supervised audio encoder, e.g. HuBERT) **before** the Residual VQ stage, and adds a **semantic reconstruction loss after** the RVQ. The result: codebook-0 carries semantically-meaningful content (phonetic/melodic), the higher codebooks carry fine acoustic detail.

YuE's X-Codec config (per the paper §3.1 and `xcodec_mini_infer`):

| Property | Value |
|---|---|
| Sample rate | **16 kHz** |
| Frame rate | **50 Hz** (one frame per 20 ms = 320 samples per frame) |
| Number of RVQ codebooks | **8** (`n_q = 8`) — paper trains with up to 12, but YuE uses 8 at inference |
| Codebook size | **1024** entries per codebook (10-bit each) |
| Bandwidth | ≈ **4.0 kbps** (50 Hz × 8 codebooks × 10 bits) |
| Training data | ~200 K hours, music : speech : SFX ≈ 1 : 1 : 0.05 |
| Codebook-0 role | Semantic anchor; consumed by S1 only |
| Codebooks 1–7 role | Acoustic residuals; predicted by S2 |
| Streams | Two parallel encoders run — one for the vocal stem, one for the accompaniment stem |

Cross-ref: deeper codec mechanics live in [AUDIO_CODECS.md](AUDIO_CODECS.md). For C# we need (a) the X-Codec **decoder** (RVQ table lookup + transposed-conv neural vocoder) and (b) optionally the **encoder** if we support `--use_audio_prompt` ICL mode.

#### 2.4 Lightweight upsampler (`YuE-upsampler`)

The codec only outputs **16 kHz** audio. The optional `m-a-p/YuE-upsampler` post-processor is a **Vocos-style** spectral super-resolution model that lifts 16 kHz → **44.1 kHz** by predicting the high-frequency components. It is not strictly required (16 kHz output is musically usable) but is what the reference pipeline runs by default for the released demo songs. Architecturally it is a ConvNeXt encoder + ISTFT head, similar in spirit to the Kokoro iSTFTNet decoder (see [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md)).

### 3. Dual-Track Decoding (Track-Decoupled Next-Token Prediction)

The headline architectural choice. Most prior music LMs (MusicGen, MusicLM) generate a single **mixed** audio stream — the model is asked to emit codebook-0 tokens whose decoded waveform contains both voice and accompaniment simultaneously. This collapses on hard cases (e.g. low vocal-to-accompaniment ratio in metal), and prevents track-level editing.

YuE's solution: **at each codec frame `t`, emit two tokens — one for the vocal stream and one for the accompaniment stream — in a fixed interleaved order.**

**Stage-1 stream layout** (single-track ICL or CoT mode):

```
[prompt tokens]  v_0  a_0  v_1  a_1  v_2  a_2  …  v_{T-1}  a_{T-1}  [EOS]
                 └─ vocal cb0 ─┘└─ accomp cb0 ─┘
                 (frame 0 = 20 ms)
```

Per the paper: *"At each time step t, the model outputs two tokens: a vocal token (v_t) and an accompaniment token (a_t)."* The vocal token always precedes the accompaniment token at each frame, and Stage-1 only ever emits **codebook-0** values for both tracks (the higher codebooks are filled in by S2).

**Stage-2 stream layout** (per ~30-second chunk):

S2's training packing puts **all of the codebook-0 tokens first** (the semantic backbone from S1), then emits per-frame blocks of the remaining codebooks. The paper notes: *"By placing all codebook-0 tokens at the beginning, the model is guaranteed to 'see' the entire semantic structure before it encounters any mixed (0–7) blocks. This allows the model to plan the later residuals by attending to a complete semantic outline from Stage-1."*

A simplified S2 chunk:

```
chunk = [cb0_v_0, cb0_a_0, cb0_v_1, cb0_a_1, … cb0_v_{N-1}, cb0_a_{N-1}]   # ~1500 tokens (30s × 50Hz × 2)
       + [<SOA>]                                                            # boundary marker
       + per-frame interleave of cb1..cb7 for vocal & accompaniment
       + [<EOA>]
```

**Special tokens summary (S1 and S2 combined):**

| Token | Purpose |
|---|---|
| `[verse]`, `[chorus]`, `[bridge]`, `[intro]`, `[outro]` | Section-structure markers in the lyrics |
| `[start_of_segment]`, `[end_of_segment]` | Chunk boundaries for progressive long-form generation |
| `<SOA>` (start-of-accompaniment) | Marks the transition in S2's stream from semantic-only block to mixed-codebook block |
| `<EOA>` (end-of-accompaniment) | Closes the mixed block |
| `<|im_start|>`, `<|im_end|>` | LLaMA-style turn markers used in the chat-formatted prompt |
| Per-codebook offset | Each codebook k has its 1024 entries placed at a distinct offset in the joint vocab to avoid collision |

### 4. Lyrics Format

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

### 5. CoT vs ICL Variants

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

### 6. Sampling

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

### 7. Long Generation (5+ minute songs)

YuE handles long-form by **explicit section-by-section chunking**, not by a single 5-minute autoregressive pass. The mechanism is what the paper calls **structural progressive conditioning**:

1. **Parse lyrics** into sections separated by `\n\n`. Each section is one autoregressive "session" of ~30 s.
2. **Session 0** (typically the first `[verse]`):
   * Prompt = `<|im_start|> system + genre tag + lyrics_for_section_0 [start_of_segment]`
   * S1 generates up to `max_new_tokens=3000` interleaved vocal+accompaniment cb0 tokens.
   * Stop at `[end_of_segment]`.
3. **Session N** (subsequent sections):
   * Prompt = `<|im_start|> system + genre + ALL prior lyrics + ALL prior audio cb0 tokens (or a sliding window of them) + lyrics_for_section_N [start_of_segment]`
   * Continue autoregressively.
   * Context grows linearly with section count; once total context approaches ~16 K tokens, the prior audio context is **windowed** (keep the most recent ~10 s of audio tokens + all lyric tags).
4. **After all S1 sessions finish:** concatenate the full cb0 stream. Pass it through S2 in batched ~30 s chunks (S2 only attends within its chunk, so this parallelizes well — hence `--stage2_batch_size 4`).
5. **Codec decode** the full 8-codebook stream to two 16 kHz waveforms (vocal + accompaniment).
6. **Mix** to a single 16 kHz waveform (a simple weighted sum; the README recommends 0.7 × vocal + 1.0 × accompaniment for the typical case).
7. **Upsample** to 44.1 kHz via `YuE-upsampler`.

Notes:

* The third-party **YuE-extend** project ([Mozer/YuE-extend](https://github.com/Mozer/YuE-extend)) explicitly implements "music continuation" — feeding a generated song's tail back as ICL prompt to extend beyond the natural 5-minute ceiling.
* The official paper claims generation up to **five minutes** of coherent music; community reports of 6–7 minute generations exist but with structural drift.
* No sliding-window attention is used inside the transformer — long context is handled by **RoPE scaling at inference** plus the explicit sectional re-prompting above.

### 8. Inference Pipeline Pseudocode

The full lyrics-to-song pipeline as it should be implemented in C#:

```csharp
// HartsyInference.Audio.YuE — pseudocode (pure C#)

public sealed class YuEPipeline : IDisposable
{
    private readonly LlamaModel _stage1;          // HartsyInference.Models.LlamaModel (dotLLM-style)
    private readonly LlamaModel _stage2;
    private readonly XCodecDecoder _codec;        // 8-codebook → 16 kHz waveform
    private readonly XCodecEncoder? _codecEnc;    // only if ICL with audio prompt
    private readonly VocosUpsampler? _upsampler;  // optional 16 kHz → 44.1 kHz
    private readonly LlamaTokenizer _textTokenizer;

    public byte[] Generate(YuEOptions opts)
    {
        // 1. Parse lyrics into sections
        var sections = ParseSections(opts.LyricsText);          // List<(string tag, string body)>
        var genreTag = opts.GenreText.Trim();

        // 2. (Optional ICL) tokenize audio prompt with X-Codec encoder
        AudioTokens? audioPrompt = null;
        if (opts.AudioPromptPath is not null)
        {
            using var wav = AudioLoader.LoadResampled(opts.AudioPromptPath, 16_000);
            var slice = wav.Slice(opts.PromptStartSec, opts.PromptEndSec);
            audioPrompt = _codecEnc!.Encode(slice);             // (8, T) int codes
        }

        // 3. Stage-1: generate codebook-0 tokens, section-by-section
        var allCb0 = new List<int>();                            // interleaved [v0,a0,v1,a1,...]
        using var s1Cache = _stage1.AllocateKvCache(maxLen: 16_384);

        var systemPrompt = BuildSystemPrompt(opts.Mode, genreTag);
        var systemIds = _textTokenizer.Encode(systemPrompt);
        _stage1.PrefillContext(systemIds, s1Cache);

        if (audioPrompt is not null)
        {
            // ICL: inject the demo audio tokens as cb0-only interleave
            var demoCb0 = audioPrompt.Codebook0AsInterleaved();
            _stage1.PrefillContext(demoCb0.Select(c => c + Constants.AudioTokenOffset), s1Cache);
        }

        for (int s = 0; s < Math.Min(sections.Count, opts.RunNSegments); s++)
        {
            // Append next section's lyric tokens
            var sectionText = $"\n\n[{sections[s].tag}]\n{sections[s].body}\n[start_of_segment]";
            var sectionIds = _textTokenizer.Encode(sectionText);
            _stage1.PrefillContext(sectionIds, s1Cache);

            // Sample up to max_new_tokens cb0 audio tokens for this section
            var generated = _stage1.SampleUntil(
                stopToken: Constants.EndOfSegment,
                maxNew: opts.MaxNewTokens,                       // default 3000
                temperature: opts.Temperature,                   // default from generation_config.json
                topP: opts.TopP,
                topK: opts.TopK,
                repetitionPenalty: opts.RepetitionPenalty,       // default 1.1
                cfgScale: opts.CfgScale,                          // optional, ICL only
                cache: s1Cache);

            // Strip control tokens, decode audio-ID offsets back to 0..1023
            allCb0.AddRange(generated
                .Where(t => t >= Constants.AudioTokenOffset && t < Constants.AudioTokenOffset + 1024)
                .Select(t => t - Constants.AudioTokenOffset));

            // Context-window management: if cache nearing limit, evict oldest audio tokens
            if (s1Cache.UsedLen > 14_000)
                s1Cache.WindowAudioTokens(keepLastSeconds: 10);
        }

        // 4. Stage-2: predict residual codebooks 1..7 in ~30 s chunks, batched
        //    cb0 stream length T  ⇒  about T / (50*2) seconds per track
        const int framesPer30s = 30 * 50;                        // 1500 frames
        const int tokensPerChunk = framesPer30s * 2;             // 3000 cb0 tokens
        var fullStream = new int[allCb0.Count / 2, 8];           // 8 codebooks × frames

        // Lay cb0 directly into final array (it's already what S1 produced)
        WriteCb0Frames(fullStream, allCb0);

        // S2 chunked decoding
        for (int chunkStart = 0; chunkStart < allCb0.Count; chunkStart += tokensPerChunk)
        {
            var cb0Chunk = allCb0.GetRange(chunkStart, Math.Min(tokensPerChunk, allCb0.Count - chunkStart));
            var residual = _stage2.GenerateResiduals(
                cb0Chunk,
                batchSize: opts.Stage2BatchSize,                  // default 4
                repetitionPenalty: opts.RepetitionPenalty);
            WriteResidualFrames(fullStream, residual, frameOffset: chunkStart / 2);
        }

        // 5. Codec decode  → two 16 kHz mono waveforms (vocal, accompaniment)
        var vocalCodes  = SliceTrack(fullStream, track: 0);      // (8, T)
        var accompCodes = SliceTrack(fullStream, track: 1);
        var vocalWav    = _codec.Decode(vocalCodes);             // float[] @ 16kHz
        var accompWav   = _codec.Decode(accompCodes);

        // 6. Mix
        var mixed = Mix(vocalWav, accompWav, vocalGain: 0.7f, accompGain: 1.0f);

        // 7. (Optional) Upsample 16 → 44.1 kHz
        var finalWav = _upsampler is not null
            ? _upsampler.Upsample(mixed)                          // float[] @ 44.1kHz
            : mixed;

        return WavWriter.ToBytes(finalWav, sampleRate: _upsampler is not null ? 44_100 : 16_000);
    }
}
```

The hot loops are (a) S1's per-token sample, which dominates wall-clock (it generates 100–6000 tokens depending on song length), and (b) S2's per-token sample, also non-trivial but batchable. Codec decode and upsampling are cheap by comparison.

### 9. VRAM and Performance

**Memory** (FP16 weights + KV cache, both stages co-resident):

| Component | bf16 weight memory | Notes |
|---|---|---|
| S1 (~7B) | ~14 GB | LLaMA-2 7B in bf16 |
| S2 (~1.5B) | ~3 GB | LLaMA decoder |
| X-Codec encoder + decoder | ~0.6 GB | |
| YuE-upsampler (Vocos) | ~0.3 GB | Optional |
| S1 KV cache (16K context, MHA 32×32×128) | ~2 GB | Grows linearly with sequence length |
| S2 KV cache (per chunk, ~1500 tok) | ~0.3 GB | Small because of chunking |
| **Total (bf16, full pipeline)** | **~20 GB** | Fits an A100-40G, H100, or 3090/4090 with headroom (just barely on 24 GB) |

**Quantized variants** (community, not official):

* **8-bit (GGUF Q8_0 / EXL2 8.0bpw)** — `tensorblock/YuE-s1-7B-anneal-en-cot-GGUF`, `sgsdxzy/YuE-exllamav2` — drops S1 to ~7 GB, total pipeline ~12 GB. Quality essentially indistinguishable.
* **4-bit (GGUF Q4_K_M / EXL2 4.0bpw)** — S1 ≈ 4 GB, full pipeline ~8 GB; fits a 12 GB card. Some loss of vocal articulation reported.
* **YuEGP (deepbeepmeep/YuEGP)** — CPU+GPU offloading flavor; runs on **6 GB** GPUs at the cost of speed.

**Speed** (observed, single-stream, official reference Python pipeline):

| GPU | Audio generated | Wall-clock | Real-time factor |
|---|---|---|---|
| H800 (80 GB) | 30 s | ~150 s | ~5× slower than realtime |
| A100 80 GB | 30 s | ~180 s | ~6× slower |
| RTX 4090 | 30 s | ~360 s | ~12× slower |
| RTX 4090 (YuEGP-optimized) | 60 s | ~240 s | ~4× slower |
| RTX 4090 (ExLlamaV2 8-bit) | 30 s | ~120 s | ~4× slower |
| L40S (community) | 180 s (3-min track) | ~300 s | ~1.7× slower |

A full **5-minute song** with the reference pipeline therefore takes ~25–35 min on a 4090, ~12–15 min on an A100, and ~2–3 min on an H100 cluster with 8-bit quant and batching. **There is no realtime path for a 7B autoregressive song model on a single consumer GPU** — this is the fundamental cost of the autoregressive design.

### 10. Comparison to ACE-Step

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

### 11. C# Implementation Notes (HartsyInference)

This section is the implementer's bridge.

**Reuse the native `HartsyInference.LLM` patterns aggressively.** Both S1 and S2 are stock LLaMA-2 decoder architectures with vocab/RoPE-scaling tweaks. The native `HartsyInference.LLM` package already implements LLaMA (its design drew on the historical [DOTLLM_ARCHITECTURE.md](DOTLLM_ARCHITECTURE.md) study). The right plan is:

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
