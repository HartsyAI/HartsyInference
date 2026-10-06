# New-model runtime plan (Breeze TTS 2, Fish Audio S2, ControlFoley, Kolibri-1, Clef)

Merged so far: contracts only (#253–#257, #261, #262). None can load weights or generate; Swarm returns `no_provider`.

## Fix-later notes
- [ ] **Kolibri-1 load blocker (from #262 review, merged as is):** `GgufConfigFactory.FromGguf` requires
  `kolibri1.feed_forward_length` before the Kolibri branch runs, but published Kolibri GGUFs omit it. Make it optional
  for `kolibri1`, falling back to `expert_feed_forward_length` / `expert_shared_feed_forward_length`; add a test.
- [ ] #262 shipped without its changelog/version entry; backfill with the next release bump.
- [ ] `MimiConfig.FrameRate` (int) still reports 25; true Mimi rate is `FrameRateHz` (12.5). Fix the pinned codec test.

## Per-model work (in order)
1. **Kolibri-1 (LLM):** fix above, add LLM catalog entry + download metadata, load a real GGUF, parity vs llama.cpp
   (routing, sandwich norm, local/global + NoPE), then LLMAssistant registration.
2. **Fish Audio S2 Pro:** resolve open questions in `docs/Research/FISH_AUDIO_S2_ARCHITECTURE.md` (key layout, tokenizer
   ids, prompt format); slow AR (reuse GenericTransformer where parity allows) + 4-layer fast AR, ModifiedDAC decoder,
   pipeline, `TtsCatalog` entry. Fish-Speech 1.5 already works live and can seed shared pieces.
3. **Breeze TTS 2:** T5Gemma2 encoder, Qwen3 backbone, depth decoder, reuse shared Mimi decoder; catalog entry.
4. **ControlFoley:** video-to-audio pipeline (text/video/reference-audio conditioning); license is non-commercial, surface it.
5. **Clef:** typed-decision inference over a 64-layer multimodal backbone; image/video input; needs its own service surface.

## Per-model definition of done
Real-weight CPU parity tensors → generation test with independent oracle (e.g. Whisper for TTS) → catalog + download
metadata → AudioLab/LLMAssistant provider and params → alpha bump + changelog → NuGet → pin extensions → restart
swarm.hartsy.ai → real live generation. Do not register providers before the engine path loads real weights.

## Pending ops
Swarm currently runs alpha.251 packages; every item above needs a new engine publish before extension PRs.
