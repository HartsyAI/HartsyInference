# Image (T2I / edit) scoreboard

Canonical, single-source scoreboard for HartsyInference's image-generation models vs their reference
baseline. This replaces three previously-diverging copies of the same table (`README.md`,
`docs/PERFORMANCE.md` §5, `docs/Checklists/MODEL_STATUS_IMAGE.md`) and retires the narrower one-off
`benchmarks/results/*.md` write-ups those numbers were originally transcribed from — this file is now the
only place the headline figures live; the Source column names the run/date each number came from for
provenance, not a live link. Full methodology (hardware baseline, the standard performance profile
and its kill-switches, how to reproduce a run) lives in **[`../../docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md)**.

**GPUs.** Most rows are **RTX 4090 24GB**. A handful of models also have a genuine **RTX 3060 12GB**
row — either because they fit resident on 12GB, or (as of 2026-07-27) because the engine's new low-VRAM
block-streaming path lets them run there at all. Rows are labeled per-row; do not assume a GPU from a
neighboring row.

**Baseline.** ComfyUI via the SwarmUI API — the identical request routed to the ComfyUI backend, then the
HartsyInference backend, on the same GPU, same seed protocol (1 cold + 3 warm, warm median reported).
**F-Lite** has no ComfyUI graph for its architecture, so its baseline is a standalone Python **diffusers**
reference run instead (`f_lite` package, sequential CPU offload — see the Notes below, this is a
memory-bound comparison, not a straight compute one). **ChromaRadiance**'s baseline is ComfyUI's own
fp8 path (`--fp8_e4m3fn-unet --fast fp8_matrix_mult`), for an apples-to-apples fp8-vs-fp8 read.

**Ratio** = HartsyInference ÷ baseline. **<1.0× means Hartsy is faster** (the faster time is bolded in its
column) — this matches the convention already used in `docs/PERFORMANCE.md` and
`image_python_baselines_2026-07-18.md`. A ratio is only computed when both sides of a row were measured
together in the same run; where that wasn't possible (e.g. a Debug-build Hartsy number against a
Release-build ComfyUI number) the cell is left as `—` and the caveat is called out.

**Freshness.** Where a model has more than one measurement, the table uses the most recent one that
reflects the **shipped default configuration**. Two optimizations exist in the engine today
(**per-model step-cache**, `HARTSY_STEP_CACHE=1`, and **limited-interval CFG**, `HARTSY_CFG_INTERVAL`)
that are **default-OFF** — real, measured wins, but opt-in, not part of the number below. They're noted
per-model as footnotes, not used to replace the baseline row.

## Scoreboard

| Model | GPU | HartsyInference | ComfyUI / Python | Ratio | Date | Source |
|---|---|---:|---:|---:|---|---|
| Flux-Schnell (4 st) | 4090 | **2.4 s** | 3.8 s | **0.63×** | 2026-07-26 | 2026-07-26 sweep |
| Flux.2 Klein 4B (4 st) | 4090 | **2.36 s** | 1.85 s | 1.28× | 2026-07-11 | [PERFORMANCE.md §5](../../docs/PERFORMANCE.md) |
| Krea2-Turbo (8 st) | 4090 | **4.5 s** | 6.5 s | **0.69×** | 2026-07-26 | 2026-07-26 sweep |
| Z-Image-Turbo (8 st) | 4090 | 4.2 s | **3.1 s** | 1.35× | 2026-07-26 | 2026-07-26 sweep¹ |
| Boogu-Turbo (4 st) | 4090 | 3.26 s | **2.54 s** | 1.28× | 2026-07-11 | [PERFORMANCE.md §5](../../docs/PERFORMANCE.md) |
| AuraFlow-0.3 (20 st) | 4090 | **13.93 s** | 14.0 s | ~1.0× (tied) | 2026-07-11 | [PERFORMANCE.md §5](../../docs/PERFORMANCE.md) |
| ERNIE-Image (20 st) | 4090 | **20.0 s** | 23.9 s | **0.84×** | 2026-07-11 | [PERFORMANCE.md §5](../../docs/PERFORMANCE.md) |
| Lance-3B (30 st) | 4090 | 9.6 s | unsupported (no ComfyUI arch) | — | 2026-07-26 | 2026-07-26 sweep |
| Chroma1-Radiance (fp8, 20 st) | 4090 | **21.46 s** | 21.07 s (fp8) | 1.02× | 2026-07-18 | python baselines |
| Krea2-Base (28 st) | 4090 | **30.3 s** | 41.5 s | **0.73×** | 2026-07-26 | 2026-07-26 sweep |
| Flux2-Dev 32B (Q4_K_S GGUF, 20 st) | 4090 | **39.6 s** | 54.0 s | **0.73×** | 2026-07-26 | 2026-07-26 sweep |
| Qwen-Image (20 st) | 4090 | **40.6 s** | 58.2 s | **0.70×** | 2026-07-26 | 2026-07-26 sweep |
| HiDream-i1 17B (25 st, cfg 5) | 4090 | 44.0 s | **35.2 s** | 1.25× | 2026-07-11 | [PERFORMANCE.md §5](../../docs/PERFORMANCE.md) |
| Chroma1-HD (30 st, cfg 7)² | 4090 | **24.6 s** | 32.6 s | **0.75×** | 2026-07-26 | 2026-07-26 sweep |
| Boogu-Base (20 st) | 4090 | 26.5 s | **17.8 s** | 1.49× | 2026-07-11 | [PERFORMANCE.md §5](../../docs/PERFORMANCE.md) |
| Ideogram4 (20 st) | 4090 | 19.5 s | **18.0 s** | 1.08× | 2026-07-26 | 2026-07-26 sweep |
| OmniGen2 (20 st) | 4090 | 15.1 s | **13.0 s** | 1.16× | 2026-07-26 | 2026-07-26 sweep |
| Flux-Dev (20 st) | 4090 | **9.5 s** | 12.5 s | **0.76×** | 2026-07-26 | 2026-07-26 sweep |
| SDXL (20 st)³ | 4090 | 10.9 s | **7.5 s** | 1.45× | 2026-07-26 | 2026-07-26 sweep |
| HunyuanImage 2.1 17B (Q4_K_M GGUF, 2048², 20 st) | 4090 | 48.3 s | **47.1 s** | 1.03× (matched) | 2026-07-26 | 2026-07-26 sweep |
| Qwen-Image-Edit 2511 (20 st + ref, edit) | 4090 | 93 s | **87.8 s** | 1.06× | 2026-07-11 | [PERFORMANCE.md §5](../../docs/PERFORMANCE.md) |
| Lumina-Image 2.0 (25 st, cfg 4)⁴ | 4090 | 17.7 s | **10.05 s** | 1.76× | 2026-07-18 | python baselines |
| F-Lite 10B (30 st, cfg 6)⁵ | 4090 | **61.5 s** | 122.98 s (Python/diffusers) | **0.50×** | 2026-07-18 | python baselines |
| Anima (Cosmos-Predict2 2B, 20 st)⁶ | 4090 | 160.0 s | **2.4 s** | 66.7× | 2026-07-26 | 2026-07-26 sweep |
| Anima (Cosmos-Predict2 2B, 20 st)⁶ | 3060 | 168.1 s | **8.3 s** | 20.3× | 2026-07-26 | 2026-07-26 sweep |
| Ideogram4 (20 st, low-VRAM stream)⁷ | 3060 | 205.2 s (Debug build) | 156.0 s (Release) | — (not same build) | 2026-07-27 | lowvram fix §5f / 2026-07-26 sweep |
| Qwen-Image (20 st, low-VRAM stream)⁷ | 3060 | 231.4 s (Debug build) | 242.9 s (Release) | — (not same build) | 2026-07-27 | lowvram fix §5g / 2026-07-26 sweep |

¹ Same-day production-path evidence conflicts with this cell — see **Notes**.
² Uses the documented compositional-prompt + negative-prompt + cfg 7.0 / 30-step recipe for this model
family, not the standard harness params (20 st / cfg 4.0) used for the older Chroma1-HD number — see Notes.
³ ComfyUI's own number also moved (3.7 s → 7.5 s) — see Notes; this is not read as a Hartsy regression.
⁴ Output-correctness caveat as of 2026-07-27 — see Notes.
⁵ Memory-bound comparison, not compute-bound — see Notes.
⁶ Anima's Hartsy-side numbers here predate the 2026-07-27 host→device port (29× per-step); see Notes.
⁷ Debug build; resolution/step-matched to the family default but not built the same way as the ComfyUI
Release number beside it — no ratio is reported. Retires the `OOM²` cell from the 2026-07-26 sweep.

## Notes

- **Z-Image-Turbo regression flag.** The 2026-07-26 sweep's headline Hartsy number (4.2 s) is used above
  as the freshest head-to-head, but it's worth flagging: three days earlier, the same model measured
  **2.74 s** through the production SwarmUI path against a documented flagship regression bar of **≤3.2 s**
  (`2026-07-23_swarm_stepcache_verification.md`).
  ComfyUI's own number is identical across both dates (3.1 s), so this isn't a moving baseline — the 07-26
  number silently fails a bar that was passing days before. Worth a re-run before trusting 4.2 s as durable.
- **SDXL.** The 2026-07-26 sweep downloaded SDXL fresh (`stabilityai/stable-diffusion-xl-base-1.0`,
  6.94 GB, byte-exact) rather than reusing whatever checkpoint produced the older **2.93 s / 3.7 s**
  ("faster than ComfyUI") figure from `docs/PERFORMANCE.md`'s 07-11 snapshot. ComfyUI's own time moved too
  (3.7 s → 7.5 s) on the same swap, which is the tell that this is a different checkpoint/config, not an
  engine-side regression. `2026-07-27_lowvram_leak_fix.md` §3
  separately shows SDXL's peak VRAM is dominated by an elastic full-resolution VAE decode workspace, not
  streamable weight — so this model needs no streaming conversion, just a from-scratch re-bench.
- **F-Lite.** No ComfyUI architecture exists for this model; the baseline is diffusers with **sequential
  CPU offload** (10B DiT + T5-XXL ≈ 29 GB > the 4090's 24 GB, so resident and accelerate-offload both
  OOM). Read the 0.50× as "beats diffusers-on-a-24GB-card," not a compute-parity claim — on a ≥40 GB GPU
  diffusers would likely run resident and be faster.
- **Lumina-Image 2.0 — correctness caveat, not just perf.** The 17.7 s / 10.05 s pair above is the last
  clean timing measurement, but `2026-07-27_lowvram_leak_fix.md`
  §5k found the currently-deployed checkpoint (after fixing an unrelated wrong-file bug) now produces
  **coherent but completely off-prompt output** — the text conditioning is wrong, and the bug was
  previously unreachable because the model couldn't load at all. Treat this row's timing as informative but
  the model's current *output* as unverified pending a fix.
- **Anima — perf numbers are stale as of the 07-26 sweep.** The 160.0 s/168.1 s figures above reflect the
  pre-fix engine. `2026-07-27_lowvram_leak_fix.md` §5c ported
  `AnimaBlock`'s host-glue to device ops (792 → 3 D2H syncs/step, **29× per-step**) but only measured the
  ratio on a Debug build on the 3060 (50.2 s at 1024²/20st) — explicitly flagged as not comparable to the
  Release-build numbers in this table. A fresh Release-build 4090/3060 bench is needed before this row can
  be updated; until then the 66.7×/20.3× gap is known-stale, not current.
- **Lens (Turbo / RL-tuned) excluded from the table.** Both variants produced solid-black output via the
  SwarmUI API in the 2026-07-26 sweep (16/16 failures) — a fast broken kernel is not a result, per
  `docs/PERFORMANCE.md`'s own methodology. Root-caused and fixed 2026-07-27 (SageAttention's INT8 path
  casts V to F16 and Lens's un-normalized V overflows F16 range by block 45); see
  `2026-07-27_lowvram_leak_fix.md` §5d. Output is verified
  correct post-fix but **no timing was re-captured**, so there's no number to put in this table yet.
- **Kandinsky 5.0 Lite** is built and end-to-end verified (`docs/Checklists/MODEL_STATUS_IMAGE.md`) but has
  no ComfyUI or Python baseline in any source file and was absent from all three superseded scoreboards —
  omitted here for the same reason, not dropped by mistake.
- **RTX 3060 capacity, general.** As of 2026-07-26, only Anima ran on the Hartsy@3060 lane; every other
  model returned `OOM²`. `2026-07-27_lowvram_leak_fix.md`
  found that sweep ran 13 models in one process while a since-fixed GPU-memory leak-on-OOM bug was active,
  so only the *first* OOM in that sequence was necessarily genuine — most of the `OOM²` cells are unverified,
  not confirmed capacity failures. New low-VRAM block streaming (`HARTSY_LOWVRAM`, default `auto`) has since
  unlocked HunyuanImage-2.1, Ideogram 4, and Qwen-Image on the 3060 (see the table and footnote 7); OmniGen2
  was shown not to need streaming at all (10.0 GB peak on a fresh process, vs. a 17.0 GB 4090 peak that
  is mostly elastic, not weight). HunyuanImage-2.1's 3060 number (19.7 s) is omitted from the table because
  it was measured at **1024²** with `HARTSY_LOWVRAM=on` forced — the 4090 row above is 2048² at the
  `auto` default, so the two aren't comparable on either axis. Krea2 (Turbo and Base) remain a confirmed
  genuine 3060 capacity failure, independently reconfirmed in a fresh process.
- **Step-cache (`HARTSY_STEP_CACHE=1`) — default OFF, queued/opt-in.** Per-model calibrated First-Block-Cache,
  SSIM ≥ 0.95-gated. Standalone + SwarmUI-production-verified additive speedups on top of the baselines
  above: Flux.2 Dev 50 st **2.49×**, Ideogram 4 **1.39×**, Qwen-Image **1.20×**, Krea2-Turbo **1.13×**,
  Z-Image-Turbo **no calibrated profile ships** (drift floor too high at its 8-step schedule; resolves to a
  no-op). See `2026-07-23_swarm_stepcache_verification.md`
  and the per-model `2026-07-22_accel_stepcache_*_4090.md` files.
- **Limited-interval CFG (`HARTSY_CFG_INTERVAL`) — default OFF, negative result for Qwen-Image.** The
  paper's early-step skip bands measurably save wall time (−15%) but flip output style (SSIM 0.35, photo→
  illustration) — rejected. A late-only band (`0.15,1`) is quality-safe but only saves 3–5% because this
  scheduler has few late-normalized-t steps. See
  `2026-07-22_accel_cfginterval_qwen_4090.md`.
