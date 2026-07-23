# SwarmUI production verification of the step-cache round — engine 2.0.0-alpha.64, RTX 4090

**What:** The full H1.5 step-cache round (per-model calibrated profiles, `HARTSY_STEP_CACHE=1`)
verified end-to-end through the PRODUCTION path: SwarmUI + the HartsyInference backend extension,
driven purely via `/API/GenerateText2Image`. Engine packed as `2.0.0-alpha.64` to the local feed,
extension pin bumped from stale `1.0.0-alpha.52`.

## Results (1024², seed 42, warm = second gen same process; walls = engine `complete in` logs)

| Model | Baseline warm | Cached warm | Speedup | Profile fired (logged) | Eyeball |
|---|---|---|---|---|---|
| Krea2-Turbo (8 st) | 4.43 s | **3.96 s** | 1.12× | `0.15, late=0.5` → 1 reuse | ✓ |
| Z-Image-Turbo (8 st) | 2.74 s | — (no profile; =1 → raw 0.10) | — | n/a | ✓ |
| Ideogram 4 (20 st) | 19.0 s | **13.7 s** | **1.39×** | `0.3, late=0.5` → 6 reuses/stream | ✓ |
| Flux.2 Dev (50 st) | 97.9 s | **39.1 s** | **2.51×** | `0.25, poly` → 31 reuses | ✓ identical detail |

- **Flagship regression bars: PASS on alpha.64** — Krea2-Turbo warm 4.43 s (<6.5 s), Z-Image-Turbo
  warm 2.74 s (≤3.2 s), both uncached defaults.
- Swarm speedups match the standalone A/B measurements (1.13× / 1.39× / 2.49×) — the calibrated
  profiles behave identically through the production path. All images on-prompt; cached Flux.2 is
  visually indistinguishable from its baseline.
- Cache remains DEFAULT-OFF: SwarmUI restored to a no-knob environment after the pass. Arm with
  `HARTSY_STEP_CACHE=1` in the SwarmUI process env to get every model's calibrated gate.

## Deployment/infra fixes made during the pass (in the SWARM tree / extension, not this repo)

1. Extension `HartsyInferenceBackend.cs:389`: `request.Fps` → `request.Fps ?? 25`
   (`VideoRequest.Fps` became nullable since the extension's last pin). **Needs committing in the
   extension repo.**
2. Extension + AudioLab pins bumped `1.0.0-alpha.52` → `2.0.0-alpha.64` (AudioLab was already
   broken-if-rebuilt: its source uses engine types newer than its pin). **Extension-repo commits.**
3. Swarm-tree staging: repaired DANGLING `Krea2/krea2_turbo_fp8_scaled.safetensors` symlink (broke
   the flagship regression path); staged `Flux2/flux2-dev-Q4_K_S.gguf`, the Mistral TE, and the
   Flux2 VAE as symlinks (Flux.2's side-model resolver roots at Swarm's Models dir and had tried to
   re-download the 18 GB TE → disk-full).
4. `Users.ldb` corrupted by overlapping SwarmUI instances during redeploy (2nd occurrence of the
   dual-instance trap) — recovered per the established procedure; backup at
   `Users.ldb.corrupt-2026-07-23-dualinstance`.

## Engine bugs found by the pass (open, this repo)

- **Cross-model VRAM eviction gap (serving path):** switching models keeps the previous pipeline
  resident (KEEP_MODELS) and OOMs 24 GB (Krea2 13 GB + Z-Image build → 74 MB free; hard OOM poisons
  the session — cuDNN conv + full-res VAE disable session-wide). Even the extension's
  `HartsyInferenceClearCache` leaves ~5.4 GB residue, tripping Ideogram's ≥20 GB guard. Serving
  needs switch-pressure eviction (evict resident pipelines before building the next model when
  free VRAM < the incoming model's footprint).
- Flux.2 on 24 GB beside the resident Q4 DiT: full-res VAE decode OOMs → session-sticky tiled
  fallback (works, logged; acceptable but worth a headroom check).
