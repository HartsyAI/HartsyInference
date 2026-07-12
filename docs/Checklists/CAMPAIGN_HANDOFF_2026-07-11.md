# Campaign handoff — 2026-07-11 (fresh-context start point)

**Deployed:** `44.84-local` (Swarm backend #2 live on the 4090). **All engine/extension changes this
session are UNCOMMITTED in the shared tree** — the user commits manually (the Animate round-11 commit is
being made now). A fresh agent must NOT `git stash/checkout/reset`; treat the working tree as the source of
truth and keep scoped patch backups in the session scratchpad.

Read first (unchanged rule): `CLAUDE.md` → `docs/CODE_STYLE.md` → `docs/Agents/AGENTS.md` → this file →
`docs/Checklists/E2E_PARITY_WORKLOG.md` (top queue + rounds 9/10/11) and `docs/Checklists/VIDEO_GENPERF_PLAN.md`.

---

## What just finished (this session, all deployed on 44.84)

- **Image bring-up + perf pass — DONE** for the whole handoff batch: F-Lite 61.5s, HunyuanImage 74.1s,
  Zeta-Chroma confetti FIXED (BF16 host-read), Radiance 12.26min→54.4s (14×), HiDream 44.0s (was OOM),
  Flux.2 Dev first-e2e + GGUF-eviction NRE fixed fleet-wide, OmniGen2 benched.
- **VIDEO arc rounds 1–11 — DONE:** LTX-2.3 **451→39s** (VAE port 24×, Gemma cache, prefix persistence,
  audio half; F16 probed→deferred at stream floor). Wan **I2V 234→32s, T2V 30s, S2V 3.44 s/step +
  2.04-min warm** (CLIP-branch host-glue fix, umT5/CLIP/cond/audio/ref caches, KEEP_MODELS residency,
  multi-group WanDitOps port). **Step floor CLOSED with evidence** (r9 LTX CFG-interleave bit-exact,
  r10 Wan proven at fp8 compute floor via the sync-after-every-op discriminator). **Animate checkerboard
  FIXED** (r11 — fp8 `.scale_weight` BF16 companions dropped by an `==F32` fold guard). MatrixGame2
  testhost crash fixed (buffer-overflow class, verified 1/1).

**Standing lesson promoted this session:** the **BF16-read-as-float\* bug class hit 3× in a row**
(F-Lite register_tokens/lambda, Zeta dec_net in_ln, Animate fp8 scale companions). Any host `(float*)`
read OR dtype guard on a checkpoint tensor MUST handle BF16 and F16, not just F32. Grep audit target.

---

## ✅ ROUND 12 DONE (44.85-local, 2026-07-11) — Wan-Animate at family parity

Shipped + validated (see `E2E_PARITY_WORKLOG.md` round 12 + memory `video-arc-round12-animate-parity`):
- **`WanAnimateLoader` warm caches** — umT5 prompt cache + CLIP-ViT-H reference cache (host-materialized,
  `EnsureEncoderHeadroom` staging, `HostCopy`), the `WanS2VLoader` pattern. **Warm prep 18.09 s → 0.00 s.**
- **Per-step perf port** — `WanAnimateTransformer.AddInPlace` (face-adapter residual host glue) → device
  `GatedResidualLastDim`. Per-step stays ~18 s/step: **Wan is at its fp8 compute floor** (r10), so per-step
  won't move — the caches are the win. `AddPose` left on host (host-slicing convention, 1×/forward).
- Validated: cold 6.86 min coherent android dance (checkerboard-free, identity, motion); warm cache HITs +
  coherent; flagships Z-Image 2.85 / Krea2 4.57; Wan 1.3B T2V regression clean.
- **Deploy trap logged:** don't run the flagship gate before a 14B video validation in the same process —
  KEEP_MODELS residency + `EvictAll` NOT trimming the mempool → `BuildMotion` OOM. Validate the 14B FIRST.

**Conditioning cache — DONE (round 12b, 44.86-local).** Split `WanAnimatePipeline.GenerateAnimation` into
cached/uncached (`WanAnimateConditioning` host struct); the loader caches the pose/gray VAE latents + StyleGAN
motion features keyed on driving-video + reference bytes + geometry. A HIT skips the whole VAE + motion encode
AND the driving decode. Measured: pre-loop encode **81.8 s → 19.1 s** warm, gen **6.99 → 5.70 min**, prep
**18.09 → 0.00 s**; coherent, flagships clean, unit test + e2e pass. **Wan-Animate is now at full family parity.**

**Animate item left — IN-ENGINE DRIVING PREPROCESSING (feature, in progress).** Full plan +
build order in `docs/Design/WAN_ANIMATE_PREPROCESSING.md` (confirmed scope: full auto pose+face; YOLOv11-pose
now, DWPose later). The checkpoint expects a rendered pose **skeleton** + a **cropped face** clip; the loader fed
the raw clip (works but OOD for the pose branch). ComfyUI makes this a manual node chain — auto-in-backend is the
new feature.
- **Phase 0 DONE (compiles, not deployed):** params `Animate Auto-Preprocess Driving` (bool, default on) +
  `Animate Pose Video` / `Animate Face Video` overrides; `WanAnimateDrivingPreprocessor.BuildDrivingClips` (override
  + raw paths wired, auto falls back to raw + one-time warn); loader branches + cond-key now includes mode +
  override bytes. Behavior-preserving (default auto → raw fallback = old behavior) + adds the pre-rendered override path.
- **Phase 1 (next):** real `FaceDetector` = YOLOv8-face (reuse YOLO backbone + 1-class + 5-landmark head; weights
  as safetensors via a `.pt`→st convert + a new `SideModels.Entry`) + `WanAnimateFacePreprocessor` (detect→square
  crop→512²).
- **Phase 2:** `YoloV11PoseModel` (backbone + kpt head, 17 COCO kpts) + `SkeletonRenderer` (OpenPose palette) +
  `WanAnimatePosePreprocessor`. Validate skeleton vs a comfy DWPose render + same-seed A/B.
- YOLO loads from safetensors (`YoloPipeline.LoadV11`), so Phases 1/2 slot into the existing `Vision/Detection` +
  `Vision/FaceDetection` scaffolding. Backups: `scratchpad/round12-backups/` (Phase 0 files).

## ▶ START HERE next — image backlog (below) or the Animate pipeline-cache refactor above

---

## Backlog after that (pick per priority)

**Image leftovers (small, code mostly staged):**
- Flux.2 Dev `HARTSY_DIT_GRAPH=1` opt-in step-graph trial — implemented (round-3 code), never GPU-tested;
  confirm capture fits or falls back cleanly on Q4.
- Flux.2 Dev benign **1017 MB VAE-decode OOM-retry** each gen → candidate pre-decode `TrimMemoryPool`
  (HiDream-style).
- **Zeta-Chroma** 13 GB pixel-space residency → add to the model-switch **sync-probe eviction matrix**
  (it OOM'd Krea2 VAE when 3 models piled in one process).
- Dev-GGUF **switch-back OOM** (VRAM reclaim in PreloadWeights — distinct from the fixed NRE).
- **HunyuanImage ByT5 glyph branch** unwired (cache hook noted; wire when the branch lands).

**Video leftovers (deferred with numbers, revisit only if targeted):**
- LTX-2.3 `HARTSY_DIT_F16` activations to grow the resident prefix — ≤~15%, numerically risky, out of
  scope unless a bigger geometry needs it. Probe numbers: video stream 15–18k absmax blocks 37–45.
- Wan is at its fp8 compute floor — do NOT re-attempt graph/batched-CFG (r9/r10 closed both with evidence).

---

## Non-negotiable process rules (unchanged, condensed)

- **Shared 4090:** chain an idle gate before EVERY GPU action:
  `until [ $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1) -lt 25 ]; do sleep 20; done && <work>`.
  Pin `exactbackendid=2`. NEVER `pkill -f SwarmUI` from a shell whose command cd'd into a path containing
  "SwarmUI" (kills your own shell) — API shutdown only.
- **Deploy:** `dotnet pack HartsyInference.sln -c Release -p:VersionSuffix=alpha.44.<N>-local -o ~/.local/share/hartsy-local-nuget --nologo -v quiet`
  (verify ALL 15 nupkgs + 0 warnings — TreatWarningsAsErrors is on) → bump the pin in
  `"<swarm>/src/Extensions/SwarmUI-HartsyInference-Backend/SwarmUI-HartsyInference.csproj"` →
  `rm -rf "<swarm>/src/bin/extensions/SwarmExtensionSwarmUI-HartsyInference-Backend"` + its obj →
  restart Swarm idle-gated with `LD_LIBRARY_PATH=$HOME/.local/lib/cuda13` → verify the `[Cuda] perf flags:`
  log line. Swarm API is `http://192.168.10.188:7801` (NOT localhost).
- **After every deploy — flagship gate:** Z-Image-Turbo ≤3.2s warm AND Krea2-Turbo <6.5s warm, both
  VISUALLY verified (Read the PNG), plus a clean video→Z-Image eviction check.
- **Correctness method:** discriminator FIRST (ComfyUI same-seed = engine-bug vs checkpoint-character),
  THEN env-gated stage-dump oracle vs a torch reference on identical inputs; follow relL2 to the first
  divergence — don't desk-guess. The "reproduce the bug in the oracle with BF16-as-F32 weights" trick is
  the smoking gun for the BF16 class.
- Multi-agent pattern that worked: code-only agents (compile-verified, scoped patches, no GPU) batched
  into ONE pack + deploy + validation matrix owned by a single GPU agent at a time. Save a memory per
  round + a one-line `MEMORY.md` index entry.
