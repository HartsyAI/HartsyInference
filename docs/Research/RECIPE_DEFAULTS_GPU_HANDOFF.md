# Recipe defaults GPU handoff

State: the recipe-contract work and the SwarmUI-aligned defaults are all merged on `main` (HartsyInference #508–#522,
SwarmUI-HartsyInference-Backend #38). Everything was verified on the CPU lane only: release build clean, unit tests green
apart from `EngineReleaseDeviceGateTests.FreeMemory_WaitsForAnAudioJob…`, which also fails on `main` without a GPU.
**No generation has run on real weights since these changes. Do not claim parity or quality.** The contract and the
per-family ledger, with every value's SwarmUI source, are in [RECIPE_CONTRACT.md](RECIPE_CONTRACT.md).

Commits:

| Ref | What it is |
|---|---|
| `cc4d184` | Before the contract work (#504). Base for the refactor identity gate. |
| `2dcb81a` | #516: refactors done, no default changed yet. Base for every pass below. |
| `0ca5fb0` | #520: `alpha.386`, published on NuGet. |
| `1f62fa5` | `main` today: adds #521 (`er_sde`) and #522 (Wan 2.2 I2V-14B), which are not on NuGet yet. |

`--head` takes a checkout path, not a ref, so make a worktree for any head that is not your working tree.

## 1. Refactor identity gate (skip if already run)

#508, #510, #511, #513, #515 and #516 moved defaults out of the pipelines into recipes and claimed bit-identical
output. Nobody has run that claim on a GPU.

```bash
git worktree add ../hi-2dcb81a 2dcb81a
tests/regression-ab.sh --base cc4d184 --head ../hi-2dcb81a --tag "" --expect identical \
  --filter sd15,krea2,zimage,qwenimage-q4k,flux1-dev,flux2-dev-q4ks,chroma-hd,omnigen2,anima,hunyuanimage-q4,lumina2,sd35m,auraflow,wan5b,wan21-14b,ltx25-dev,ltx25-distilled,h3-fl2va-fp8,hunyuanvideo,stableaudio,acestep-v1
```

Every digest must match. A mismatch is a real regression in that range: bisect it across the six PRs before touching
anything else.

## 2. Pinned cases across the default changes

Base `2dcb81a`, head `main`. The existing rows pin most values, so most must stay identical. The rows below leave a
changed value unset, so their output moves on purpose.

```bash
tests/regression-ab.sh --base 2dcb81a --tag "" --expect identical \
  --filter sd15,krea2,zimage,qwenimage-q4k,flux1-dev,hunyuanimage-q4,lumina2,sd35m,auraflow,hidream,ideogram4,boogu-turbo,lens-mxfp8,wan5b,wan21-14b,h3-fl2va-fp8,hunyuanvideo,stableaudio,acestep-v1
tests/regression-ab.sh --base 2dcb81a --tag "" --expect bounded:0 --speed-tolerance 150 \
  --filter chroma-hd,flux2-dev-q4ks,omnigen2,anima,ltx25-dev,ltx25-distilled
```

Expected in the second run:

| Case | Unset value that moved | Expect |
|---|---|---|
| `chroma-hd` | CFG 5.0 → 3.5 | changed |
| `flux2-dev-q4ks` | CFG 4.0 → 3.5 | changed, or identical if Flux.2 Dev ignores the field (record which) |
| `omnigen2` | CFG 4.0 → 5.0 | changed |
| `anima` | sampler Euler → `er_sde` | changed; run it twice and confirm the same seed gives the same digest |
| `ltx25-dev` | CFG 4.0 → 3.0 | changed |
| `ltx25-distilled` | CFG 1.0 → 3.0 | changed, about 2× step time from the negative branch; **look at it first** |

In the first run, `wan5b` and `wan21-14b` change only fps (25 → 22 and 25 → 24). The digest covers frames, so it must
still match.

## 3. Unset defaults, per family

This pass shows what a user gets with nothing set. Add rows tagged `defaults` to `tests/regression-cases.sh` on a branch,
with every changed value left off, and run with `--tag defaults --expect bounded:0 --speed-tolerance 1000`. Output differs
by design. The report gives SSIM, step time and VRAM; then look at each pair yourself.

Rows with checkpoints already on the rig:

```
chroma-defaults	defaults	Stable-Diffusion/Chroma/Chroma1-HD-fp8mixed-final.safetensors	image|a red apple on a wooden table|-m chroma
flux2-defaults	defaults	Stable-Diffusion/Flux2/flux2-dev-Q4_K_S.gguf	image|a red apple on a wooden table|-m flux2
omnigen2-defaults	defaults	Stable-Diffusion/omnigen-2-1_0-omnigen-2-base-fp16.safetensors	image|a red apple on a wooden table|-m omnigen2
hunyuanimage-defaults	defaults	Stable-Diffusion/HunyuanImage/hunyuanimage2.1-q4_0.gguf	image|a red fox sitting in a snowy forest at sunrise, detailed fur, soft golden light|-m hunyuan-image
lens-defaults	defaults	Stable-Diffusion/Lens/lens_turbo_mxfp8.safetensors	image|a red bicycle leaning on a brick wall, photograph|-m lens
anima-defaults	defaults	Stable-Diffusion/Anima/anima-preview3-base.safetensors	image|a bowl of ripe cherries on a wooden table, soft morning light|-m anima
wan5b-defaults	defaults	Stable-Diffusion/Wan/wan2.2_ti2v_5B_fp16.safetensors	video|a red fox trotting through a sunlit snowy forest, cinematic|-m wan-22-5b
wan21-14b-defaults	defaults	Stable-Diffusion/wan2-1-t2v-14b-scaled-1_0-wan-2-1-text2video-14b-base-fp8.safetensors	video|a red fox trotting through a sunlit snowy forest, cinematic|-m wan-21-14b --steps 20
ltx25-dev-defaults	defaults	Stable-Diffusion/ltx-2-5-22b-dev-1_0-lightricks-ltx-2-5-base-int8.safetensors	video|a red fox trotting through a sunlit snowy forest, cinematic|-m ltx-2.5
ltx25-distilled-defaults	defaults	Stable-Diffusion/ltx-2-5-22b-distilled-1_0-lightricks-ltx-2-5-distilled-base-int8.safetensors	video|a red fox trotting through a sunlit snowy forest, cinematic|-m ltx-2.5-distilled
hunyuanvideo-defaults	defaults	Stable-Diffusion/hunyuan-video-t2v-720p-1_0-hunyuan-video-base-bf16.safetensors	video|a red fox trotting through a sunlit snowy forest, cinematic|-m hunyuan-video
```

`wan21-14b-defaults` pins 20 steps only to keep 81 frames on a 14B model affordable. Its CFG (5 → 6), frames and fps
stay unset.

Families with no row yet. Find a checkpoint under the models root, add a row, and report any you cannot find:

| Family | What to look at |
|---|---|
| Flux.2 Klein 4B / 9B | 8 steps (was 4) |
| Flux 3 Image Base / distilled | Base: `er_sde`. Distilled: 16 steps (was 15); also try `--sampler er_sde`, which decides whether distilled should default to it too |
| Flux.1 Schnell | CFG 1 (was 3.5) |
| Boogu Base | 20 steps, CFG 4 (was 25 / 3.5) |
| Mage-Flow Base | 20 steps (was 30) |
| Z-Image Base | sigma shift 3 (was 6) |
| Wan VACE | CFG 6, 81 frames @ 24 |
| Wan 2.2 I2V-14B | Resolves as `i2v-22-14b` (check the log line), CFG 3.5 |
| LTX-Video 0.9 | 768×512 @ 24 fps (was 704×480 @ 25) |
| Kandinsky 5 Video Lite / Pro | CFG 1 (was 5), 640², 73 frames. CFG 1 on the regular, non-distilled models is the riskiest value here |
| ACE-Step 1.5 turbo / sft | 120 s. Turbo must equal the old output at an explicit duration |
| MiniMax Music 3 | 120 s, flow CFG unchanged |
| YuE2 | 300 s |

**Music caveat:** `hartsy music --duration` defaults to 10 in the CLI (`MusicCommand.cs`, `int Duration = 10`), so the
CLI still masks the new recipe durations. Until that is nullable, pass `--duration 120` or `--duration 300` yourself to
see them, and pass the same explicit duration on both arms. A bounded run fails on any audio change, so judge music by
ear.

**Decide from what you see.** If a value is clearly worse (most likely LTX-2.5 distilled at CFG 3 or Kandinsky 5
regular at CFG 1), open a PR that reverts that one value. Bump its `Revision`, update the `RECIPE_CONTRACT.md` ledger
row with the evidence, and say which run showed it. Do not change a value that merely looks different.

## 4. Live SwarmUI session

Use Backend `main` (`cb1260b`) with the engine from NuGet `2.0.0-alpha.386`. The csproj floats to the newest
prerelease. That package does not have #521 or #522, so leave those two out of this pass.

1. **Defaults route.** Pick a video model (Wan 2.2 5B, LTX-2.5). `POST /API/HartsyInferenceGetModelDefaults` with
   `model_name` returns `success: true` and a descriptor with `$type: "video"`. The Video Frames and Video FPS fields
   show those defaults as placeholder text and are never filled in.
2. **Music model.** Pick ACE-Step. The same route answers `success: false`. In the browser's network tab, check that it
   is asked once, not again on every parameter refresh.
3. **Audio duration.** Generate ACE-Step 1.5 with *Text2Audio Duration* toggled off: the clip is about 120 s. Toggle it
   on at 30: the clip is 30 s.
4. **Features route.** `HartsyInferenceGetModelFeatures` returns `variant` and `variant_source` and no `variant_name`.
   Images and video generate from the gen page with no new errors in the log.
5. **Image inputs.** `HartsyInferenceGetImageInputs` answers as before for an image model, a video model and a music
   model, with no exception in the log.

## 5. Report back

For each pass, record the command, the result, and the case ids that failed or changed. Put them in the PR you open,
or in `docs/Checklists/MODEL_STATUS_*.md` for a family whose status changes. Then tick the A/B boxes on #517, #518,
#519, #521 and #522 with a link to the evidence.

Untested and worth a look if something is off: Wan 2.2 TI2V-5B at 22 fps. Core uses 22; the docs say 24 native.
