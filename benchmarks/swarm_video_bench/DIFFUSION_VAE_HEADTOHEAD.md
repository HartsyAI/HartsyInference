# LTX-2.5 diffusion-VAE head-to-head — runbook

**Purpose.** Produce the *matched-quality* row of the Hartsy-vs-ComfyUI LTX-2.5 scoreboard, with **both**
engines decoding through the **diffusion** video VAE (`ltx-2.5-video-vae-bf16.safetensors`) instead of the
convolutional one.

**Status of the prior row (done 2026-08-13/14).** conv-vs-conv, 768x512x97f / 30 steps / cfg 3.0 / 24 fps:
Hartsy **47.40 s** vs ComfyUI **42.48 s** (1.12x). Both decoded
`LTX-2/ltx-2.5-video-vae-conv-bf16.safetensors`.

This document was written by reading source only — **nothing here was executed**. Every claim carries a file
path and line. Where a thing could not be proven statically it is labelled
**[UNPROVEN]** with a note on what was checked.

---

## 0. Pre-flight (do these before anything else)

### 0.1 Disk — this is the single most dangerous item

```
/dev/nvme0n1p3  916G  862G  6.8G 100% /
```

`/var/lib/systemd/coredump/` currently holds **11 GB in 24 files**, including five SwarmUI cores written
at 08:43 **on 2026-08-14**. `/proc/sys/kernel/core_pattern` is
`|/usr/lib/systemd/systemd-coredump …` with a size limit of `9223372036854775808` — i.e. **effectively
unlimited**. A crashing ComfyUI (python, tens of GB RSS with a 22B model resident) will write a multi-GB
core and take the root filesystem to zero. That has already happened once and it **truncated
`Data/Backends.fds`**.

Do first:

```bash
df -h /
sudo rm -f /var/lib/systemd/coredump/core.SwarmUI.*        # frees ~11 GB
df -h /                                                     # want >> 20 GB before starting
```

Then back up config to timestamped copies (never in-place):

```bash
cd "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Data"
TS=$(date +%Y%m%d-%H%M%S)
cp -a Backends.fds "Backends.fds.bak.$TS"
cp -a Settings.fds "Settings.fds.bak.$TS"
```

Delete each arm's output videos as you go — 6 gens x ~97 frames of h264 is small, but the ComfyUI temp/latent
spill is not.

### 0.2 Service and network facts

- SwarmUI runs as a **systemd user unit**, `swarmui.service`. Never launch it by hand — `~/bin/swarmui-run.sh`
  takes a `flock` on `Data/.swarmui.lock` and a second instance is what corrupted `Users.ldb` twice
  (`~/bin/SWARMUI-SERVICE.md`).
- It binds **192.168.10.188:7801 only** — `Data/Settings.fds:136-138` (`Host: 192.168.10.188`, `Port: 7801`).
  `http://localhost:7801` will **not** connect. `bench_ltx25.py:68` already uses the right base URL.
- `~/bin/swarmui-run.sh` exports `HARTSYINFERENCE_MODELS=/home/hartsy/Desktop/HartsyInference/Models`, and
  `Data/Settings.fds:6` sets `ModelRoot: /home/hartsy/Desktop/HartsyInference/Models`. The `Models/` tree
  under the SwarmUI checkout does **not** exist; all model paths below are under the HartsyInference repo.

### 0.3 GPU numbering — two schemes, inverted on this box

| Scheme | 4090 | 3060 |
|---|---|---|
| CUDA ordinal (`GPU_ID` in `Backends.fds`) — enumerates **fastest-first** | **0** | 1 |
| `nvidia-smi --query-gpu=index` | **1** | 0 |

Evidence:

- `nvidia-smi --query-gpu=index,name` → `0, NVIDIA GeForce RTX 3060` / `1, NVIDIA GeForce RTX 4090`.
- `Data/Backends.fds` — backend `0:` has `GPU_ID: 1` (line 33); backend `1:` has `GPU_ID: 0` (line 73).
- Runtime confirmation in `Data/Logs/2026-08/*.log`:
  - `[ComfyUI-1/STDERR] [INFO] Device: cuda:0 NVIDIA GeForce RTX 4090`
  - `[ComfyUI-0/STDERR] [INFO] Device: cuda:0 NVIDIA GeForce RTX 3060`

  Both print `cuda:0` because each backend gets its own `CUDA_VISIBLE_DEVICES` — **the GPU *name* is the
  discriminator, never the ordinal in that line.**

So **ComfyUI backend id=1 is the 4090**, and `bench_ltx25.py:70`'s `GPU_SMI_INDEX = 1  # nvidia-smi index 1 =
RTX 4090` is correct and consistent with the above.

Backend inventory (`Data/Backends.fds`):

| id | type | title | GPU |
|---|---|---|---|
| 0 | comfyui_selfstart | (untitled) | 3060 — `enabled: false` |
| **1** | comfyui_selfstart | ComfyUI Self-Starting | **4090** — `enabled: false` |
| **7** | hartsyinference | HartsyInference GPU0 (4090) | **4090** — `enabled: true` |
| 8 | hartsyinference | HartsyInference GPU1 (3060) | 3060 — `enabled: true` |

**Both ComfyUI backends are currently `enabled: false`.** Enabling id=1 is a required step (§4.1).

### 0.4 Never hand-edit `Backends.fds` while the service is running

SwarmUI holds backend config in memory and rewrites the file. Toggle backends through the **UI or the API**,
after the `cp` backup in §0.1. Editing the file under a live service loses the edit or the file.

---

## A. ComfyUI side — decoding with the diffusion VAE

### A.1 Where the conv VAE comes from

`src/Text2Image/CommonModels.cs:94`:

```csharp
Register(new("ltx2-5-video-vae", "LTX-2.5 Video VAE", "The video VAE for Lightricks LTX-2.5.",
  "https://huggingface.co/mcmonkey/swarm-vaes/resolve/main/ltx-2.5-video-vae-conv-bf16.safetensors",
  "685b06ee…", "VAE", "LTX-2/ltx-2.5-video-vae-conv-bf16.safetensors"));
```

consumed at `src/BuiltinExtensions/ComfyUIBackend/WorkflowGeneratorModelSupport.cs:1373`:

```csharp
helpers.DoVaeLoader(null, (T2IModelCompatClass)null, "ltx2-5-video-vae");
```

### A.2 It CAN be overridden per-generation. Use the `vae` param — no source edit, no file rename.

`DoVaeLoader` checks the user's VAE parameter **before** falling back to the known-models table —
`WorkflowGeneratorModelSupport.cs:526-536`:

```csharp
public void DoVaeLoader(string defaultVal, string compatClass, string knownName)
{
    string vaeFile = defaultVal;
    string nodeId = null;
    CommonModels.ModelInfo knownFile = knownName is null ? null : CommonModels.Known[knownName];
    if (!g.NoVAEOverride && g.UserInput.TryGet(T2IParamTypes.VAE, out T2IModel vaeModel))
    {
        vaeFile = vaeModel.Name;      // <-- user param wins
        nodeId = "11";
    }
    …
    if (string.IsNullOrWhiteSpace(vaeFile) && knownFile is not null && …)
    {
        vaeFile = knownFile.FileName; // <-- registry fallback only if unset
    }
```

**The ordering concern is resolved — the param does not reroute the LTX branch.** The architecture branch
at line 1368 is guarded by `if (LoadingVAE is null)`, and there is a *separate* step
(`WorkflowGeneratorSteps.cs:72`, priority **-14**) that also assigns `g.LoadingVAE` from the same param. If
that step ran first, `LoadingVAE` would be non-null and the LTX-2.5 branch would fall into its `else`
(`LoadClipAudio` + `AudioVaeLoad`) — a different and wrong path. It does **not** run first:

- `CreateModelLoader` runs the `Priority <= -100` steps at `WorkflowGeneratorModelSupport.cs:880`,
- then executes the architecture `if/else` chain inline (LTX-2.5 at line 1368),
- and only afterwards runs the `Priority > -100` steps at line **1563**.

Priority -14 is `> -100`, so it fires **after** the branch. Both sites then converge on the same file.

**No duplicate VAELoader node.** `CreateVAELoader` (`WorkflowGenerator.cs:720-726`) memoises on the
normalised path:

```csharp
string vaeFixed = vae.Replace('\\','/').Replace("/", ModelFolderFormat ?? …);
if (NodeHelpers.TryGetValue($"vaeloader-{vaeFixed}", out string helper)) { return [helper, 0]; }
```

`DoVaeLoader` passes `vaeModel.Name`; the -14 step passes `vae.ToString(g.ModelFolderFormat)`, which is just
`Name.Replace("/", folderFormat)` (`src/Text2Image/T2IModel.cs:565-568`). After `CreateVAELoader`'s own
normalisation both produce the identical `vaeFixed`, so the second call hits the cache and returns the same
node. One `VAELoader`, one copy in VRAM.

**Param id.** Param IDs are the display name lowercased and stripped to letters only
(`T2IParamTypes.cs:200` `ID => CleanTypeName(Name)`, `:235` `LowercaseLetters`), so `"VAE"` → **`vae`**.
It is `Toggleable` with `IgnoreIf: "None"` (`T2IParamTypes.cs:693-695`), so simply omitting it restores
today's behaviour exactly.

**Least invasive / most revertible: pass `"vae"` in the API payload.** Do **not** edit `CommonModels.cs`
(needs a SwarmUI rebuild + restart, and the entry carries a hash used for auto-download) and do **not**
rename files in `Models/VAE/LTX-2/` (breaks the conv row's reproducibility and the `.swarm.json` sidecar).
Revert = stop passing the param.

Value to pass: `LTX-2/ltx-2.5-video-vae-bf16.safetensors`. Model lookup accepts the name with or without the
extension (`src/Text2Image/T2IModelHandler.cs:242`). The file is present:

```
/home/hartsy/Desktop/HartsyInference/Models/VAE/LTX-2/ltx-2.5-video-vae-bf16.safetensors   1472223346 bytes
```

It has **no `.swarm.json` sidecar** (the conv one does). Sidecars are optional metadata
(`T2IModelHandler.cs:372` lists them as *alt* metadata suffixes) and models are enumerated by file scan, so
it will be listed. If Swarm has not rescanned since the file appeared (2026-08-12), hit **Refresh Models** in
the UI first.

### A.3 ComfyUI auto-detects diffusion-vs-conv from the file. It is key-driven, not name-driven.

`dlbackend/ComfyUI/comfy/sd.py:587`:

```python
elif "decoder.conv_in_x_t.weight" in sd:  # lightricks LTX 2.4 diffusion VAE decoder
    vae_config = None
    if metadata is not None and "config" in metadata:
        vae_config = json.loads(metadata["config"]).get("vae", None)
    self.first_stage_model = comfy.ldm.lightricks.vae.na_diffusion_decoder.CausalDiffusionVAE(config=vae_config)
    self.latent_channels = sd["decoder.conv_in.weight"].shape[1]
    self.latent_dim = 3
    self.disable_offload = True
    …
    self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)
    self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 32, 32)
```

The conv VAE falls through to the `elif "decoder.conv_in.weight" in sd:` branch at line 601. So a plain
`VAELoader` pointed at the bf16 file gets the diffusion decoder automatically — **the registry is not
involved in decoder selection at all**, only in *which file* is loaded. Nothing else needs changing on the
Comfy side.

Two consequences worth knowing:

- `disable_offload = True` — Comfy will **not** offload this VAE mid-decode; it stays resident alongside
  whatever the DiT left behind.
- Comfy's own decode memory estimate for it is
  `1700 * T * H * W * 512 * dtype_size` (line 594) — see §C for the numbers.

### A.4 The tiling numbers: hardcoded for LTX-2, but user-overridable — and the override changes all four

`src/BuiltinExtensions/ComfyUIBackend/WGNodeData.cs:112-150`. The order is:

1. **If the user set `VAETileSize` *or* `VAETemporalTileSize`** (line 112) → `VAEDecodeTiled` with
   `UserInput.Get(VAETileSize, 256)`, `Get(VAETileOverlap, 64)`,
   `Get(VAETemporalTileSize, … 32)`, `Get(VAETemporalTileOverlap, 4)`.
2. **Else if `IsLTXV2()` and `ModelSpecificEnhancements` (default true)** (line 138) → `VAEDecodeTiled` with
   **hardcoded** `tile_size 2048, overlap 256, temporal_size 64, temporal_overlap 16`. ← this is what the
   conv row used.
3. **Else** → plain `VAEDecode`, untiled.

> **Landmine.** The branches are mutually exclusive and the fallbacks in branch 1 are **not** the LTX
> numbers. Setting only `vaetilesize` silently gives you `overlap 64 / temporal 32 / temporal_overlap 4`
> instead of `256 / 64 / 16`. If you override, **pass all four** (`vaetilesize`, `vaetileoverlap`,
> `vaetemporaltilesize`, `vaetemporaltileoverlap`).
>
> Setting `modelspecificenhancements: false` drops to plain untiled `VAEDecode` (branch 3) — that is the
> knob for an untiled comparison, not the tile params.

**What those numbers actually mean at the VAE.** `dlbackend/ComfyUI/nodes.py:354-372` divides the node
inputs by the VAE's compression before tiling:

```python
temporal_compression = vae.temporal_compression_decode()
if temporal_compression is not None:
    temporal_size = max(2, temporal_size // temporal_compression)
    temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
compression = vae.spacial_compression_decode()
images = vae.decode_tiled(latent, tile_x=tile_size // compression, tile_y=tile_size // compression,
                          overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
```

For the diffusion VAE, `spacial_compression_decode() = upscale_ratio[-1] = 32` and
`temporal_compression_decode() = round(upscale_ratio[0](8192)/8192) = round((8192*8-7)/8192) = 8`
(`comfy/sd.py:1426-1442`). So the shipped LTX-2 defaults resolve to:

| node input | ÷ compression | effective (latent units) |
|---|---|---|
| `tile_size 2048` | ÷32 | `tile_x = tile_y = 64` |
| `overlap 256` | ÷32 | `8` |
| `temporal_size 64` | ÷8 | `8` |
| `temporal_overlap 16` | ÷8 | `2` |

At 768x512x97f the latent is **W 24 x H 16 x T 13** (24 = 768/32, 16 = 512/32,
13 = `floor((97+7)/8)`).

**Therefore: spatial tiling does NOT engage (64 >> 24 and 16 — one tile), but temporal tiling DOES
(13 latent frames in chunks of 8 with overlap 2 → 2 chunks).** This is the single most important number in
this section: the Comfy arm at 768x512x97f is temporally tiled at ~8 latent frames per chunk, and that is
what its decode time will reflect.

**Recommendation: leave the tiling alone.** The conv row used branch 2's hardcoded values, and branch 2 fires
identically for the diffusion VAE (`IsLTXV2()` is a compat-class test —
`WorkflowGeneratorModelSupport.cs:49` — driven by the *checkpoint*, not the VAE). Changing tiling would
confound the VAE change with a tiling change. If the diffusion decode OOMs, *then* lower
`vaetemporaltilesize` — and pass all four params.

---

## B. Hartsy side — exact, revertible procedure

Two things must both be true: the env switch **and** the model folder carrying the bf16 VAE.

### B.1 The env switch

`src/HartsyInference.Engine/Recipes/Video/LtxVideo2Recipe.cs:140`:

```csharp
bool wantDiffusionVae = EnvSwitch.IsEnabled("HARTSY_LTX2_DIFFUSION_VAE", defaultOn: false);
```

`EnvSwitch.IsEnabled` (`src/HartsyInference.Core/Runtime/EnvSwitch.cs:14-31`): unset/empty → default;
`"1"`/`"true"` (any case) → true; `"0"`/`"false"` → false; **anything else silently falls back to the
default**. Use exactly `1`.

**How to set it for the user unit — systemd drop-in (recommended).** This adds one new file and touches
nothing existing, so revert is a single `rm`:

```bash
mkdir -p ~/.config/systemd/user/swarmui.service.d
cat > ~/.config/systemd/user/swarmui.service.d/ltx-diffusion-vae.conf <<'EOF'
[Service]
Environment=HARTSY_LTX2_DIFFUSION_VAE=1
EOF
systemctl --user daemon-reload
systemctl --user restart swarmui
```

Verify it took:

```bash
systemctl --user show swarmui -p Environment
```

Why a drop-in and not the alternatives:

- `~/bin/swarmui-run.sh` also exports env (`export HARTSYINFERENCE_MODELS=…`) and would work, but it is a
  shared launcher script and the edit is easy to forget.
- Editing `~/.config/systemd/user/swarmui.service` directly modifies a file with hand-written comments about
  the OOM containment — revert risks clobbering them.

The drop-in is additive and `rm` + `daemon-reload` restores the byte-identical prior state.

> The variable is **process-wide**. It applies to the whole SwarmUI process, therefore to **both**
> HartsyInference backends (id=7 on the 4090 and id=8 on the 3060) — see §B.3.

### B.2 The model folder — SWAP the symlink, do not ADD

The engine loads the **folder**, not the file. Proven by today's conv run,
`Data/Logs/2026-08/14-08-17.log`:

```
08:18:37.077 [Verbose] [HartsyInference] LTX-2.5 split bundle — loading the folder
  '/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/LTX-2.5' rather than the bare checkpoint,
  so the Gemma-4 tower and both VAEs are merged in.
08:18:37.501 [Info] [LtxVideo2Recipe] Converted 4091 DiT, 262 connector, 170 VAE, 102 audio-VAE,
  1227 vocoder, 678 text-encoder keys.
```

`LtxVideo2Recipe.AddFile` (`LtxVideo2Recipe.cs:276-300`) given a directory does
`Directory.GetFiles(path, "*.safetensors", SearchOption.AllDirectories)` and merges **every** shard into one
flat dictionary.

Current contents of `Models/Stable-Diffusion/LTX-2.5/`:

```
gemma4-12b-with-proj-ltx-2.5-int8_lean_convrot.safetensors -> ../../text_encoders/…
ltx-2.5-22b-dev-transformer-int8_lean_convrot.safetensors  -> ../../diffusion_models/…
ltx-2.5-22b-dev-transformer-int8_lean_convrot.swarm.json
ltx-2.5-audio-vae-bf16.safetensors      -> ../../VAE/LTX-2/ltx-2.5-audio-vae-bf16.safetensors
ltx-2.5-video-vae-conv-bf16.safetensors -> ../../VAE/LTX-2/ltx-2.5-video-vae-conv-bf16.safetensors
```

> **You MUST NOT leave both video VAEs in this folder.** The decoder is selected by
> `LtxVideo2CheckpointConverter.IsDiffusionVideoVae` (`LtxVideo2CheckpointConverter.cs:115-120`), whose
> signature key is `decoder.conv_in_x_t.weight`. In `Convert` (line 258) that is a **single boolean over the
> merged key set**:
>
> ```csharp
> bool diffusionVae = IsDiffusionVideoVae(allWeights.Keys);
> …
> if (diffusionVae && bucket == Ltx2Bucket.Vae && mapped is not null
>     && mapped.StartsWith("decoder.", StringComparison.Ordinal))
> {
>     bucket = Ltx2Bucket.VaeDiffusionDecoder;
> }
> ```
>
> With both files merged the flag flips true and **every** `decoder.*` key — including the conv decoder's —
> is routed into the diffusion bucket, and overlapping key names collide with last-write-wins ordered by
> `Directory.GetFiles`. That corrupts both decoders. Swap, never add.

**Swap to the diffusion VAE:**

```bash
cd /home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/LTX-2.5
rm ltx-2.5-video-vae-conv-bf16.safetensors
ln -s ../../VAE/LTX-2/ltx-2.5-video-vae-bf16.safetensors ltx-2.5-video-vae-bf16.safetensors
ls -la          # exactly ONE ltx-2.5-video-vae-*.safetensors must be present
```

(Both `rm` and `ln` act on the symlink only; the real 1.4 GB / 1.5 GB files in `Models/VAE/LTX-2/` are
untouched. Leave the audio VAE symlink alone.)

**Restart is mandatory after the swap.** Pipelines are cached by
`$"video-recipe:{spec.LocalPath}|…"` (`src/HartsyInference.Engine/InferenceEngine.cs:356`) and `LocalPath`
is the *folder*, which does not change when you re-point a symlink inside it — so a cached pipeline would
keep serving the old decoder. The §B.1 restart covers this if you do the swap first; otherwise restart
again.

### B.3 The other backend (id=8, the 3060)

`Data/Backends.fds` backend `8:` is `HartsyInference GPU1 (3060)`, `GPU_ID: 1`, `enabled: true`. The env var
is process-wide, so a gen routed there would also try the diffusion decoder. The 22B int8 DiT plus this
decoder will not fit in 12 GB.

Mitigation: **pin every generation** with the `exactbackendid` param rather than disabling backend 8.
`T2IParamTypes.cs:813` registers `"Exact Backend ID"` → id **`exactbackendid`**, enforced in
`src/Text2Image/T2IEngine.cs:81-99`:

```csharp
bool requireId = user_input.TryGet(T2IParamTypes.ExactBackendID, out string reqIdStr);
…
if (requireId && backend.ID != reqId && (backend.Parent?.ID ?? int.MaxValue) != reqId) { … return false; }
```

Use `"exactbackendid": "7"` for the Hartsy arm and `"exactbackendid": "1"` for the Comfy arm.

> **`exactbackendid` filters among *running* backends — it cannot enable a disabled one.** ComfyUI backend
> id=1 is currently `enabled: false` and must still be enabled through the UI/API (§4.1).

---

## C. Geometry

### The facts

- **Hartsy, untiled diffusion decode:** OOM at 768x512x97f — "requested 9312 MB but only 7981 MB available";
  runs at 512x320x97f in **38.4 s**. Recorded in the comment at `LtxVideo2Recipe.cs:130-138`.
- **Hartsy temporal chunking now exists in source.** `src/HartsyInference.Diffusion/Models/Vae/`
  `LtxVideo25TemporalChunks.cs` (created 08:57 on 2026-08-14) and `LtxVideo25DiffusionDecoder.cs` (modified
  08:54) with `ResolveChunkBytes` (line 271), `DefaultChunkBytes = 2 GiB` (line 27) and
  `LtxVideo25TemporalChunks.FramesPerChunk(chunkBytes, plane, bytesPerToken, frames)` (line 244).
- **But it is almost certainly NOT in the deployed extension.** The deployed
  `HartsyInference.*.dll` are timestamped **08:17**, i.e. ~40 minutes *before* the chunking source was
  written. The 08:56 build artifacts in `obj/Release/` are **net10.0**; the extension ships **net8.0**.
- **ComfyUI's own estimate**, `memory_used_decode = 1700 * T * H * W * (8*8*8) * dtype_size`
  (`comfy/sd.py:594`), at bf16 and latent 13x16x24:
  - full untiled: `1700 * 13*16*24 * 512 * 2` ≈ **8.1 GiB**
  - per temporal tile of 8 latent frames: ≈ **5.0 GiB**

  (Heuristic scheduling numbers, not measured allocations.)

### Recommendation

**Run the matched row at 768x512x97f / 30 steps / cfg 3.0 / 24 fps — identical to the conv row.** Matching
the conv row's geometry is the entire point of the matched row; a diffusion row at a different geometry
cannot be compared to anything.

This is **conditional on a smoke test**: after §B and a fresh deploy (§4.2), run **one** 768x512x97f Hartsy
gen and confirm it completes. Per §A.4 the Comfy arm is temporally tiled at this geometry regardless, so the
risk is entirely on the Hartsy side.

**If the smoke gen OOMs (i.e. chunking has not landed in the deployed build):**

- **Do not** mix geometries between arms. That produces a number that looks like a comparison and is not one.
- Drop **both** arms to **512x320x97f**, everything else identical, and label the result explicitly as a
  *secondary, non-comparable* row — it cannot be placed next to the 47.40 / 42.48 conv figures.
- Prefer, if the schedule allows: wait for the chunking work to be built and deployed, then run at
  768x512x97f. That yields the row that was actually asked for.

**Do not treat the recipe's log text as evidence about tiling.** `LtxVideo2Recipe.cs:147-148` prints
"Untiled: expect an OOM above ~512x320" and the comment at line 131 says "we have no tiling" — both are
stale the moment the chunking build deploys. Trust the deployed DLL timestamps and the actual gen result.

---

## D. Landmines (all re-verified against source/logs for this document)

1. **CUDA ordinal vs nvidia-smi index are inverted here.** CUDA is fastest-first: ordinal 0 = 4090.
   nvidia-smi index 1 = 4090. ComfyUI backend **id=1** (`GPU_ID: 0`) is the 4090; id=0 is the 3060.
   Hartsy backend **id=7** is the 4090; id=8 is the 3060. Both ComfyUI backends log `Device: cuda:0` — read
   the **GPU name**, never the ordinal. (§0.3, verified.)

2. **`bench_ltx25.py` refuses the hartsy arm on a stale deploy** — `assert_extension_matches_build()`
   (line 35) md5s every `HartsyInference.*.dll` and `Ptx/*.ptx` in
   `src/bin/extensions/SwarmExtensionSwarmUI-HartsyInference-Backend` against
   `src/HartsyInference.Cli/bin/Release/net8.0`, and `sys.exit(2)` on mismatch. Verified.
   **Two holes in that gate:**
   - It **skips entirely** if there is no local net8.0 build — line 41 prints
     `!!! no local net8.0 build … deploy check skipped` and continues.
   - It passes if the deployed DLLs match a **stale local build**. It compares deployed-vs-local, not
     local-vs-HEAD.
   - It does **not** cover the extension's own assembly (`SwarmUI-HartsyInference.dll`), only
     `HartsyInference.*.dll`.

   So: rebuild net8.0 from current source, deploy, restart — *then* run. That both arms the gate and answers
   "is chunking in the deployed build".

3. **Root filesystem at 100%, ~6.8 GB free, and 11 GB of coredumps already sitting there.** Unlimited
   `core_pattern`. A ComfyUI crash writes a multi-GB core and has previously taken the disk to zero and
   **truncated `Backends.fds`**. Clear coredumps, back up config to timestamped copies, delete outputs as you
   go. (§0.1, verified.)

4. **SwarmUI binds 192.168.10.188:7801 only** (`Data/Settings.fds:136-138`); localhost will not connect.
   Verified.

5. **Never hand-edit `Backends.fds`/`Settings.fds` while the service runs** — the process rewrites them.
   Toggle via UI/API. (§0.4.)

6. **Do not put both video VAEs in `Models/Stable-Diffusion/LTX-2.5/`** — the folder is merged wholesale and
   `IsDiffusionVideoVae` is a single flag over the merged key set. (§B.2, verified against source + today's
   log.)

7. **Overriding one VAE tile param silently changes all four** on the Comfy side. (§A.4, verified.)

8. **A restart is required after the symlink swap** — the pipeline cache keys on the folder path, which does
   not change. (§B.2, verified.)

9. **`HARTSY_LTX2_DIFFUSION_VAE` accepts only `1`/`true`/`0`/`false`**; any other value silently falls back
   to the default (off). (`EnvSwitch.cs:14-31`, verified.)

10. **[UNPROVEN — flagged deliberately] The deployed extension contains code that is not in either SwarmUI
    working tree.** The log line `[HartsyInference] LTX-2.5 split bundle — loading the folder … rather than
    the bare checkpoint` is emitted at runtime, but a search for `"rather than the bare checkpoint"`,
    `"Gemma-4 tower"` and `"split bundle"` across `/home/hartsy/Desktop/` found it **only in log files** —
    not in `src/Extensions/SwarmUI-HartsyInference-Backend/`, not in the parallel
    `/home/hartsy/Desktop/Swarm/SwarmUI/` tree, and not in the HartsyInference repo. So the folder-loading
    behaviour is proven **empirically** (the log, plus `AddFile`'s directory branch at
    `LtxVideo2Recipe.cs:289`) but its call site could not be located. Treat the deployed extension as
    possibly ahead of source, and rely on §4.2's rebuild + the run's own log lines rather than on reading
    the extension tree.

---

## Procedure

### 1. Pre-flight
- [ ] `df -h /`; clear `/var/lib/systemd/coredump/`; confirm ample free space.
- [ ] `cp -a Backends.fds Backends.fds.bak.$TS` and same for `Settings.fds`.
- [ ] Record the conv-row baseline you are matching: 768x512x97f/30, Hartsy 47.40 s, Comfy 42.48 s.

### 2. Deploy a current Hartsy build
- [ ] Build `HartsyInference.Cli` **Release / net8.0**.
- [ ] Copy `HartsyInference.*.dll` **and** `Ptx/*.ptx` into
      `src/bin/extensions/SwarmExtensionSwarmUI-HartsyInference-Backend`.
- [ ] Confirm whether temporal chunking is actually in the deployed build — this decides §C:
      ```bash
      strings -e l "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/src/bin/extensions/SwarmExtensionSwarmUI-HartsyInference-Backend/HartsyInference.Diffusion.dll" | grep -i TemporalChunks
      ```
      (`strings -e l` is required — these assemblies store UTF-16 literals. No hits = no chunking.)

### 3. Apply the Hartsy changes
- [ ] Swap the symlink (§B.2). Confirm exactly one `ltx-2.5-video-*-vae-*.safetensors` in the folder.
- [ ] Create the systemd drop-in (§B.1).
- [ ] `systemctl --user daemon-reload && systemctl --user restart swarmui`
- [ ] `systemctl --user show swarmui -p Environment` shows `HARTSY_LTX2_DIFFUSION_VAE=1`.
- [ ] `systemctl --user status swarmui` is `active (running)`.

### 4. Enable the Comfy backend
- [ ] In the SwarmUI UI (Server → Backends), enable backend **id=1** (`ComfyUI Self-Starting`, `GPU_ID: 0`).
- [ ] Wait for it to report ready; confirm in `Data/Logs` that it printed
      `[ComfyUI-1/STDERR] … Device: cuda:0 NVIDIA GeForce RTX 4090`.
- [ ] Consider leaving Hartsy id=7 enabled at the same time only if VRAM allows — a 22B model resident in
      both engines will not fit on one 24 GB card. Safer: run the arms **one at a time**, enabling only the
      arm under test, and still pass `exactbackendid` as a belt-and-braces guard.
- [ ] **`systemctl --user restart swarmui` between arms.** Disabling Hartsy backend id=7 through the UI is
      **not** verified to release the engine's VRAM: HartsyInference runs **in-process** inside SwarmUI (not
      as a subprocess like ComfyUI), and the engine deliberately keeps weights resident across gens
      (`HARTSY_KEEP_MODELS`, referenced at `InferenceEngine.cs:192-194`). A Comfy-arm gen sharing the 4090
      with a still-resident 22B Hartsy model is the largest remaining cross-arm confound. A restart costs
      ~5 s and the protocol already absorbs a cold gen.

### 5. Smoke test each arm (one gen each, before the 6-gen protocol)
- [ ] **Hartsy**, 768x512x97f. Expect **both** log lines:
      1. `[HartsyInference] LTX-2.5 split bundle — loading the folder … rather than the bare checkpoint`
      2. `[LtxVideo2Recipe] HARTSY_LTX2_DIFFUSION_VAE set — decoding with the LTX-2.5 diffusion video decoder`

      **Check line 1 explicitly, because step 2 replaced the deployed `HartsyInference.*.dll` with builds
      from current HEAD and landmine 10 says that behaviour's call site is not in any on-disk source.** If
      the folder-loading regressed, `conv.Vae.Count == 0` and the recipe silently downloads the **LTX-2.3**
      side VAE instead (`LtxVideo2Recipe.cs:91`) — none of the three diffusion-VAE branches fire, no error
      is raised, and the run measures the wrong decoder entirely. Line 2 missing while line 1 is present
      means the env var did not take; line 1 missing means the redeploy regressed the bundle loading.

      If it OOMs → go to §C's fallback.
- [ ] **Comfy**, 768x512x97f. In the saved workflow JSON — it is echoed into
      `Data/Logs/2026-08/*.log` (that is where the conv row's evidence came from) — confirm
      `"vae_name": "LTX-2/ltx-2.5-video-vae-bf16.safetensors"` (the conv row shows
      `…-conv-bf16.safetensors` — that string is the tell).
- [ ] **Confirm the `exactbackendid` pin actually took.** The param is registered with
      `Permission: Permissions.ParamBackendID` (`T2IParamTypes.cs:813`); if the anonymous
      `GetNewSession` session lacks that permission the param may be rejected or silently ignored. Do not
      assume — check *which backend served the request*: Hartsy shows `[LtxVideo2Recipe]` / `[HartsyInference]`
      lines, Comfy shows `[ComfyUI-1/…]` activity. A gen served by `[ComfyUI-0]` or Hartsy id=8 is on the
      3060 and must be discarded.
- [ ] **Watch both output videos.** Matching durations, frame counts and file sizes prove nothing; a wrong
      decoder can still produce a plausible-sized file.

### 6. Run the benchmark

`bench_ltx25.py` hardcodes `PARAMS`/`COMMON` (lines 75-77) with **no** `vae` and **no** `exactbackendid`,
and routes purely by which backend is enabled (docstring, line 8-9). Add per arm:

- **Comfy arm** — add to the payload:
  ```python
  {"vae": "LTX-2/ltx-2.5-video-vae-bf16.safetensors", "exactbackendid": "1"}
  ```
- **Hartsy arm** — add **only**:
  ```python
  {"exactbackendid": "7"}
  ```
  **Do not pass `vae` on the Hartsy arm.** Its VAE selection is folder + env driven (§B); whether the
  extension tolerates or ignores a `vae` param on a `hartsyinference` backend was **not verified**, and an
  ignored-but-present param is exactly the kind of thing that makes a row unreproducible.

Protocol is unchanged: 1 cold gen + 5 warm, random seed each, peak VRAM sampled on nvidia-smi index 1.

### 7. Record
- [ ] Both arms' warm mean, cold wall, peak VRAM.
- [ ] Explicitly record the geometry and that **both** arms used the diffusion VAE.
- [ ] Record whether the Hartsy build had temporal chunking, and Comfy's effective tiling
      (tile_t 8 / overlap_t 2 latent frames at the shipped defaults, §A.4).

---

## Revert checklist

Work top to bottom; each step is independent and none requires a rebuild.

1. **Env var**
   ```bash
   rm ~/.config/systemd/user/swarmui.service.d/ltx-diffusion-vae.conf
   rmdir ~/.config/systemd/user/swarmui.service.d 2>/dev/null
   systemctl --user daemon-reload
   ```
2. **Symlink** — restore the conv VAE:
   ```bash
   cd /home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/LTX-2.5
   rm -f ltx-2.5-video-vae-bf16.safetensors
   ln -s ../../VAE/LTX-2/ltx-2.5-video-vae-conv-bf16.safetensors ltx-2.5-video-vae-conv-bf16.safetensors
   ls -la      # must match the listing in §B.2
   ```
3. **Restart** so the pipeline cache drops the diffusion decoder:
   ```bash
   systemctl --user restart swarmui
   systemctl --user status swarmui
   ```
4. **Backend enablement** — disable ComfyUI backend id=1 again **through the UI/API** (it was
   `enabled: false`). Re-enable Hartsy id=7 and id=8 if you disabled them.
5. **Bench script** — revert the `vae` / `exactbackendid` additions to `bench_ltx25.py`.
6. **Config files** — if anything looks wrong, restore from the `§0.1` timestamped copies **with the service
   stopped**, then start it. Delete the `.bak.$TS` copies once satisfied.
7. **Disk** — delete the run's output videos; re-check `df -h /`.
8. **Sanity** — one 768x512x97f gen on Hartsy should log the *conv* decoder line
   (`… using the conv decoder, which has no geometry ceiling`) and land near **47.4 s**.

## What is NOT changed by any of the above

`CommonModels.cs`, the `Models/VAE/LTX-2/` files themselves, `Data/Settings.fds`, the systemd unit file, and
the SwarmUI binary. That is deliberate — every change in this runbook is a symlink, a drop-in file, an API
parameter, or a UI toggle.

---

## Independent re-verification of the `vae` param override (2026-08-14)

The whole ComfyUI arm rests on one claim — that a per-generation `vae` param beats the known-models table — so it
was checked a second time, by a different reader, against the source rather than the doc.

**Confirmed, with the guard the first pass did not mention.** `ModelLoadHelpers.DoVaeLoader`
(`WorkflowGeneratorModelSupport.cs:527-541`) reads:

```csharp
if (!g.NoVAEOverride && g.UserInput.TryGet(T2IParamTypes.VAE, out T2IModel vaeModel))
{ vaeFile = vaeModel.Name; nodeId = "11"; }
...
if (string.IsNullOrWhiteSpace(vaeFile) && knownFile is not null && ...) { vaeFile = knownFile.FileName; }
```

The user param wins, and `CommonModels.Known[...]` is only consulted when it is empty — but the whole branch is
gated on `!g.NoVAEOverride`, which the first pass did not check. **`NoVAEOverride = true` is set in exactly two
places** (`WorkflowGenerator.cs:2732` and `:2750`), both on the **PiD pixel-decoder** paths
(`CreatePidCompatLatent`, `CreatePixelDecode`). Neither is on a plain text-to-video decode, so for this benchmark
the flag is false and the override applies.

Worth restating because it is the failure mode that would waste a campaign: if `NoVAEOverride` *were* set, the
`vae` param would be **silently ignored** and Comfy would decode with the conv VAE from the registry while the
run looked entirely healthy — producing a "diffusion-vs-diffusion" row that was really conv-vs-diffusion. So the
runbook's instruction to read the VAE path out of the logged workflow JSON is not belt-and-braces; it is the only
thing that distinguishes those two outcomes. Do not skip it.
