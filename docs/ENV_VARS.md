# Environment variables — full inventory and disposition

> **What this doc is for.** The list of variables is recoverable from the code (see *Regenerating* at the
> bottom), so the list alone would not earn its place under `docs/README.md`'s rule. What is **not**
> recoverable is the **Disposition** column: whether a knob is a supported control, an undocumented
> default-ON numerics switch, a debug hook that silently corrupts output, or a name that only survives in a
> doc. That judgement is the point of this file; the table is the scaffolding it hangs on.

> ⚠️ **Superseded in part, 2026-08-26.** `EngineKnobs` (`src/HartsyInference.Core/Configuration/`) is now the
> declared registry: **210 knobs**, each with an id, type, default, scope, domain and its legacy environment
> name. `KnobRegistryTests` ties that surface to a source scan, so the registry — not this file — is the
> authority on *what exists*. This doc remains the authority on **disposition**, and on the history below.
>
> **How to set something now.** Environment variables still work — every knob records its legacy name and
> `KnobStore` honours it — but they are the lowest-precedence layer and will be retired in C7. Prefer:
>
> | Where | How |
> |---|---|
> | CLI | `--set numerics.sageAttn=0`, `--profile reference`, `--list-settings` |
> | API | `{"settings": {"profile": "reference", "set": {"numerics.ditF16": "0"}}}` |
> | Precedence | scoped profile → process override → legacy env var → declared default |
>
> `--profile reference` pins 49 knobs to their most numerically faithful setting for parity work.
>
> ⚠️ **Per-request settings reach per-call knobs only.** Anything bound while the engine, backend or pipeline is
> built is already fixed when a request arrives on a long-lived server — including the backend's TF32, F16-GEMM
> and native-FP8 decisions, which are assigned in `CudaBackend`'s constructor. A request naming a load-time
> setting is rejected with 400 rather than silently ignored. The CLI does not have this limit, because it applies
> settings before the engine is constructed.

**Scale.** The original estimate of "~45 real knobs" and the first inventory's "146" were both wrong. A
literal scan for `GetEnvironmentVariable("HARTSY_…")` has **five** blind spots, each found the hard way:

| Blind spot | Example | Count |
|---|---|---|
| Named after the model, not the engine | `WAN_SOLVER_ORDER`, `LTX_DIAG`, `QWEN3_DEBUG` | 18 |
| Reached only through a helper | `EnvFlag("HARTSY_NO_TF32")` — the GEMM/SDPA family | 13 |
| Held in a `const`, passed by reference | `HARTSY_AUDIO_LM_QUANT`, `HARTSY_ANIMATE2_BF16_DRIVING_CACHE` | 4 |
| A helper **constructor** argument | `new DebugDumpSink("WAN_DEBUG_DIR")` | 19 |
| Only ever a **default parameter value** | `FromEnvironment(string v = "HARTSY_CFG_INTERVAL")` | 1 |

The real engine surface is **~209 knobs**. A further ~150 names are test-fixture paths read by test code, not
the engine, and are out of scope.

**Three boolean grammars are live and they genuinely disagree** — this is why the migration preserves each
rather than unifying them:

- **Exact** (the historic `== "1"` / `!= "0"` sites): only the exact opposite-of-default spelling flips it, so
  on a default-ON knob `HARTSY_X=false` resolves to **true**.
- **TriState** (`EnvSwitch.IsEnabled`): `true`/`false` are also recognized.
- **Presence-only** (`is null`): **any** set value enables it, **including `0`**. `MUSICGEN_GRAPH_OFF=0` still
  disables graph decode. These are declared as `string?` knobs so that stays true.

---

## 1. Environment variables are the FALLBACK, not the mechanism

For anything with a real configuration home, the env var is the lowest-precedence layer. VRAM is the
worked example, and the shape it uses is the one new settings should copy:

| Precedence | Layer | Scope | Set by |
|---|---|---|---|
| 1 (wins) | `VramPolicyScope` | one generation | the request's `vram` field |
| 2 | `VramPolicyRegistry` | one backend | `EngineOptions.VramPolicy` (SwarmUI backend setting) |
| 3 | `HARTSY_LOWVRAM` | whole process | operator / systemd unit |
| 4 | code default | — | `VramTier.Auto` |

```csharp
// VramPolicyRegistry.Resolve
if (VramPolicyScope.Current is VramPolicy scoped) return scoped;             // per-request
if (_policies.TryGetValue(backend, out VramPolicy? pinned)) return pinned;   // per-backend
return VramPolicyResolver.FromLegacyMode(LowVramPolicy.ResolveEnvironment());// env
```

An individual lever left on `Auto` falls back to its own env var the same way
(`VramLevers.Resolve`: `On → true`, `Off → false`, `Auto → EnvSwitch.IsEnabled(envVar, defaultOn)`).

**Why this matters:** the process-wide env var is last-writer-wins, which is precisely what broke
multi-backend setups — two SwarmUI backends on differently-sized cards could not hold different policies.
That defect is the reason layers 1 and 2 exist.

### The same control, three ways

```bash
# HTTP — per request, nothing exported
POST /v1/native/images
{ "model": "krea2",
  "request": { "prompt": "a red fox in snow", "vram": { "tier": "Aggressive" } } }

# ...or pin individual levers instead of a whole tier
"vram": { "weightStreaming": "On", "keepResident": "Off", "caches": "Half", "chunkScale": 0.5 }

# CLI — per run
hartsy image "a red fox in snow" -m krea2 --vram-mode Aggressive

# SwarmUI — per backend, in Server → Backends
VramMode = Aggressive        # plus per-generation overrides in the "VRAM & Memory" param group
```

camelCase, string enums, case-insensitive. Omitting `vram` means "follow the backend" — it is not an empty
override that decides something. `VideoRequest` and `MusicRequest` carry the identical shape.

---

## 2. Disposition legend

| Tag | Meaning |
|---|---|
| **KEEP** | Supported control, correctly scoped and documented. |
| **DOCUMENT** | Load-bearing (often default-ON and output-affecting) but in no doc. Cheapest fix in this file. |
| **FIX** | Works, but the grammar/scope/name is a trap. |
| **FOLD** | Superseded — belongs in `VramPolicy` or a settings object, not an env var. |
| **DELETE** | Dead, or a debug hook that has no business in a shipping binary. |
| **TEST** | Test-fixture path; harmless, out of scope for cleanup. |

---

## 3. Configuration & paths

| Name | Default | Gates | Disposition |
|---|---|---|---|
| `HARTSY_LOWVRAM` | `Auto` | Streaming/eviction posture | **FOLD** — layers 1–2 above supersede it; keep only as back-compat |
| `HARTSY_KEEP_MODELS` | **ON** | Weights stay resident between generations | **FOLD** — already absorbed into `VramPolicy.KeepResident`; env is now just the `Auto` fallback |
| `HARTSY_LOWVRAM_QUANT` | off | LLM quantized-GEMM path | **FIX** — name collides with `HARTSY_LOWVRAM` but is unrelated (numerics, not placement); read at two sites that can disagree |
| `HARTSY_AUDIO_LM_QUANT` | `q4k` / `off` when sharded | Audio-LM weight quantization | **KEEP** — documented, sane enum grammar |
| `HARTSY_LOG_LEVEL` | `Info` | CLI log level | **KEEP** |
| `HARTSYINFERENCE_MODELS` / `_MODEL_CACHE` / `_OUTPUT` / `_REPO_ROOT` | repo-derived | Path roots | **KEEP** (rename for prefix consistency — see §7) |
| `HF_TOKEN` / `HF_ENDPOINT` | none | HuggingFace auth / mirror | **KEEP** — third-party convention, not ours to rename |
| `NO_COLOR` / `COLORFGBG` / `ESPEAK_DATA_DIR` | — | Terminal + espeak conventions | **KEEP** — external standards |
| `HARTSY_SAME_GPU_CONCURRENT` | off | Bypasses the per-device gate | **KEEP** (documented; genuinely experimental) |
| `HARTSY_AUDIO_EVICT_BELOW_GB` | **14** | Free-VRAM threshold for audio eviction | **DOCUMENT** — a hard-coded GB threshold that silently changes behavior near it |
| `HARTSY_STREAM_PIN` | **ON** | Pinned host staging for streaming | **DOCUMENT** |

## 4. CUDA numerics — the largest undocumented surface

Every one of these changes output bits or kernel selection. **All are default-ON and none are documented.**
Setting any to `0` silently changes results, which makes them the most dangerous group in this file.

| Name | Default | Disposition |
|---|---|---|
| `HARTSY_SDPA_CUDNN` | **ON** | **DOCUMENT** — switches the entire attention backend |
| `HARTSY_CONV_CUDNN`, `HARTSY_AUDIO_CONV_CUDNN` | **ON** | **DOCUMENT** — switches all convolution |
| `HARTSY_EPILOGUE_FUSION`, `HARTSY_DP4A_ON`, `HARTSY_ROPE_V2`, `HARTSY_FP8_STATIC_INPUT_SCALE`, `HARTSY_MODULATE_EMIT_FP8` | **ON** | **DOCUMENT** — standard-profile numerics |
| `HARTSY_INT8_FUSED_MMA`, `HARTSY_INT8_MMA_SWIZZLE`, `HARTSY_GROUPED_LINEAR`, `HARTSY_QUANT_AT_PRODUCER`, `HARTSY_BF16_GEMV` | **ON** | **DOCUMENT** |
| `HARTSY_SSM_DELTA_V2`, `HARTSY_SSM_DELTA_WARPROW`, `HARTSY_SSM_GRAPH`, `HARTSY_SSM_DEVICE_STEP` | **ON** | **DOCUMENT** — four undocumented SSM switches |
| `HARTSY_QK_FUSION`, `HARTSY_QK_SCATTER`, `HARTSY_QKNORM_SCATTER`, `HARTSY_SANDWICH_FUSION` | **ON** | **DOCUMENT + FIX** — LLM hot path, and three of them overlap (see §6) |
| `HARTSY_LTX2_GATEFUSE`, `HARTSY_LTX2_TOKENMAJOR`, `HARTSY_LTX25_NA3D_TILED`, `HARTSY_VAE_FULLRES`, `HARTSY_ORPHAN_SWEEP`, `HARTSY_AURAFLOW_PACKED`, `HARTSY_CSM_CFG_BATCH`, `HARTSY_CSM_CFG_GRAPH`, `HARTSY_MM3_DEPTH_QUANT`, `HARTSY_MM3_CFG_BATCH` | **ON** | **DOCUMENT** |
| `HARTSY_W8A8` | off | **DOCUMENT** — quality-affecting, undocumented |
| `HARTSY_FP8_NATIVE` | **hardware-dependent** | **DOCUMENT** — unset behavior is unknowable from the name |
| `HARTSY_CFG_INTERVAL` | `Always` | **DOCUMENT** — changes sampling **and is the only var that can throw** on a malformed value (deliberate: a mistyped perf knob must not silently invalidate an A/B run) |
| `HARTSY_GEMV_WPB` (4), `HARTSY_GEMV_KSPLIT` (-1), `HARTSY_IM2COL_BAND_MB` (1 GB), `HARTSY_INT8_ROW_BUDGET_MB` (0), `HARTSY_GRAPH_ARENA_MB` (32), `HARTSY_AUTOPROMOTE_HEADROOM_MB` (1536) | various | **DOCUMENT** — numeric tuning with non-obvious defaults and clamps |
| `HARTSY_CUDNN_AUTOFETCH` + `_URL` + `_VERSION` | off | **FIX** — runtime download of a native library from an operator-supplied URL is a code-execution vector in a shipping binary. Default-off, but it should not be reachable by env var alone |

## 5. Debug hooks that corrupt output

These produce plausible-looking but **wrong** results with no error. They are the strongest deletion
candidates: a support report from anyone who set one is unfalsifiable from the output alone.

| Name | What it does | Disposition |
|---|---|---|
| `HARTSY_SKIP_ROPE` | **Skips RoPE entirely** — corrupts every image | **DELETE** (also read 3× per RoPE call, see §8) |
| `HM_ROPE_CPU` | Self-labeled `// TEMP perf-repro gate`; old CPU fallback | **DELETE** — temporary code that shipped |
| `HARTSY_VLM_ENCODE_ONLY` | `return "";` mid-generation — empty output, no error | **DELETE** |
| `HARTSY_ANIMATE_NO_POSE`, `HARTSY_ANIMATE_NO_FACE` | Disable conditioning; output looks plausible, is wrong | **DELETE** |
| `HARTSY_SAGE_UNSAFE_F32_V_NARROW` | F32→F16 V narrowing; ∞ on out-of-range values | **KEEP** — correctly quarantined behind a *second* explicit opt-in |
| `HARTSY_PROFILE_SYNC` | Device sync around every NVTX range — serializes the pipeline | **KEEP** (profiling-only) but **FIX** the `static readonly` scope so it can't be stuck on |
| `HARTSY_PROFILE_EACH` + `HARTSY_PROFILE_OUT` | Writes a profile after **every** generation to a fixed `/tmp/hartsy_profile.txt` | **FIX** — no PID in the path; two processes clobber each other |
| ~24 `*_DEBUG_DIR` / `*_DUMP` vars (§O of the sweep) | Unbounded tensor dumps from production paths | **KEEP** as a family but **FIX** — one `HARTSY_DEBUG_DIR` + a model selector beats 24 names |

## 6. Duplicates and overlapping knobs

| Cluster | Problem | Disposition |
|---|---|---|
| `HARTSY_SDPA_V2` / `_FORCE_FLASH` / `_FORCE_TILED` | Three overlapping forcing knobs on one dispatch | **FOLD** → `HARTSY_SDPA_KERNEL=auto\|v2\|flash\|tiled` |
| `HARTSY_SDPA_F16` / `HARTSY_SDPA_NO_F16` | A flag and its own negation; both `=1` is undefined | **FOLD** → one tri-state |
| `HARTSY_FLASH_SPLIT_FORCE` / `_OFF` | Same force/disable anti-pattern | **FOLD** → one tri-state |
| `HARTSY_QK_FUSION` / `HARTSY_QK_SCATTER` / `HARTSY_QKNORM_SCATTER` | The code itself says one "mirrors" another | **FOLD** → one switch |
| `ARCFACE_WEIGHTS` / `ARCFACE_WEIGHTS_PATH` | Both read in one test | **FIX** (test-only) |

> **Checked and NOT a duplicate:** `HARTSY_SAGE_ATTN` is read two ways — `UseSageAttn` (`!= "0"`, default ON)
> and `SageExplicitlyEnabled` (`== "1"`, default OFF). That looks like an inconsistency and is not: the
> second gates the *unsafe* F32-narrowing path, which deliberately requires an explicit opt-in rather than
> accepting "left on by default". Sound design; leave it alone.

## 7. Naming and grammar inconsistency

**Four prefixes for one product:** `HARTSY_*` (~153) · `HARTSYINFERENCE_*` (15, mostly Vulkan) ·
**unprefixed model names** (38: `SD3_DEBUG_DIR`, `WAN_SOLVER_ORDER`, `LTX_DIAG`, `QWEN3_DEBUG`, `HM_ROPE_CPU`,
`HYV_VAE_STAGES`, `HIFT_DETERMINISTIC`, …) · `AUDIOLAB_*` (2, extension-local).

`HM_ROPE_CPU` and `HYV_VAE_STAGES` are unprefixed *and* cryptic — a real collision risk with unrelated
software in the same process.

**Six value grammars coexist**, and three of them are traps:

| Grammar | Count | Trap |
|---|---|---|
| `EnvSwitch` tri-state (`1`/`true`/`0`/`false`) | ~35 | The intended convention |
| Strict `== "1"` | ~110 | `HARTSY_X=true` does **nothing** |
| Inverted `!= "0"` | ~25 | `HARTSY_BF16_GEMV=false` **enables** it |
| Inverted `!= "1"` | 4 | Double negatives (`HARTSY_NO_AUTOPROMOTE`) |
| Presence-only | 2 | **`HARTSY_MUSICGEN_GRAPH_OFF=0` DISABLES graph decode** — the opposite of what anyone would expect |
| Word enums (`on`/`off`, `q4k`) | 2 | Fine |

Plus a magic string: `HARTSY_SAGE_PV` compares against `"f16acc"`; any other value silently means off.

**Config-dependent defaults** (`HARTSY_FP8_NATIVE`, `HARTSY_LTX2_ANCESTRAL`, `HARTSY_LTX2_TWO_STAGE`,
`HARTSY_DIT_GRAPH` — which has two different defaults for one name) mean the unset behavior cannot be known
from the name. `HARTSY_DIT_GRAPH`'s split is deliberate and documented in code; the others are not.

**Cross-repo parse mismatch:** `HARTSY_LTX2_DIFFUSION_VAE` is parsed by `EnvSwitch` (tri-state) in the engine
but read raw in the SwarmUI extension — `=true` enables the engine path while the extension's validation
disagrees. **FIX.**

## 8. Hot-path reads

Each is a managed→native lookup with a string allocation **inside a per-step / per-block / per-token loop**.
All are one-line fixes to `static readonly`, and all are inconsistent with the codebase's own dominant
pattern (`NvtxRange`, `CudaPeerAccess`, `KvCaches`, `DitRuntimeFlags` all cache correctly).

`HARTSY_SKIP_ROPE` (×3 per RoPE call) · `HM_ROPE_CPU` · `HARTSY_SDPA_V2` · `HARTSY_SDPA_FORCE_FLASH` ·
`HARTSY_SDPA_FORCE_TILED` (×2) · `HARTSY_SDPA_TILE` · `HARTSY_FLASH_SPLIT_FORCE` · `HARTSY_FLASH_SPLIT_OFF` ·
`HARTSY_SAGE_PV` (+ a string compare per kernel launch) · `HARTSY_SAGE_ATTN` · `HARTSY_SAGE_F16_MIN_SKV`
(+ `int.TryParse`) · `HARTSY_QK_FUSION` · `HARTSY_QKNORM_SCATTER` · `HARTSY_QK_SCATTER` ·
`HARTSY_SANDWICH_FUSION` (×2 per layer per token) · `HARTSY_SSM_DEVICE_STEP` · `HIFT_DETERMINISTIC` ·
`HARTSY_LTX25_NA3D_TILED` · `ACE_STEP_DEBUG_DIR` (re-resolved on every dump check, even when off).

> **Do not "fix" by caching everywhere.** `LowVramMode` and `VramLevers` both document a *deliberate*
> decision **not** to cache, because SwarmUI writes those after a warm-up generation has already run. That
> reasoning applies to per-generation policy vars only — never to anything in the list above.

## 9. Dead — and worse, documented as working

Each of these appears in a doc as a usable knob and **has no read site**. An operator following the docs
gets silence. These are worse than undocumented vars, because the doc actively misleads.

| Name | Where it is claimed |
|---|---|
| `HARTSY_DUMP_CONTROL` | `docs/Checklists/MODEL_STATUS_IMAGE.md:265` documents it as a working debug aid |
| `HARTSY_HEARTMULA_QUANT` | `docs/Research/HEARTMULA_ARCHITECTURE.md:49,58` + `MODEL_STATUS_AUDIO.md:460`, **with benchmark numbers attached** |
| `HARTSY_AUDIO_CUDA_DEVICE` | `benchmarks/scoreboards/AUDIO.md:7` states audio "is usually pinned here via" it — a methodology note describing a knob that does nothing |
| `HARTSY_LTX2_AUDIO_RESCALE` | `MODEL_STATUS_VIDEO.md:180` claims "default 1.0" |
| `HARTSY_CAST_STATS` | `MINIMAX_H3.md:527` — the doc itself admits it "emits nothing" |
| `HARTSY_CUDA_GRAPH` | `CUDA_GRAPH_FINDINGS.md:47`, superseded by `HARTSY_DIT_GRAPH` |
| `HARTSY_SDPA_V2_OFF` | `FLASH_ATTENTION_PLAN.md:65` — a planned kill-switch never built |
| `HARTSYINFERENCE_BENCH_OUT` | `PROFILING_METHODOLOGY.md:127`, in a copy-paste command line |
| `HARTSY_MODEL_GGUF_PATH` | `ADD_MODEL.md:55` — a code *template* placeholder, not a real var |

**Compliance, not cleanup:** `HARTSYINFERENCE_ACCEPT_TENCENT_HUNYUAN_LICENSE` is specified in
`HUNYUAN_GAMECRAFT_ARCHITECTURE.md:48,238,245` as a **license gate** and is not implemented. If GameCraft
loading ships, that is a licensing gap, not a dead variable.

**Also stale in the other direction:** `MINIMAX_MUSIC3_PERF.md:117` says `HARTSY_MM3_FLOW_CFG_BATCH` "stays
off", but the code reads `defaultOn: true`. Verified — the doc contradicts the default.

## 10. Test-only fixture paths (~150) — **TEST**

Parity/reference-weight paths read only under `tests/`, e.g. `MIMI_WEIGHTS`, `SNAC_REF_IO`,
`ARCFACE_WEIGHTS`, `LTX25_INT8_CHECKPOINT`, `GPT_OSS_ORACLE_JSON`. Uniform shape: path in, skip-or-run a
parity test, operator-supplied, undocumented by design. Harmless and out of scope — but they are why the
raw count reads ~195 rather than ~45.

Harness gates worth knowing: `HARTSY_REQUIRE_REAL_WEIGHTS` (exported unconditionally by
`tests/run-multigpu-campaign.sh:71`), `HARTSY_TEST_GPU`, `PARITY_BACKEND`.

## 11. Shell-only (never read by C#) — **KEEP**

`HARTSY_RESTART_DELAY_SECS`, `HARTSY_CRASH_WINDOW_SECS`, `HARTSY_MAX_CRASHES_IN_WINDOW`
(`deploy/run-with-restart.sh`) · `CUDA_LIB` / `CUDA_INC` (six kernel `build.sh` copies) · `ASPNETCORE_URLS` ·
`HartsyInference__*` (standard .NET `IConfiguration` double-underscore binding, all commented out in the
systemd unit — this is the only env→settings config layer and it is generic .NET, not custom code).

---

## Suggested order of work

1. **DELETE the output-corrupting debug hooks** (§5, five vars). Highest risk, lowest effort, no migration.
2. **Fix the three grammar traps** (§7): `HARTSY_MUSICGEN_GRAPH_OFF`'s presence-only parse, the
   `HARTSY_LTX2_DIFFUSION_VAE` cross-repo mismatch, and the `!= "0"` family that treats `false` as true.
3. **Correct the nine misleading docs** (§9). Pure doc edits; each one currently costs somebody a debugging session.
4. **DOCUMENT the default-ON numerics switches** (§4). No code change — but today an operator cannot know
   which knobs alter output.
5. **Cache the hot-path reads** (§8), respecting the do-not-cache carve-out.
6. **FOLD the duplicate clusters** (§6) behind single tri-state knobs.
7. **FOLD `HARTSY_LOWVRAM` / `HARTSY_KEEP_MODELS`** once every host sets a policy — they are already
   back-compat-only.

## Regenerating the raw list

The inventory is mechanical; the dispositions are not. To re-derive names after a refactor:

```bash
{ grep -rhoE 'GetEnvironmentVariable\("[A-Z0-9_]+"' --include="*.cs" src
  grep -rhoE 'EnvSwitch\.(IsEnabled|GetInt|GetFloat|GetLong)\("[A-Z0-9_]+"' --include="*.cs" src
  grep -rhoE '(EnvVar|EnvironmentVariable)\s*=\s*"[A-Z0-9_]+"' --include="*.cs" src
} | grep -oE '"[A-Z0-9_]+"' | tr -d '"' | sort -u
```

That misses three indirections, which is how a hand-written list goes stale: `EnvFlag(name)`
(`CudaBackend.cs:941`, 16 call sites), the `DebugDumpSink(envVar)` constructor (18 `*_DEBUG_DIR` vars whose
names appear only as ctor arguments), and default-parameter indirection
(`GuidanceInterval.FromEnvironment(string variable = "HARTSY_CFG_INTERVAL")`).
