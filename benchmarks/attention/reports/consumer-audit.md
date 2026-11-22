# Consumer audit: attention computed outside the backend entry points

Plan items 14 to 17 ask for every package to be audited for attention that bypasses the backend entry points
(`IBackend.ScaledDotProductAttention`, `IBackend.FlashAttention`, the planner, or the shared
`Core/Layers/Attention/AttentionLayer`). This is that audit at `main` `f9a0cedeb`, with the statuses of the first table
brought up to `d5492a019`. It is a source audit: it says what the code does, not what any model produces.

## How it was done

A source sweep of every package for host loops that take a softmax over `q · k` scores, then a second pass that
opened each hit. I read the eight sites marked "read by me" myself; the rest were read by a read-only sub-agent whose
findings I have not all re-read line by line. Line numbers are as of that commit.

Packages with no attention at all: Voice, World, WorldViewer, WorldViewer.Avalonia, API, Meta. Cli and Engine hold
settings text and capability checks only. Most consumer attention already goes through `AttentionLayer` or
`JointAttention`: about 9 files in LLM, 63 in Diffusion, 41 in Audio, 14 in ThreeD, 15 in Vision and one each in
Video and ModelAssets. Those counts come from the first pass and I did not keep the search that produced them, so
treat them as approximate.

## Plain attention on the host

"Plain" means `softmax(q · k · scale + terms a standard entry takes) · V`.

| Site | What it is | Status at `f9a0cedeb` |
|---|---|---|
| Diffusion `Models/TextEncoders/GptOssEncoder.cs` | Causal or windowed, grouped heads, per-head sink; single-threaded | Migrated in #1005 (merged). Real Lens generation on the RTX 4090, CUDA: 24% less total time; the image is 32.0 dB from the base, the size a rounding-only control gives (31.7 dB) |
| Diffusion `Models/Denoisers/DiTBlocks/WanAnimateFaceBlock.cs` | Frame-local cross-attention, 16 calls a step with CFG; single-threaded | Migrated in #1011 (merged). Real Wan Animate generation on the RTX 4090, CUDA, 5 frames at 480 by 480: 62% less request time; frames 25.2 dB from the base, where a rounding-only control is 19.9 dB from it |
| Audio `Models/IndexTts/IndexTtsPerceiver.cs` | Unmasked; single-threaded | Migrated in #1010 (merged). Real runs on the RTX 4090, CUDA: on IndexTTS 1.5 the output is 56.35 dB from the base and two runs are bit-identical; on IndexTTS 2.5 it is identical to the base's, with the new calls shown executing |
| Audio `Models/GptSoVits/SoVitsEncP.cs` `CrossAttention` | Unmasked, channels-first; single-threaded | Migrated in #1010 (merged). **No real run**: no reference clip with a transcript was on the RTX 4090 host. Unit tests only |
| Audio `Models/ControlFoley/ControlFoleyStyleEncoder.cs` | Unmasked self-attention over frames; single-threaded | Migrated in #1010 (merged). Real run on the RTX 4090, CUDA, with a reference clip: the new calls execute and the output is bit-identical to the base's; a video-only request does not reach this code |
| Audio `Models/FishAudio/DacOps.cs` `RunTransformer` | Causal, window of 128 keys (512 in the encoder), host RoPE; single-threaded | Open (read by me). On CUDA the causal F32 route narrows to F16, a larger precision change than TF32 inside a codec, and no Fish Audio weights are on either test machine. Looked at again after #1026: its window as an additive bias would be a `t × t` tensor a call, where the loop is linear in `t` for its fixed window of 128 or 512 keys, so the two-sided-window route of #1026 is not a fit at codec lengths |
| Audio `Models/BreezeTts/T5Gemma2TextEncoder.cs` `Encode` 87-112 | Two-sided window on some layers, grouped heads; single-threaded | Migrated in #1026 (merged): grouped heads go to the backend as they are (CUDA repeats K and V before its dense row), the two-sided window is an additive bias, and none is passed where the window covers the segment. Checked against the former loop on the CPU backend, CUDA and Vulkan, on the RTX 3060 and the RTX 4090; no real generation |
| Audio `Models/ControlFoley/ControlFoleySwinStage.cs` `WindowAttention` 153-201 | Swin: learned relative-position bias and the shift mask, both additive; single-threaded | Migrated in #1014 (merged): one batch item a window, with the relative-position bias and the shift mask as one additive bias. Real run on the RTX 4090, CUDA, with a reference clip: the new calls execute and the output is 28.2 dB from the base, where a rounding-only control is 31.6 dB from it |
| Audio `Models/IndexTts/IndexTtsConformerLayer.cs` `RelPosAttention` 100-157 | Positional terms indexed by the key alone, with no relative shift, so the score is `q · (k + p)` plus `u · k + v · p`, a bias a head and key that no query row changes; single-threaded | Migrated in #1013 (merged). Real runs on the RTX 4090, CUDA: on IndexTTS 1.5 the output is 56.79 dB from the base, two runs are bit-identical and it is 24% faster; on IndexTTS 2.5 it is identical to the base's, with the emotion conformer's calls shown executing. Plain with an additive bias, confirmed by two further readings (an earlier pass had it as specialized) |
| Video `Tokenizers/CosmosDvModule.cs` `SpatialAttn` 120-147, `TemporalAttn` 168-195 | Single head; temporal is causal; single-threaded | Left on the host on purpose: the tokenizer is written for a deterministic F32 result |
| LLM `Decision/Clef/ClefJointHead.cs` `Attention` (from line 286) | `nn.MultiheadAttention` mirror; already `Parallel.For` | Open, low value (read by me): small and already multi-threaded |
| ThreeD `Models/Idol/Budget/BudgetRefineStage.cs` `Attend` | Block-diagonal over 2 or 4 children a parent; `Parallel.For` | Leave (read by me): tiny fixed groups, linear cost |
| Vision `Detection/Blocks/PsaAttention.cs` 111-166 | V is twice as wide as Q and K (`_keyDim` is half `_headDim`); single-threaded | Migrated in #1016 (merged) by zero-padding Q and K to the value width and passing the scale explicitly, which leaves every `q · k` unchanged. A call as the loop had it is refused: the flash lowering refuses a value wider than the query and key (`AttentionFlashLowering.cs`), and the legacy CPU kernel reads one width for all three. Real YOLO11n detection on the RTX 3060 (CPU backend, CUDA, Vulkan) on one image: the same eight detections as the base, at the printed precision |

## Specialized: not a standard call

These need terms no standard entry has, and are left alone.

| Site | Why |
|---|---|
| Audio `Vits/VitsTextEncoder.cs`, `Vits/VitsFftBlock.cs`, `GptSoVits/SoVitsEncP.cs` `AttnLayer` | Relative-position terms on keys and on values |
| Audio `Wav2Vec2Bert/Wav2Vec2BertConformerLayer.cs`, `Codecs/NeuCodec/NeuCodecEncoder.cs` | A relative key term `q · E[j - i]`; could become a bias only by first building it as a tensor |
| Audio `CosyVoice/UpsampleConformerEncoder.cs`, Diffusion `Models/Music/AceStepLyricEncoder.cs` | Transformer-XL position terms with a relative shift |
| Audio `ZipVoice/ZipformerAttentionPrimitives.cs` | Shifted position scores; produces weights that several value modules share |
| ThreeD `Models/Pixal3D/Pixal3DNaf.cs` `Attend` | Neighbourhood attention at sample points with queries at another resolution. Whether `IBackend.NeighborhoodAttention` can express it was not checked |

## Not attention

- LLM `Transformer/AttnRes/AttentionResidual.cs`: a learned mix over depth; its backend path already exists, and the
  host `MixReference` has no caller in `src/`.
- LLM `Decision/Clef/ClefJointHead.cs` 122-156: pooling with one query per question over ragged, tiny groups. This is
  a different function from the `Attention` method listed above.
- ThreeD `Models/UniRig/UniRigSkinModel.cs` `ScoreChunk`, Audio `Models/Whisper/WhisperAttentionProbe.cs`: attention
  weights only, with no value mix.
- Attentive-statistics pooling and layer-weight softmaxes (ECAPA encoders, Zonos, MERT, MiniMax Music, Wan S2V audio).

## Uses of the specialized backend entries

Video sparse attention (Video `MiniMaxH3Pipeline`, Diffusion `MiniMaxH3Transformer`), neighbourhood attention
(Diffusion `LtxVideo25NeighborhoodAttention3d`), sparse latent attention (LLM `HcaAttention`, `DeepSeekV41Attention`,
`DeepSeekV41DSparkAttention`), deformable attention (Vision `RtDetrDecoder`, GroundingDino
`MultiscaleDeformableAttention`). No consumer calls the token-major entry, the planner or the paged constructor
directly.

One site already on the device repeats a pattern that a migrated site now does in one call: Diffusion
`WanS2VAudioInjector` attended frame by frame, one call a frame, where the face adapter in #1011 uses one batch item a
frame. It now makes one call with a batch item a frame; on CUDA and on Vulkan the result is bit-identical to the
per-frame forward on the same backend (RTX 3060).

## What a migration changes

The first three migrations (#1005, #1010, #1011) move exact-F32 host loops to the shared layer. On the CPU backend and
on Vulkan the result stays F32. **On CUDA the products become TF32**, as for every other caller of that layer, inside
the contract bound of 2e-3. Each keeps the former loop as its test oracle and names this change. The later ones (#1013,
#1014, #1016) do the same and say so in their own text.

Real generations before and after, run on the RTX 4090 on 2026-11-21 and 22 (and on the RTX 3060 for #1016) and
recorded on each pull request with the expectations fixed beforehand:

- **#1005, GPT-OSS encoder (Lens), merged.** Each arm's two images are identical, the new rows execute, total time
  is 24% lower. The image is 32.0 dB from the base where I had required 40 dB beforehand, so that gate was missed. A
  control (the old loop with its sums reordered, which is the same arithmetic) is 31.7 dB from the base: the pipeline
  amplifies last-bit rounding that far, and no implementation that rounds differently could have met 40 dB. It merged on
  the corrected criterion, stated after the fact: no further from the base than the rounding-only control.
- **#1010 and #1013, merged.** See the table rows above; all of these runs were on CUDA, where the changed attention
  multiplies in TF32. IndexTTS 2.5 and the ControlFoley style encoder come out identical to the base, IndexTTS 1.5 is
  56 dB from it, GPT-SoVITS 53.7 dB, each with the new calls shown executing. **IndexTTS 2.0 is 2,048 samples (93 ms)
  shorter with #1010**, where I had expected the same length; a rounding-only control of the old perceiver loop is
  2,560 samples shorter, so its sampled token sequence moves with any last-bit change in its conditioning. With #1013
  as well the length is the same as with #1010 and the audio is 25 dB from it; that step has no control of its own. One
  limit of method: a first run asked ControlFoley for a video-only request, which never reaches its reference encoders,
  so it showed nothing and was repeated with a reference clip.
- **#1014, ControlFoley Swin stage, merged.** With a reference clip the new calls execute and the output is 28.2 dB
  from the base (the arm without this change is bit-identical to the base); a rounding-only control is 31.6 dB from
  the base. No bound had been set beforehand.
- **#1011, Wan Animate face adapter, merged.** The first Wan Animate checkpoint was downloaded to the 4090 host for
  this (a repack, `Kijai/WanVideo_comfy_fp8_scaled`). 5 frames at 480 by 480: each arm repeats itself, the face
  adapter's calls execute, the timed request falls from 97.0 s to 36.9 s, and the frames are 25.2 dB from the base. A
  rounding-only control is 19.9 dB from the base, further than the change itself. That control fell outside both
  readings I had written for it beforehand (near 25 dB, or 40 dB and above); it lies beyond the first on the same side.
- **#1016, PSA block, merged.** YOLO11n was converted from the publisher's checkpoint for this. On one street image,
  the RTX 3060 gives the same eight detections before and after on the CPU backend, CUDA and Vulkan, at the precision
  the CLI prints (scores to two decimals, boxes to whole pixels). Not run on the RTX 4090.

What the four controls have in common: on each of these pipelines, changing only the order of a few sums in the old
host loop moves the final image or audio by 20 to 32 dB, or changes its length. So a comparison of final outputs
cannot tell a correct change to a conditioning encoder from a wrong one at that level. For such changes the bound
belongs on the encoder's output, set against a rounding floor measured first.

## Timing seen so far (smoke only, RTX 3060 box, attention alone)

Each figure is the best of two to five runs in one process, not a campaign. The former loop does not scale exactly
with the square of the frame count here, which a single process on a shared box does not explain further.

| Site | Size | Former loop | Now, CPU | Now, CUDA |
|---|---|---|---|---|
| Wan Animate face adapter, one call | 21 frames of 1,560 tokens, 5 motion tokens | 2,050 ms | 571 ms | 108 ms |
| ControlFoley style encoder, one layer | 750 frames | 748 ms | 26 ms | 2.7 ms |
| ControlFoley style encoder, one layer | 1,500 frames | 3,994 ms | 92 ms | 8.1 ms |

The IndexTTS perceiver and the GPT-SoVITS MRTE were a few milliseconds at their model sizes and stay there.

## Package tests renewed after the K-series (plan item 17)

The K-series is plan items K01 to K09 of `proposed-queue.md`: the kernel work (INT8 attention, decode with lengths,
merges, Sol-Attn) that landed between the first audit and this update.

Whole test projects at `main` `f28b0f384` on the RTX 3060 box, every backend the box has, on 2026-11-21, in two
sittings (Audio and Diffusion, then the other four). That commit has the merged #1026 and #1028, so the Audio and
Diffusion totals include `T5Gemma2AttentionTests` and `WanS2VAudioInjectorTests`.

| Project | Listed | Ran | Passed | Failed | Skipped | Passed after printing `SKIPPED` |
|---|---|---|---|---|---|---|
| `HartsyInference.Audio.Tests` | 137 | 137 | 135 | 0 | 2 | 0 |
| `HartsyInference.Diffusion.Tests` | 383 | 383 | 373 | 0 | 10 | 1 |
| `HartsyInference.Core.Tests` | 466 | 466 | 466 | 0 | 0 | 2 |
| `HartsyInference.Vision.Tests` | 69 | 69 | 65 | 0 | 4 | 0 |
| `HartsyInference.ThreeD.Tests` | 1,434 | 1,443 | 1,412 | 0 | 31 | 55 |
| `HartsyInference.LLM.Tests` (whole project, not only the GPU lane) | 749 | 749 | 715 | 0 | 34 | 2 |

Expectation written before the runs: no failures. It held, and every result file ends `Completed` with none aborted. A
first pass of Audio and Diffusion at `24ff78dbb`, before those two merges, had 132 and 374 listed with the same two and
ten skips and no failures; a review pointed out that it left the migrated code out, and their rows are the rerun.

How to read the columns:

- **Listed against ran.** A crashed test host prints a pass over a short count, so each run was compared with the
  project's `--list-tests` count. ThreeD ran nine more than listed, for a known reason: three theories in
  `MeshPostProcess/UvUnwrapperTests.cs` take four meshes from `MemberData`, the mesh type cannot be serialized, so
  discovery lists each theory once and the run expands it to four cases (3 x 3 = 9). The count to hold a ThreeD run to
  is therefore 1,443, not the listed 1,434.
- **Skipped** are cases the framework reports as not executed: real-weight checks whose files are not on this box. In
  Audio, word alignment; in Diffusion, Sana WM and one LTX Video 2 VAE check; in Vision, DINOv2 and BiRefNet; in ThreeD,
  HairDeck, Hunyuan Paint, Pixal3D, P3-SAM, UniRig, X-Part and one Trellis 2 check; in LLM, MOSS-VL and the Kimi
  tokenizer. They are unvalidated here, not passed.
- **Passed after printing `SKIPPED`** are cases the framework counts as passed although the test printed `SKIPPED` and
  returned early, so they validated nothing either: the two- and eight-GPU NCCL tests in Core and the two sharded
  Kimi tests in LLM (one GPU here), one T5 reference check in Diffusion, and 55 real-weight parity cases in ThreeD
  (Hunyuan 3D shape, Budget, Idol, paint bake, P3-SAM geometry and others). The count is of tests whose captured
  output holds the word; a test that returns early without printing it is not counted, and I did not search for
  those.

This is unit and parity coverage with synthetic weights; it is not a real generation, and it says nothing about the
RTX 4090.

The statuses in the first table are as of the commit in its header. Every migration it names has merged; the rows
still marked open or left on the host are the ones nobody has changed.
