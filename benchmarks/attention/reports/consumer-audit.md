# Consumer audit: attention computed outside the backend entry points

Plan items 14 to 17 ask for every package to be audited for attention that bypasses the backend entry points
(`IBackend.ScaledDotProductAttention`, `IBackend.FlashAttention`, the planner, or the shared
`Core/Layers/Attention/AttentionLayer`). This is that audit at `main` `f9a0cedeb`. It is a source audit: it says what
the code does, not what any model produces.

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
| Diffusion `Models/TextEncoders/GptOssEncoder.cs` | Causal or windowed, grouped heads, per-head sink; single-threaded | Migrated in draft #1005 (read by me) |
| Diffusion `Models/Denoisers/DiTBlocks/WanAnimateFaceBlock.cs` | Frame-local cross-attention, 16 calls a step with CFG; single-threaded | Migrated in draft #1011 (read by me) |
| Audio `Models/IndexTts/IndexTtsPerceiver.cs` | Unmasked; single-threaded | Migrated in draft #1010 (read by me) |
| Audio `Models/GptSoVits/SoVitsEncP.cs` `CrossAttention` | Unmasked, channels-first; single-threaded | Migrated in draft #1010 (read by me) |
| Audio `Models/ControlFoley/ControlFoleyStyleEncoder.cs` | Unmasked self-attention over frames; single-threaded | Migrated in draft #1010 (read by me) |
| Audio `Models/FishAudio/DacOps.cs` `RunTransformer` | Causal, window of 128 keys (512 in the encoder), host RoPE; single-threaded | Open (read by me). On CUDA the causal F32 route narrows to F16, a larger precision change than TF32 inside a codec, and no Fish Audio weights are on either test machine |
| Audio `Models/BreezeTts/T5Gemma2TextEncoder.cs` `Encode` 87-112 | Two-sided window on some layers, grouped heads; single-threaded | Migrated to the shared layer: grouped heads native, the two-sided window as an additive bias and none where the window covers the segment. CPU backend verified against the former loop; no GPU run yet |
| Audio `Models/ControlFoley/ControlFoleySwinStage.cs` `WindowAttention` 153-201 | Swin: learned relative-position bias and the shift mask, both additive; single-threaded | Open. Fits an additive-bias call after the window gather |
| Audio `Models/IndexTts/IndexTtsConformerLayer.cs` `RelPosAttention` 100-157 | Positional terms indexed by the key alone, with no relative shift, so the score is `q · (k + p)` plus `u · k + v · p`, a bias a head and key that no query row changes; single-threaded | Plain with an additive bias, confirmed by two further readings (an earlier pass had it as specialized). A migration is prepared on a branch and passes on the CPU backend; its GPU run and draft are pending |
| Video `Tokenizers/CosmosDvModule.cs` `SpatialAttn` 120-147, `TemporalAttn` 168-195 | Single head; temporal is causal; single-threaded | Left on the host on purpose: the tokenizer is written for a deterministic F32 result |
| LLM `Decision/Clef/ClefJointHead.cs` `Attention` (from line 286) | `nn.MultiheadAttention` mirror; already `Parallel.For` | Open, low value (read by me): small and already multi-threaded |
| ThreeD `Models/Idol/Budget/BudgetRefineStage.cs` `Attend` | Block-diagonal over 2 or 4 children a parent; `Parallel.For` | Leave (read by me): tiny fixed groups, linear cost |
| Vision `Detection/Blocks/PsaAttention.cs` 111-166 | V is twice as wide as Q and K (`_keyDim` is half `_headDim`); single-threaded | Open. A call as it stands is refused: the flash lowering refuses a value wider than the query and key (`AttentionFlashLowering.cs`), and the legacy CPU kernel reads one width for all three. The likely route is to zero-pad Q and K to the value width and pass the scale explicitly, which leaves every `q · k` unchanged |

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
`WanS2VAudioInjector` attends frame by frame, one call a frame, where the face adapter in #1011 uses one batch item a
frame.

## What a migration changes

The three drafts so far move exact-F32 host loops to the shared layer. On the CPU backend and on Vulkan the result
stays F32. **On CUDA the products become TF32**, as for every other caller of that layer, inside the contract bound
of 2e-3. Each draft keeps the former loop as its test oracle and names this change.

None of the three has a real generation before and after: the weights are not on the RTX 3060 box. Weights for the
three audio models are on the RTX 4090 host (by file name, not hash). For the face adapter there is nothing to run on
either machine: the 4090 host holds Wan Animate 2 checkpoints, and `WanAnimate2Transformer` does not use
`WanAnimateFaceBlock`; the first Wan Animate, which does, has no weights there.

## Timing seen so far (smoke only, RTX 3060 box, attention alone)

Each figure is the best of two to five runs in one process, not a campaign. The former loop does not scale exactly
with the square of the frame count here, which a single process on a shared box does not explain further.

| Site | Size | Former loop | Now, CPU | Now, CUDA |
|---|---|---|---|---|
| Wan Animate face adapter, one call | 21 frames of 1,560 tokens, 5 motion tokens | 2,050 ms | 571 ms | 108 ms |
| ControlFoley style encoder, one layer | 750 frames | 748 ms | 26 ms | 2.7 ms |
| ControlFoley style encoder, one layer | 1,500 frames | 3,994 ms | 92 ms | 8.1 ms |

The IndexTTS perceiver and the GPT-SoVITS MRTE were a few milliseconds at their model sizes and stay there.

The statuses in the first table are as of the commit in its header. When a draft named there merges or changes,
this file needs the same update.
