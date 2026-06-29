using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Ideogram 4 (<c>ideogram-oss/ideogram4</c>, 9.3B, non-commercial license) text-to-image pipeline. Orchestrates Qwen3-VL-8B (13-layer tap) → two single-stream DiTs (conditional + unconditional, asymmetric CFG) → Flux.2 VAE. Ported from upstream <c>pipeline_ideogram4.py</c>.
///
/// Pipeline-level specifics:
/// <list type="bullet">
///   <item><b>Unified sequence</b> per prompt: <c>[text tokens][image tokens]</c>. Text features (Qwen 13-layer concat) sit at text positions; noise sits at image positions; an <c>image_indicator</c> embedding + 3D MRoPE keep them apart.</item>
///   <item><b>Asymmetric CFG</b>: the positive pass runs the conditional transformer over the full sequence and keeps only the image-token velocity; the negative pass runs the <i>unconditional</i> transformer over an image-only sequence with zeroed text features. Combined as <c>v = gw·pos + (1−gw)·neg</c> with a per-step guidance schedule (gw≈7 main, gw≈3 polish).</item>
///   <item><b>Logit-normal schedule</b> (<see cref="LogitNormalSchedule"/>), resolution-adjusted mean, plain Euler <c>z += v·(s−t)</c>.</item>
///   <item><b>Fixed-constant latent norm</b> (<see cref="Ideogram4LatentNorm"/>) applied to the packed token latent before the 2×2 unpatchify — NOT the Flux.2 VAE BatchNorm.</item>
/// </list>
///
/// <para><b>VRAM:</b> both 9.3B transformers must be resident during the loop (each step runs both), so this needs roughly 2× the DiT footprint plus the VAE. The Qwen encoder is freed before the loop. Realistically a multi-GPU / high-VRAM host; documented in the Phase 4 checklist.</para></summary>
public sealed unsafe class Ideogram4Pipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Ideogram4Transformer _conditional;
    private readonly Ideogram4Transformer _unconditional;
    private readonly VaeDecoder _vaeDecoder;
    private readonly Ideogram4Config _config;

    private const int LlmTokenIndicator = 3;
    private const int OutputImageIndicator = 2;
    private const int ImagePositionOffset = 65536;

    /// <summary>Creates an Ideogram 4 pipeline from pre-loaded components.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="textEncoder">Qwen3-VL-8B language tower (<see cref="LlamaStyleEncoderConfig.Qwen3_VL_8B"/>).</param>
    /// <param name="conditional">Conditional transformer (<c>transformer/</c> weights).</param>
    /// <param name="unconditional">Unconditional transformer (<c>unconditional_transformer/</c> weights).</param>
    /// <param name="vaeDecoder">Flux.2 VAE decoder (<c>VaeConfig.Flux2</c>).</param>
    /// <param name="config">Architecture config (pinned to the loaded transformers).</param>
    public Ideogram4Pipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        Ideogram4Transformer conditional, Ideogram4Transformer unconditional,
        VaeDecoder vaeDecoder, Ideogram4Config config)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _conditional = conditional;
        _unconditional = unconditional;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from chat-templated prompt token ids (the Qwen3 chat template must already be applied). The negative branch needs no tokens — Ideogram's CFG zeroes the text features.</summary>
    /// <param name="promptTokenIds">Tokenized, chat-templated prompt.</param>
    /// <param name="request">Width/Height/Seed (Steps and CfgScale come from <paramref name="preset"/>).</param>
    /// <param name="preset">Sampler preset (steps + guidance schedule + logit-normal mu/std). Defaults to <see cref="Ideogram4SamplerPreset.Default20"/>.</param>
    /// <param name="onProgress">Optional per-step callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        TextToImageRequest request,
        Ideogram4SamplerPreset? preset = null,
        Action<GenerationProgress>? onProgress = null,
        RegionalPlan? regionalPlan = null)
    {
        ThrowIfDisposed();
        preset ??= Ideogram4SamplerPreset.Default20;

        if (promptTokenIds.Length == 0)
            throw new ArgumentException("Prompt token ids must be non-empty.", nameof(promptTokenIds));
        if (promptTokenIds.Length > _config.MaxTextTokens)
            throw new ArgumentException($"Prompt has {promptTokenIds.Length} tokens, exceeds MaxTextTokens={_config.MaxTextTokens}.", nameof(promptTokenIds));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.Generic.Width;
        int height = request.Height ?? GenerationDefaults.Generic.Height;
        int patch = _config.PatchSize * _config.VaeScaleFactor; // 2 × 8 = 16
        if (width % patch != 0 || height % patch != 0)
            throw new ArgumentException($"Width and height must be divisible by {patch} for Ideogram 4.");

        int gridH = height / patch;
        int gridW = width / patch;
        int numImageTokens = gridH * gridW;
        int numText = promptTokenIds.Length;
        int seqLen = numText + numImageTokens;
        int latentDim = _config.InChannels;
        int steps = preset.NumSteps;

        Logs.Info($"Ideogram 4: generating {width}x{height}, preset={preset.Name} ({steps} steps), seed={seed}");
        Logs.Warning("Ideogram 4 runs TWO 9.3B transformers concurrently (asymmetric CFG) — expect very high VRAM use.");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode prompt: Qwen3-VL 13-layer concat → [1, numText, llmFeaturesDim] ──
        // Diagnostic: log the token ids so they can be compared 1:1 against a reference
        // transformers Qwen3-VL run (tests/python-reference/dump_qwen3vl_ideogram4_encoder.py).
        // A token-id mismatch ⇒ tokenizer/chat-template bug; a match with differing encoder
        // features ⇒ encoder forward/weight-load bug.
        LogTokenIds(promptTokenIds);
        Backend.PreloadWeights(_textEncoder.EnumerateWeights());
        // interleavedLayout: true — Ideogram 4 concatenates the 13 taps HIDDEN-MAJOR (c = h*13 + tap), matching
        // upstream pipeline_ideogram4.py (permute(stack,(1,2,3,0)).reshape) and ComfyUI (permute(0,2,3,1)). The
        // llm_cond_norm/llm_cond_proj weights are trained on this order; tap-major scrambles every input channel.
        Tensor textFeatures = _textEncoder.EncodeMultiLayer(Backend, [promptTokenIds], Ideogram4Config.QwenActivationLayersHf, interleavedLayout: true);
        if ((int)textFeatures.Shape[2] != _config.LlmFeaturesDim)
            throw new InvalidOperationException($"Encoder produced {textFeatures.Shape[2]}-dim features, expected {_config.LlmFeaturesDim} (13 × 4096). Check the tap-layer indices / encoder preset.");
        Backend.Sync();
        LogEncoderStats(textFeatures);
        // When IDEOGRAM4_DEBUG_DIR is set, dump the raw encoder output so it can be diffed
        // element-wise against the transformers Qwen3-VL reference (same tokens).
        HartsyInference.Diffusion.Models.Denoisers.Ideogram4DebugDump.Dump("textFeatures", textFeatures);
        Backend.FreeWeights(_textEncoder.EnumerateWeights());
        Logs.Info($"Prompt encoded in {sw.ElapsedMilliseconds}ms");

        // ── 2. Build the unified-sequence conditioning tensors ──
        // Regional conditioning appends each region's encoded features after the base text tokens
        // so image tokens can be biased toward their region's prompt (RegionalAttentionBias). With
        // no plan this collapses exactly to the base single-prompt path.
        bool hasRegions = regionalPlan is not null && regionalPlan.Regions.Count > 0;
        int effNumText = numText;
        List<(int Start, int End)>? regionRanges = null;
        List<float[]>? regionGridMasks = null;
        float[]? regionWeights = null;
        if (hasRegions)
        {
            (effNumText, regionRanges, regionGridMasks) = BuildRegionLayout(regionalPlan!, numText, numImageTokens, gridH, gridW);
            regionWeights = new float[regionalPlan!.Regions.Count];
        }
        int effSeqLen = effNumText + numImageTokens;
        Tensor llmFull = BuildLlmFull(textFeatures, numText, effSeqLen, _config.LlmFeaturesDim, hasRegions ? regionalPlan : null);
        textFeatures.Dispose();
        Tensor posIds = BuildPositionIds(effNumText, gridH, gridW);           // [1, L, 3]
        int[] indicator = BuildIndicator(effNumText, numImageTokens);         // [L]
        Tensor posIdsImageOnly = SliceImagePositions(posIds, effNumText, numImageTokens);
        int[] indicatorImageOnly = new int[numImageTokens];
        Array.Fill(indicatorImageOnly, OutputImageIndicator);
        Tensor negLlm = new Tensor(new TensorShape(1, numImageTokens, _config.LlmFeaturesDim), DType.F32); // zeros

        // ── 3. Initial noise (image-token format) [1, nImg, 128] ──
        Tensor z = SeedGenerator.CreateNoise(new TensorShape(1, numImageTokens, latentDim), seed);

        // ── 4. Schedule ──
        LogitNormalSchedule schedule = LogitNormalSchedule.ForResolution(height, width, preset.Mu, preset.Std);
        float[] grid = LogitNormalSchedule.MakeStepIntervals(steps);

        // ── 5. Denoise loop (both transformers resident) ──
        Backend.PreloadWeights(_conditional.EnumerateWeights());
        Backend.PreloadWeights(_unconditional.EnumerateWeights());
        LogVram("after DiT weight preload");

        for (int i = steps - 1; i >= 0; i--)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            Backend.ResetD2hSyncCount();
            HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.Ideogram4Profile.Reset();
            float tVal = schedule.Map(grid[i + 1]);
            float sVal = schedule.Map(grid[i]);
            float delta = sVal - tVal;
            float gw = preset.GuidanceSchedule[i];

            // Positive pass: full sequence, keep image-token velocity. Regional bias (when present)
            // steers each image token toward its region's appended conditioning columns.
            Tensor posX = BuildConditionalLatent(z, effNumText, numImageTokens, latentDim);
            Tensor? regionBias = null;
            if (hasRegions)
            {
                regionalPlan!.ResolveStep(steps - 1 - i, regionWeights!);
                regionBias = RegionalAttentionBias.Build(effSeqLen, effNumText, numImageTokens, regionRanges!, regionGridMasks!, regionWeights!);
            }
            Tensor posOut = _conditional.Forward(Backend, llmFull, posX, tVal, posIds, indicator, regionBias);
            regionBias?.Dispose();
            posX.Dispose();
            Tensor posV = SliceImageVelocity(posOut, effNumText, numImageTokens, latentDim);
            posOut.Dispose();

            // Negative pass: image-only sequence, zeroed text, unconditional transformer.
            Tensor negV = _unconditional.Forward(Backend, negLlm, z, tVal, posIdsImageOnly, indicatorImageOnly, null);

            // Conditioning-effect probe (profile-gated): relative RMS difference between the conditional
            // and unconditional velocities. ~0 means the prompt has no effect (encoder/feature bug);
            // a substantial value means text conditioning is live. Reads DataPointer (2 D2H syncs), so
            // it only runs under HARTSY_DIT_PROFILE=1 to keep real runs fully resident.
            double cfgDelta = -1;
            if (HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.Ideogram4Profile.Enabled)
            {
                float* pp = (float*)posV.DataPointer;
                float* np = (float*)negV.DataPointer;
                long nv = posV.ElementCount;
                double sd = 0, sn = 0;
                for (long e = 0; e < nv; e++) { double d = pp[e] - np[e]; sd += d * d; sn += (double)np[e] * np[e]; }
                cfgDelta = Math.Sqrt(sd / nv) / (Math.Sqrt(sn / nv) + 1e-9);
            }

            // v = gw·pos + (1−gw)·neg ; z = z + v·delta — in-place on the GPU-resident latent.
            Backend.CfgEulerStep(z, posV, negV, gw, delta);
            posV.Dispose();
            negV.Dispose();

            stepSw.Stop();
            (long freeB, long totalB) = Backend.GetVramInfo();
            long syncs = Backend.GetD2hSyncCount();
            string vram = totalB > 0 ? $"{(totalB - freeB) / 1073741824.0:F1}/{totalB / 1073741824.0:F1} GiB used" : "n/a";
            string profile = HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.Ideogram4Profile.Enabled
                ? $" | attn {HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.Ideogram4Profile.AttentionMs:F0}ms mlp {HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.Ideogram4Profile.MlpMs:F0}ms (both passes) | cfgΔ {cfgDelta:F3}"
                : "";
            Logs.Info($"[Ideogram4] step {steps - i}/{steps} t={tVal:F3} gw={gw:F1} {stepSw.ElapsedMilliseconds}ms | VRAM {vram} | D2H syncs {syncs}{profile}");
            // Tag the latent family for live-preview decoders (Flux.2 VAE, shared with Lens). The working
            // latent is token-packed [1, nImg, 128], so no per-step Latent snapshot is provided.
            onProgress?.Invoke(new GenerationProgress(steps - i, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                LatentArch = LatentArchitecture.Flux2,
            });
        }

        llmFull.Dispose();
        posIds.Dispose();
        posIdsImageOnly.Dispose();
        negLlm.Dispose();

        // ── 6. Free DiT weights before VAE decode ──
        Backend.Sync();
        Backend.FreeWeights(_conditional.EnumerateWeights());
        Backend.FreeWeights(_unconditional.EnumerateWeights());

        // ── 7. Latent un-normalize (fixed constants) + unpatchify → [1, 32, H/8, W/8] ──
        ApplyLatentNorm(z);
        Tensor vaeIn = Unpatchify(z, gridH, gridW, _config.PatchSize);
        z.Dispose();

        // ── 8. VAE decode ──
        Logs.Verbose("Decoding latents (Flux.2 VAE, tiled)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, vaeIn);
        Backend.Sync();
        vaeSw.Stop();
        Logs.Info($"[Ideogram4] VAE decode in {vaeSw.ElapsedMilliseconds}ms");
        vaeIn.Dispose();

        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Ideogram 4 generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgb, width, height, seed);
    }

    /// <summary>Places encoded base text features <c>[1, numText, D]</c> (and any regional features) into a full <c>[1, seqLen, D]</c> tensor: base text at the front, region features immediately after, zeros at image positions.</summary>
    private static Tensor BuildLlmFull(Tensor textFeatures, int numText, int seqLen, int dim, RegionalPlan? plan)
    {
        Tensor full = new Tensor(new TensorShape(1, seqLen, dim), DType.F32);
        float* dst = (float*)full.DataPointer;
        new Span<float>(dst, checked((int)((long)seqLen * dim))).Clear();
        float* src = (float*)textFeatures.DataPointer;
        long baseBytes = (long)numText * dim * sizeof(float);
        Buffer.MemoryCopy(src, dst, baseBytes, baseBytes);
        if (plan is not null)
        {
            long offsetTokens = numText;
            foreach (RegionConditioning region in plan.Regions)
            {
                int len = (int)region.Cond.Shape[1];
                float* regionSrc = (float*)region.Cond.DataPointer;
                long regionBytes = (long)len * dim * sizeof(float);
                Buffer.MemoryCopy(regionSrc, dst + offsetTokens * dim, regionBytes, regionBytes);
                offsetTokens += len;
            }
        }
        return full;
    }

    /// <summary>Computes the regional layout for the unified sequence: the extended text-token count, each region's conditioning column range, and each region's image-grid mask (row-major <c>[numImg]</c>).</summary>
    private (int ExtNumText, List<(int Start, int End)> Ranges, List<float[]> GridMasks) BuildRegionLayout(
        RegionalPlan plan, int numText, int numImg, int gridH, int gridW)
    {
        List<(int Start, int End)> ranges = new List<(int Start, int End)>(plan.Regions.Count);
        List<float[]> masks = new List<float[]>(plan.Regions.Count);
        int cursor = numText;
        foreach (RegionConditioning region in plan.Regions)
        {
            if ((int)region.Cond.Shape[2] != _config.LlmFeaturesDim)
            {
                throw new InvalidOperationException($"Region conditioning dim {region.Cond.Shape[2]} != LlmFeaturesDim {_config.LlmFeaturesDim}.");
            }
            int len = (int)region.Cond.Shape[1];
            ranges.Add((cursor, cursor + len));
            cursor += len;
            Tensor latentMask = region.Mask.ToLatentMask(gridH, gridW);
            float[] grid = latentMask.AsReadOnlySpan<float>().ToArray();
            latentMask.Dispose();
            if (grid.Length != numImg)
            {
                throw new InvalidOperationException($"Region grid mask length {grid.Length} != numImg {numImg}.");
            }
            masks.Add(grid);
        }
        return (cursor, ranges, masks);
    }

    /// <summary>Builds <c>[1, L, 3]</c> MRoPE positions: text tokens <c>(i, i, i)</c>; image tokens <c>(0, row, col) + IMAGE_POSITION_OFFSET</c>.</summary>
    private static Tensor BuildPositionIds(int numText, int gridH, int gridW)
    {
        int numImg = gridH * gridW;
        int seqLen = numText + numImg;
        Tensor pos = new Tensor(new TensorShape(1, seqLen, 3), DType.F32);
        float* p = (float*)pos.DataPointer;
        for (int t = 0; t < numText; t++)
        {
            long off = (long)t * 3;
            p[off + 0] = t;
            p[off + 1] = t;
            p[off + 2] = t;
        }
        for (int idx = 0; idx < numImg; idx++)
        {
            int row = idx / gridW;
            int col = idx % gridW;
            long off = (long)(numText + idx) * 3;
            p[off + 0] = ImagePositionOffset;
            p[off + 1] = ImagePositionOffset + row;
            p[off + 2] = ImagePositionOffset + col;
        }
        return pos;
    }

    /// <summary>Slices the image-token rows <c>[numText..L)</c> out of a <c>[1, L, 3]</c> position tensor into <c>[1, numImg, 3]</c>.</summary>
    private static Tensor SliceImagePositions(Tensor posIds, int numText, int numImg)
    {
        Tensor outP = new Tensor(new TensorShape(1, numImg, 3), DType.F32);
        float* src = (float*)posIds.DataPointer;
        float* dst = (float*)outP.DataPointer;
        long bytes = (long)numImg * 3 * sizeof(float);
        Buffer.MemoryCopy(src + (long)numText * 3, dst, bytes, bytes);
        return outP;
    }

    private static int[] BuildIndicator(int numText, int numImg)
    {
        int[] ind = new int[numText + numImg];
        for (int t = 0; t < numText; t++) ind[t] = LlmTokenIndicator;
        for (int i = 0; i < numImg; i++) ind[numText + i] = OutputImageIndicator;
        return ind;
    }

    /// <summary>Builds the conditional-pass latent <c>[1, L, 128]</c> = <c>[zeros(numText) ; z]</c>.</summary>
    /// <summary>Builds the conditional latent <c>[1, numText+numImg, dim]</c>: text rows zeroed (masked out
    /// downstream), image rows = the current latent <paramref name="z"/>. GPU-resident scatter.</summary>
    private Tensor BuildConditionalLatent(Tensor z, int numText, int numImg, int dim)
    {
        int seqLen = numText + numImg;
        Tensor x = new Tensor(new TensorShape(1, seqLen, dim), DType.F32);
        Backend.ScatterRowsAfter(x, z, numText);
        return x;
    }

    /// <summary>Slices image-token velocity rows <c>[numText..L)</c> out of <c>[1, L, 128]</c> into <c>[1, numImg, 128]</c>. GPU-resident.</summary>
    private Tensor SliceImageVelocity(Tensor full, int numText, int numImg, int dim)
    {
        Tensor outV = new Tensor(new TensorShape(1, numImg, dim), DType.F32);
        Backend.SliceRows(outV, full, numText);
        return outV;
    }

    /// <summary>Diagnostic: logs the prompt token ids (count + a head/tail window) so they can be compared
    /// 1:1 against a reference transformers Qwen3-VL tokenization of the same prompt + chat template.</summary>
    private static void LogTokenIds(int[] ids)
    {
        int head = Math.Min(24, ids.Length);
        string headStr = string.Join(",", ids[..head]);
        string tailStr = ids.Length > 24 ? " … " + string.Join(",", ids[^Math.Min(8, ids.Length)..]) : "";
        Logs.Info($"[Ideogram4] promptTokens count={ids.Length} ids=[{headStr}{tailStr}]");
    }

    /// <summary>Diagnostic: logs summary stats of the Qwen3-VL encoder output. The text features are already
    /// host-synced (BuildLlmFull reads them on CPU), so this is nearly free. Run two DIFFERENT prompts and compare:
    /// identical stats ⇒ the encoder is producing prompt-independent features (weight-load / tokenization bug);
    /// differing stats but an image that still ignores the prompt ⇒ the bug is downstream in the DiT conditioning
    /// or MRoPE. NaN/inf &gt; 0 ⇒ the encoder forward is numerically broken.</summary>
    private static void LogEncoderStats(Tensor textFeatures)
    {
        float* tf = (float*)textFeatures.DataPointer;
        long n = textFeatures.ElementCount;
        if (n == 0) return;
        double sum = 0, sumsq = 0;
        float mn = float.MaxValue, mx = float.MinValue;
        long badCount = 0;
        for (long e = 0; e < n; e++)
        {
            float v = tf[e];
            if (float.IsNaN(v) || float.IsInfinity(v)) { badCount++; continue; }
            sum += v; sumsq += (double)v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        long good = n - badCount;
        double mean = good > 0 ? sum / good : 0;
        double std = good > 0 ? Math.Sqrt(Math.Max(0, sumsq / good - mean * mean)) : 0;
        Logs.Info($"[Ideogram4] textFeatures [{textFeatures.Shape[1]}x{textFeatures.Shape[2]}] " +
            $"mean={mean:F5} std={std:F5} min={mn:F4} max={mx:F4} nan/inf={badCount} " +
            $"first=[{tf[0]:F4}, {tf[1]:F4}, {tf[2]:F4}, {tf[3]:F4}]");
    }

    /// <summary>Logs current device VRAM usage at an Info-level checkpoint (no-op detail on the CPU backend).</summary>
    private void LogVram(string stage)
    {
        (long freeB, long totalB) = Backend.GetVramInfo();
        if (totalB > 0)
            Logs.Info($"[Ideogram4] VRAM {stage}: {(totalB - freeB) / 1073741824.0:F1}/{totalB / 1073741824.0:F1} GiB used ({freeB / 1073741824.0:F1} free)");
    }

    /// <summary>Applies the fixed per-channel latent norm in-place: <c>z[...,c] = z[...,c]·Scale[c] + Shift[c]</c> on a <c>[1, nImg, 128]</c> packed latent.</summary>
    private void ApplyLatentNorm(Tensor z)
    {
        int channels = (int)z.Shape[2];
        if (channels != Ideogram4LatentNorm.Channels)
            throw new InvalidOperationException($"Latent channels {channels} != {Ideogram4LatentNorm.Channels}.");
        long tokens = z.Shape[0] * z.Shape[1];
        float* zp = (float*)z.DataPointer;
        float[] scale = Ideogram4LatentNorm.Scale;
        float[] shift = Ideogram4LatentNorm.Shift;
        for (long tok = 0; tok < tokens; tok++)
        {
            long baseOff = tok * channels;
            for (int c = 0; c < channels; c++)
                zp[baseOff + c] = zp[baseOff + c] * scale[c] + shift[c];
        }
    }

    /// <summary>Unpatchifies the packed token latent <c>[1, nImg, patch²·aeC]</c> → <c>[1, aeC, gridH·patch, gridW·patch]</c> (upstream <c>view(B,gh,gw,p,p,ae).permute(0,5,1,3,2,4)</c>). For packed feature <c>f</c>: <c>ae = f % aeC, p2 = (f/aeC) % patch, p1 = f / (aeC·patch)</c>.</summary>
    private static Tensor Unpatchify(Tensor z, int gridH, int gridW, int patch)
    {
        int packedC = (int)z.Shape[2];
        int aeC = packedC / (patch * patch);
        int outH = gridH * patch;
        int outW = gridW * patch;
        Tensor outT = new Tensor(new TensorShape(1, aeC, outH, outW), DType.F32);
        float* src = (float*)z.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int gh = 0; gh < gridH; gh++)
        {
            for (int gw = 0; gw < gridW; gw++)
            {
                long token = (long)gh * gridW + gw;
                long srcBase = token * packedC;
                for (int p1 = 0; p1 < patch; p1++)
                {
                    for (int p2 = 0; p2 < patch; p2++)
                    {
                        int dstH = gh * patch + p1;
                        int dstW = gw * patch + p2;
                        for (int ae = 0; ae < aeC; ae++)
                        {
                            int f = p1 * (aeC * patch) + p2 * aeC + ae;
                            long dstOff = ((long)ae * outH + dstH) * outW + dstW;
                            dst[dstOff] = src[srcBase + f];
                        }
                    }
                }
            }
        }
        return outT;
    }
}
