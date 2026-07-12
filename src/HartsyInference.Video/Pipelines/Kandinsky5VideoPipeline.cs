using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>Kandinsky 5.0 T2V/I2V pipeline (kandinskylab/Kandinsky-5.0-T2V-Lite-*-Diffusers). Reuses the
/// <see cref="Kandinsky5Transformer"/> built for T2I via its video forward (33-channel visual-cond
/// packing, RoPE scale factors) and decodes with the new <see cref="HunyuanVideoVaeDecoder"/>.
///
/// <para>Like the T2I <see cref="Kandinsky5Pipeline"/>, the dual text encoder stack (Qwen2.5-VL +
/// CLIP-L) is supplied as <b>pre-computed embeddings</b>. CFG is the standard dual full pass
/// (<c>v = v_u + g·(v_c − v_u)</c>); scheduler is <c>FlowMatchEulerDiscreteScheduler(shift=5.0)</c>
/// (the native single-file release reportedly uses 10.0 — constructor parameter). Defaults per the
/// reference: 50 steps, CFG 5.0 (1.0 for the nocfg/distilled checkpoints), 121 frames @ 24 fps.
/// <b>Validation-gated:</b> numerics vs the reference checkpoint, dense-vs-NABLA attention deviation
/// for &gt;121-frame clips, and untiled VAE decode memory at full resolution.</para></summary>
public sealed unsafe class Kandinsky5VideoPipeline : DiffusionPipelineBase
{
    private readonly Kandinsky5Transformer _transformer;
    private readonly HunyuanVideoVaeDecoder _vae;
    private readonly HunyuanVideoVaeEncoder? _encoder;
    private readonly Kandinsky5Config _config;
    private readonly float _schedulerShift;

    /// <summary><paramref name="encoder"/> is optional and only needed for RGB-input I2V
    /// (<see cref="EncodeFirstFrame"/>); it loads from the same <c>vae/</c> shard as the decoder.</summary>
    public Kandinsky5VideoPipeline(IBackend backend, Kandinsky5Transformer transformer,
        HunyuanVideoVaeDecoder vae, Kandinsky5Config config,
        HunyuanVideoVaeEncoder? encoder = null, float schedulerShift = 5.0f)
        : base(backend)
    {
        if (!config.VisualCond)
            throw new ArgumentException(
                "Kandinsky5VideoPipeline requires a visual_cond config (Kandinsky5Config.VideoLite2B/VideoPro19B).",
                nameof(config));
        _transformer = transformer;
        _vae = vae;
        _config = config;
        _encoder = encoder;
        _schedulerShift = schedulerShift;
    }

    /// <summary>Resolution-dependent RoPE scale factor per the diffusers pipeline: <c>(1, 2, 2)</c> when
    /// both dimensions are 480p-class (480–854 px), else <c>(1, 3.16, 3.16)</c>.</summary>
    public static (float T, float H, float W) GetRopeScaleFactor(int height, int width)
    {
        static bool Between480p(int x) => x is >= 480 and <= 854;
        return Between480p(height) && Between480p(width) ? (1f, 2f, 2f) : (1f, 3.16f, 3.16f);
    }

    /// <summary>Encodes an interleaved-RGB24 conditioning frame to the model-space first-frame latent
    /// (<c>[1, 16, 1, H/8, W/8]</c>, scaling factor applied) for the I2V path. Caller disposes.</summary>
    public Tensor EncodeFirstFrame(ReadOnlySpan<byte> rgb24, int width, int height)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException(
                "RGB-input I2V needs a HunyuanVideoVaeEncoder — construct the pipeline with one (it loads from the same VAE weights).");
        Tensor raw = _encoder.EncodeRgbFrame(Backend, rgb24, width, height);
        Tensor scaled = new Tensor(raw.Shape, DType.F32);
        Backend.Scale(scaled, raw, _vae.Config.ScalingFactor);
        raw.Dispose();
        return scaled;
    }

    /// <summary>Generates a clip from pre-computed text encoder outputs. Returns one interleaved-RGB
    /// <c>byte[]</c> per frame. <paramref name="firstFrameLatent"/> (a model-space
    /// <c>[1, 16, 1, H/8, W/8]</c> latent from <see cref="EncodeFirstFrame"/>) switches to the I2V path:
    /// latent frame 0 is pinned to the image, the cond channels/mask carry it, only frames 1.. denoise,
    /// and the reference "mesh artifact" first-frame normalization runs after the loop.</summary>
    /// <param name="qwenEmbeds">Qwen2.5-VL sequence embeddings <c>[B, S_t, in_text_dim]</c>.</param>
    /// <param name="clipPooled">CLIP-L pooled embeddings <c>[B, in_text_dim2]</c>.</param>
    /// <param name="negQwenEmbeds">Negative-prompt Qwen embeddings (required if cfg &gt; 1).</param>
    /// <param name="negClipPooled">Negative-prompt CLIP pooled (required if cfg &gt; 1).</param>
    public (byte[][] frames, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor qwenEmbeds, Tensor clipPooled,
        Tensor? negQwenEmbeds, Tensor? negClipPooled,
        TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null,
        Tensor? firstFrameLatent = null)
    {
        Tensor latent = RunDenoise(qwenEmbeds, clipPooled, negQwenEmbeds, negClipPooled,
            request, numFrames, onProgress, firstFrameLatent, out int seed);

        // ── VAE decode: model space → raw VAE space, then HunyuanVideo causal decode. Tiled — the
        // full-res 3D decode peaks >24 GB (same VAE as HunyuanVideo); tiling bounds it to a per-tile
        // working set. Weights preloaded/freed around the stage per the shared pipeline convention. ──
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor decodeIn = new Tensor(latent.Shape, DType.F32);
        Backend.Scale(decodeIn, latent, 1.0f / _vae.Config.ScalingFactor);
        latent.Dispose();

        Tensor rgb;
        Backend.PreloadWeights(_vae.EnumerateWeights());
        try { rgb = _vae.DecodeTiled(Backend, decodeIn); }
        finally { decodeIn.Dispose(); }
        Backend.Sync();
        Backend.FreeWeights(_vae.EnumerateWeights());
        vaeSw.Stop();
        Logs.Verbose($"Kandinsky5-Video: VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        int frameCount = (int)rgb.Shape[2];
        byte[][] frames = new byte[frameCount][];
        for (int i = 0; i < frameCount; i++)
            frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        rgb.Dispose();

        Logs.Info($"Kandinsky5-Video complete ({frames.Length} frames, seed={seed})");
        return (frames, request.Width ?? 768, request.Height ?? 512, seed);
    }

    /// <summary>Runs the flow-match denoise loop and returns the model-space latent <c>[1, 16, T_lat, H_lat, W_lat]</c>.</summary>
    private Tensor RunDenoise(Tensor qwenEmbeds, Tensor clipPooled, Tensor? negQwenEmbeds, Tensor? negClipPooled,
        TextToImageRequest request, int numFrames, Action<GenerationProgress>? onProgress,
        Tensor? firstFrameLatent, out int seed)
    {
        ThrowIfDisposed();
        seed = request.Seed ?? SeedGenerator.RandomSeed();

        int width = request.Width ?? 768, height = request.Height ?? 512;
        int sp = _vae.Config.SpatialCompression;                       // 8
        int tp = _vae.Config.TemporalCompression;                      // 4
        (int _, int pH, int pW) = _config.PatchSize;
        if (width % (sp * pW) != 0 || height % (sp * pH) != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp * pW} for Kandinsky 5 video.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int tLat = (numFrames - 1) / tp + 1;
        int hLat = height / sp, wLat = width / sp;
        int latCh = _config.InVisualDim;
        if (firstFrameLatent is not null &&
            (firstFrameLatent.Shape.Rank != 5 || firstFrameLatent.Shape[0] != 1 || firstFrameLatent.Shape[1] != latCh
             || firstFrameLatent.Shape[2] != 1 || firstFrameLatent.Shape[3] != hLat || firstFrameLatent.Shape[4] != wLat))
            throw new ArgumentException(
                $"firstFrameLatent must be [1,{latCh},1,{hLat},{wLat}]; got {firstFrameLatent.Shape}.",
                nameof(firstFrameLatent));

        int steps = request.Steps ?? 50;
        float cfgScale = request.CfgScale ?? 5.0f;
        bool useCfg = cfgScale > 1.0f;
        if (useCfg && (negQwenEmbeds is null || negClipPooled is null))
            throw new ArgumentException(
                "cfgScale > 1.0 requires both negQwenEmbeds and negClipPooled.", nameof(negQwenEmbeds));

        (float scaleT, float scaleH, float scaleW) = GetRopeScaleFactor(height, width);
        string mode = firstFrameLatent is null ? "T2V" : "I2V";
        Logs.Info($"Kandinsky5-Video {mode}: {numFrames}f {width}x{height}, {steps} steps, " +
                  $"cfg={cfgScale}, seed={seed} (latent {latCh}x{tLat}x{hLat}x{wLat}, shift={_schedulerShift}, " +
                  $"rope_scale=({scaleT},{scaleH},{scaleW}))");
        Logs.Warning("Kandinsky5-Video is first-run-validation pending — numerics unverified vs the reference checkpoint.");
        if (numFrames > 121)
            Logs.Warning($"Kandinsky5-Video: {numFrames} frames exceeds the 121-frame (5 s) envelope the dense-attention " +
                         "path is sized for; the checkpoint was trained with NABLA block-sparse attention at this length " +
                         "and dense results may deviate (and attention cost grows quadratically).");

        // ── Noise + static conditioning channels. ──
        TensorShape latentShape = new TensorShape([1L, latCh, tLat, hLat, wLat]);
        Tensor noisy = SeedGenerator.CreateNoise(latentShape, seed);

        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_schedulerShift);
        scheduler.SetTimesteps(steps);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            Backend.Scale(scaled, noisy, initSigma);
            noisy.Dispose();
            noisy = scaled;
        }

        Tensor condLatent = Zeros(new TensorShape([1L, latCh, tLat, hLat, wLat]));
        Tensor condMask = Zeros(new TensorShape([1L, 1, tLat, hLat, wLat]));
        if (firstFrameLatent is not null)
        {
            WriteFrame(noisy, 0, firstFrameLatent);
            WriteFrame(condLatent, 0, firstFrameLatent);
            FillFrame(condMask, 0, 1.0f);
        }

        // Bulk-upload transformer weights before the denoise loop; paired with FreeWeights after it.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < steps; i++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor packed = PackVisualCond(noisy, condLatent, condMask);
            Tensor velocity;
            if (useCfg)
            {
                // Paired forward: cond+uncond, captured as one CUDA graph under HARTSY_DIT_GRAPH (both passes share
                // the packed latent's patch-embed), else two eager forwards. Same numerics either way.
                (Tensor cond, Tensor uncond) = _transformer.ForwardVideoPaired(Backend, packed, t,
                    qwenEmbeds, clipPooled, negQwenEmbeds!, negClipPooled!, scaleT, scaleH, scaleW);
                velocity = CfgHelper.ApplyCfg(uncond, cond, cfgScale);
                uncond.Dispose();
                cond.Dispose();
            }
            else
            {
                velocity = _transformer.ForwardVideo(Backend, packed, t, qwenEmbeds, clipPooled,
                    scaleT, scaleH, scaleW);
            }
            packed.Dispose();

            Tensor next = new Tensor(latentShape, DType.F32);
            scheduler.Step(next, velocity, noisy, i);
            velocity.Dispose();
            noisy.Dispose();
            noisy = next;

            // I2V: frame 0 stays the image latent (the reference only steps frames 1..).
            if (firstFrameLatent is not null)
                WriteFrame(noisy, 0, firstFrameLatent);

            sw.Stop();
            Logs.Debug($"Kandinsky5-Video step {i + 1}/{steps} (t={t:F1}) in {sw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, sw.Elapsed.TotalMilliseconds));

            // Reclaim GPU-resident activations between steps (undisposed cached intermediates otherwise
            // accumulate to OOM over the loop). Safe: the scheduler steps the latent on the host, and the
            // conditioning tensors are host-backed (re-uploaded per step). trimPool:false — steps are
            // identical, so the pool reservation is reused instead of a per-step driver release/re-map.
            // SKIP on the step-graph path: the captured graph balances its own allocations and holds the fixed
            // buffers at the addresses the capture bakes — freeing them here would corrupt the replay.
            if (!_transformer.StepGraphActive) Backend.FreeActivations(trimPool: false);
        }

        condLatent.Dispose();
        condMask.Dispose();

        if (firstFrameLatent is not null)
            NormalizeFirstFrames(noisy);

        Backend.Sync();
        // The captured step graph bakes the DiT weight pointers; invalidate before freeing the weights so the next
        // generation re-warms and re-captures instead of replaying against freed memory.
        _transformer.InvalidateStepGraph(Backend);
        Backend.FreeWeights(_transformer.EnumerateWeights());
        return noisy;
    }

    /// <summary>Packs the 33-channel model input <c>concat([noisy(16), cond(16), mask(1)])</c> along the channel axis.</summary>
    private Tensor PackVisualCond(Tensor noisy, Tensor condLatent, Tensor condMask)
    {
        int latCh = (int)noisy.Shape[1];
        long b = noisy.Shape[0], t = noisy.Shape[2], h = noisy.Shape[3], w = noisy.Shape[4];
        Tensor packed = new Tensor(new TensorShape([b, 2L * latCh + 1, t, h, w]), DType.F32);

        long per = t * h * w;
        float* dst = (float*)packed.DataPointer;
        float* pn = (float*)noisy.DataPointer;
        float* pc = (float*)condLatent.DataPointer;
        float* pm = (float*)condMask.DataPointer;
        long chOut = 2L * latCh + 1;
        for (long bi = 0; bi < b; bi++)
        {
            long dstBase = bi * chOut * per;
            Buffer.MemoryCopy(pn + bi * latCh * per, dst + dstBase, latCh * per * 4, latCh * per * 4);
            Buffer.MemoryCopy(pc + bi * latCh * per, dst + dstBase + latCh * per, latCh * per * 4, latCh * per * 4);
            Buffer.MemoryCopy(pm + bi * per, dst + dstBase + 2L * latCh * per, per * 4, per * 4);
        }
        return packed;
    }

    /// <summary>Ports the reference I2V <c>normalize_first_frame</c> "mesh artifact" fix: adaptive
    /// mean/std normalization of the first 4 latent frames against frames 4..8, with the reference's
    /// clamp constants. Skipped when there are not enough frames to form a reference window.</summary>
    private static void NormalizeFirstFrames(Tensor latent)
    {
        int tLat = (int)latent.Shape[2];
        const int firstCount = 4;
        const int referenceCount = 5;
        if (tLat <= firstCount + 1) return;

        int refCount = Math.Min(referenceCount, tLat - firstCount);
        (float srcMean, float srcStd) = FrameRangeStats(latent, 0, firstCount);
        (float refMean, float refStd) = FrameRangeStats(latent, firstCount, refCount);
        if (srcStd < 1e-12f) return;

        // Magic constants from the reference — limit how far the first frames may move.
        refMean = Math.Clamp(refMean, srcMean - 0.05f, srcMean + 0.1f);
        refStd = Math.Clamp(refStd, srcStd - 0.1f, srcStd + 0.25f);

        int ch = (int)latent.Shape[1];
        long frame = latent.Shape[3] * latent.Shape[4];
        float* p = (float*)latent.DataPointer;
        for (int c = 0; c < ch; c++)
        {
            long chBase = (long)c * tLat * frame;
            for (int t = 0; t < firstCount; t++)
            {
                long off = chBase + t * frame;
                for (long i = 0; i < frame; i++)
                {
                    float normalized = (p[off + i] - srcMean) / srcStd;
                    p[off + i] = normalized * refStd + refMean;
                }
            }
        }
    }

    /// <summary>Mean/std (population) over frames <c>[tStart, tStart+count)</c> of a <c>[1, C, T, H, W]</c> latent.</summary>
    private static (float mean, float std) FrameRangeStats(Tensor latent, int tStart, int count)
    {
        int ch = (int)latent.Shape[1];
        int tLat = (int)latent.Shape[2];
        long frame = latent.Shape[3] * latent.Shape[4];
        float* p = (float*)latent.DataPointer;
        double sum = 0, sumSq = 0;
        long n = (long)ch * count * frame;
        for (int c = 0; c < ch; c++)
        {
            long chBase = (long)c * tLat * frame;
            for (int t = tStart; t < tStart + count; t++)
            {
                long off = chBase + t * frame;
                for (long i = 0; i < frame; i++)
                {
                    double v = p[off + i];
                    sum += v;
                    sumSq += v * v;
                }
            }
        }
        double mean = sum / n;
        // Unbiased (n−1) variance to match torch.std() defaults in the reference.
        double variance = n > 1 ? Math.Max(0.0, (sumSq - n * mean * mean) / (n - 1)) : 0.0;
        return ((float)mean, (float)Math.Sqrt(variance));
    }

    private static Tensor Zeros(TensorShape shape)
    {
        Tensor t = new Tensor(shape, DType.F32);
        new Span<float>((float*)t.DataPointer, checked((int)shape.ElementCount)).Clear();
        return t;
    }

    /// <summary>Overwrites latent frame <paramref name="frameIndex"/> of <c>[1, C, T, H, W]</c> with <paramref name="condition"/> <c>[1, C, 1, H, W]</c>.</summary>
    private static void WriteFrame(Tensor latent, int frameIndex, Tensor condition)
    {
        int c = (int)latent.Shape[1], t = (int)latent.Shape[2];
        long frame = latent.Shape[3] * latent.Shape[4];
        float* lp = (float*)latent.DataPointer;
        float* cp = (float*)condition.DataPointer;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(cp + ci * frame, lp + ((long)ci * t + frameIndex) * frame, frame * 4, frame * 4);
    }

    /// <summary>Fills latent frame <paramref name="frameIndex"/> of <c>[1, C, T, H, W]</c> with a constant.</summary>
    private static void FillFrame(Tensor latent, int frameIndex, float value)
    {
        int c = (int)latent.Shape[1], t = (int)latent.Shape[2];
        long frame = latent.Shape[3] * latent.Shape[4];
        float* lp = (float*)latent.DataPointer;
        for (int ci = 0; ci < c; ci++)
        {
            long off = ((long)ci * t + frameIndex) * frame;
            for (long i = 0; i < frame; i++) lp[off + i] = value;
        }
    }
}
