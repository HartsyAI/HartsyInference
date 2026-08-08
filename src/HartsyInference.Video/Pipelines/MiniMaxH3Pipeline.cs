using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>One MiniMax-H3 generation: a flow-match loop over the packed audio+video sequence, then the two VAEs.
/// The sampler integrates the video sigma only — the audio stream rides its own shifted schedule and has its velocity
/// rescaled by the schedule map's derivative, so a single Euler integrator solves both streams correctly.</summary>
public sealed unsafe class MiniMaxH3Pipeline : DiffusionPipelineBase
{
    private readonly MiniMaxH3Transformer _transformer;
    private readonly MiniMaxH3VideoVaeDecoder _videoVae;
    private readonly MiniMaxH3AudioVaeDecoder? _audioVae;
    private readonly MiniMaxH3Config _config;
    private readonly bool _preloadTransformer;

    /// <param name="preloadTransformer">False for checkpoints too large to stay device-resident (the bf16 build).</param>
    public MiniMaxH3Pipeline(IBackend backend, MiniMaxH3Transformer transformer, MiniMaxH3VideoVaeDecoder videoVae,
        MiniMaxH3AudioVaeDecoder? audioVae, bool preloadTransformer = true) : base(backend)
    {
        _preloadTransformer = preloadTransformer;
        _transformer = transformer;
        _videoVae = videoVae;
        _audioVae = audioVae;
        _config = transformer.Config;
    }

    /// <summary>Decoded output: RGB frames plus the jointly generated stereo soundtrack.</summary>
    public readonly record struct Result(byte[][] Frames, int Width, int Height, int Seed,
        float[][]? Audio, int AudioSampleRate);

    /// <summary>Runs the denoise loop and both decodes. <paramref name="textStates"/> is the Qwen3-VL hidden state;
    /// <paramref name="textTagRuns"/> carries the per-token modality tags so vision pads inside the text span
    /// modulate as video.</summary>
    public Result Generate(Tensor textStates, MiniMaxH3GenerationRequest request,
        IReadOnlyList<(int Start, int Stop, int Tag)>? textTagRuns = null, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(textStates);
        ArgumentNullException.ThrowIfNull(request);

        int latentT = request.LatentFrames, latentH = request.Height / 16, latentW = request.Width / 16;
        if (latentH * 16 != request.Height || latentW * 16 != request.Width)
        {
            // 16x spatial compression; the caller snaps geometry before reaching here.
            Logs.Warning($"[MiniMaxH3] {request.Width}x{request.Height} is not a multiple of 16.");
        }
        int audioT = request.AudioLatentFrames;
        int textLen = (int)textStates.Shape[0];
        int seed = request.Seed;

        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(textLen, latentT, latentH, latentW, audioT,
            request.Keyframes, request.Refs, request.FrameCount);
        int frameRows = (latentH / _config.PatchH) * (latentW / _config.PatchW);
        int videoRowCount = latentT * frameRows;
        int audioRowCount = audioT * 2;

        (int condVideoRows, int condAudioRows) = MiniMaxH3Conditioning.ConditioningRowCounts(layout);
        RequireConditioningRows(request.CondVideoRows, condVideoRows, _config.VideoPatchDim, "video");
        RequireConditioningRows(request.CondAudioRows, condAudioRows, _config.AudioLatentsDim, "audio");

        Tensor videoLat = SeedGenerator.CreateNoise(new TensorShape(videoRowCount, _config.VideoPatchDim), seed);
        Tensor audioLat = SeedGenerator.CreateNoise(new TensorShape(audioRowCount, _config.AudioLatentsDim), seed ^ 0x5D2B);
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(layout.PositionIds, _transformer.RopeInvFreq(), _config.AttentionHeadDim);
        // The reference re-derives augmented conditioning from the same seeded stream on every forward, so hoisting it
        // out of the loop is the same values for a fraction of the work.
        Tensor? condVideo = NoiseAugment(request.CondVideoRows, request.VisualCondNoiseAug, seed);
        Tensor? condAudio = NoiseAugment(request.CondAudioRows, request.AudioCondNoiseAug, seed + 1);

        double shiftV = request.SigmaShiftVideo, shiftA = request.SigmaShiftAudio;
        double[] sigmas = MiniMaxH3Schedule.VideoSigmas(request.Steps, shiftV);

        // Park the DiT on device up front. Without this its weights land in the cache lazily per op and get evicted
        // by whatever else is resident, so every Linear pays a fresh host->device upload. Gated on it actually
        // fitting: the 66 GB bf16 build is larger than the card and must stay on the per-call streaming path.
        bool preloaded = TryPreloadTransformer();
        try
        {
            for (int step = 0; step < request.Steps; step++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                double sigma = sigmas[step];
                double dSigma = sigmas[step + 1] - sigma;
                (float tVideo, float tAudio) = MiniMaxH3Schedule.Timesteps(sigma, shiftV, shiftA);
                (float[] uniqueT, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
                    MiniMaxH3Conditioning.BuildTimestepRows(layout, tVideo, tAudio,
                        request.VisualCondNoiseAug, request.AudioCondNoiseAug);

                if (step == 0)
                {
                    Probe("video latent (noise, pre-step)", videoLat);
                    Probe("audio latent (noise, pre-step)", audioLat);
                }
                // Conditioning rows ride every forward at their fixed content and are never integrated, so they are
                // spliced back in fresh each step rather than living in the state the sampler advances.
                Tensor videoIn = PackConditioned(condVideo, videoLat);
                Tensor audioIn = PackConditioned(condAudio, audioLat);
                try
                {
                    (Tensor vVideo, Tensor vAudio) = DitShardBackend is not null
                        ? _transformer.ForwardSharded(
                            Backend, DitShardBackend, layout, videoIn, audioIn, textStates, cos, sin, uniqueT, rowOf,
                            DitShardSplitBlock, textTagRuns)
                        : _transformer.Forward(
                            Backend, layout, videoIn, audioIn, textStates, cos, sin, uniqueT, rowOf, textTagRuns);
                    try
                    {
                        // Both heads return the flow velocity; the audio one is integrated over the video sigma, so it
                        // takes the schedule map's derivative as an extra factor.
                        float slope = (float)MiniMaxH3Schedule.ShiftSlope(sigma, shiftV, shiftA);
                        if (step == 0 || step == request.Steps / 2 || step == request.Steps - 1)
                        {
                            Probe($"DiT velocity (video, step {step})", vVideo);
                            Probe($"DiT velocity (audio, step {step}, sigmaV={sigma:F4} tA={tAudio:F4} slope={slope:F4})", vAudio);
                        }
                        EulerStep(videoLat, vVideo, (float)-dSigma);
                        EulerStep(audioLat, vAudio, (float)(-dSigma * slope));
                    }
                    finally
                    {
                        vVideo.Dispose();
                        vAudio.Dispose();
                    }
                }
                finally
                {
                    if (!ReferenceEquals(videoIn, videoLat)) { videoIn.Dispose(); }
                    if (!ReferenceEquals(audioIn, audioLat)) { audioIn.Dispose(); }
                }
                Backend.Sync();
                DitShardBackend?.Sync();
                sw.Stop();
                Logs.Info($"[minimax-h3] step {step + 1}/{request.Steps}: {sw.ElapsedMilliseconds} ms");
                onProgress?.Invoke(new GenerationProgress(step + 1, request.Steps, sw.Elapsed.TotalMilliseconds));
                // Window the op profiler onto the steady-state steps: step 0 carries the weight-residency
                // warm-up, and everything before the loop is text encode. Both are one-time and vary enough
                // run to run that differencing two runs to cancel them does not work. No-op when off.
                if (step == 0)
                {
                    Backend.ResetOpProfile();
                }
            }
            Backend.DumpOpProfile($"denoise{Math.Max(1, request.Steps - 1)}");

            Probe("video latent (final)", videoLat);
            Probe("audio latent (final)", audioLat);
            Dump("video_latent_final", videoLat);
            Dump("audio_latent_final", audioLat);
            // Denoising is done: hand the DiT's ~20 GB back before the VAE needs its own ~5 GB. Sharded, the free
            // mirrors the asymmetric preload — the whole-set free would silently no-op on the shard backend's
            // range (frees are per-backend) and leak it; this also returns the second card's share pre-decode.
            if (preloaded)
            {
                if (DitShardBackend is not null)
                {
                    Backend.FreeWeights(_transformer.EnumerateSharedWeights());
                    Backend.FreeWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
                    DitShardBackend.FreeWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _config.NumLayers));
                }
                else
                {
                    Backend.FreeWeights(_transformer.EnumerateWeights());
                }
            }

            Tensor videoLatent = MiniMaxH3Latents.UnpackVideo(videoLat, latentT, latentH, latentW, _config);
            Tensor rgb;
            bool videoVaePreloaded = TryPreloadWeights(Backend, "video VAE decoder", _videoVae.EnumerateWeights());
            try
            {
                rgb = _videoVae.Decode(Backend, videoLatent);
            }
            finally
            {
                videoLatent.Dispose();
                if (videoVaePreloaded) Backend.FreeWeights(_videoVae.EnumerateWeights());
            }

            byte[][] frames;
            int outW, outH;
            try
            {
                outH = (int)rgb.Shape[3];
                outW = (int)rgb.Shape[4];
                int f = (int)rgb.Shape[2];
                frames = new byte[f][];
                for (int i = 0; i < f; i++)
                {
                    frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
                }
            }
            finally
            {
                rgb.Dispose();
            }

            float[][]? audio = null;
            int sampleRate = 0;
            if (_audioVae is not null)
            {
                Tensor audioLatent = MiniMaxH3Latents.UnpackAudio(audioLat, audioT, _config);
                bool audioVaePreloaded = TryPreloadWeights(Backend, "audio VAE decoder", _audioVae.EnumerateWeights());
                Tensor wave = _audioVae.Decode(Backend, audioLatent);
                try
                {
                    sampleRate = _audioVae.SampleRate;
                    int ch = (int)wave.Shape[1], samples = (int)wave.Shape[2];
                    audio = new float[ch][];
                    float* wp = (float*)wave.DataPointer;
                    for (int c = 0; c < ch; c++)
                    {
                        audio[c] = new float[samples];
                        for (int i = 0; i < samples; i++) audio[c][i] = wp[(long)c * samples + i];
                    }
                }
                finally
                {
                    audioLatent.Dispose();
                    wave.Dispose();
                    if (audioVaePreloaded) Backend.FreeWeights(_audioVae.EnumerateWeights());
                }
            }
            return new Result(frames, outW, outH, seed, audio, sampleRate);
        }
        finally
        {
            videoLat.Dispose();
            audioLat.Dispose();
            cos.Dispose();
            sin.Dispose();
            if (!ReferenceEquals(condVideo, request.CondVideoRows)) { condVideo?.Dispose(); }
            if (!ReferenceEquals(condAudio, request.CondAudioRows)) { condAudio?.Dispose(); }
        }
    }

    /// <summary>The packed input for one stream: conditioning rows then the denoise target. Returns
    /// <paramref name="target"/> itself when there is no conditioning, so plain t2va allocates and copies nothing.</summary>
    private Tensor PackConditioned(Tensor? conditioning, Tensor target)
    {
        if (conditioning is null)
        {
            return target;
        }
        Tensor packed = new Tensor(new TensorShape(conditioning.Shape[0] + target.Shape[0], target.Shape[1]), target.DType);
        Backend.Concat(packed, [conditioning, target], 0);
        return packed;
    }

    /// <summary>Blends conditioning rows toward seeded noise by <c>1 - aug</c>, keeping the model from treating them as
    /// perfectly clean. Returns the input untouched at <c>aug >= 1</c>, so the caller must compare by reference before
    /// disposing.</summary>
    private static Tensor? NoiseAugment(Tensor? rows, float aug, int seed)
    {
        if (rows is null || aug >= 1f)
        {
            return rows;
        }
        Tensor augmented = new Tensor(rows.Shape, DType.F32);
        using Tensor noise = SeedGenerator.CreateNoise(rows.Shape, seed);
        float* src = (float*)rows.DataPointer;
        float* np = (float*)noise.DataPointer;
        float* dst = (float*)augmented.DataPointer;
        for (long i = 0; i < rows.ElementCount; i++)
        {
            dst[i] = aug * src[i] + (1f - aug) * np[i];
        }
        return augmented;
    }

    /// <summary>Fails a mismatch between the layout's conditioning rows and the content supplied for them — a silent
    /// disagreement here shifts every packed row and decodes to plausible garbage rather than erroring.</summary>
    private static void RequireConditioningRows(Tensor? rows, int expectedRows, int expectedWidth, string stream)
    {
        long actual = rows?.Shape[0] ?? 0;
        if (actual != expectedRows)
        {
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"MiniMax-H3 layout expects {expectedRows} conditioning {stream} row(s), got {actual}.");
        }
        if (rows is not null && rows.Shape[1] != expectedWidth)
        {
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"MiniMax-H3 conditioning {stream} rows are {rows.Shape[1]} wide, expected {expectedWidth}.");
        }
    }

    /// <summary>Logs min/max/mean/rms under <c>HARTSY_H3_PROBE=1</c>; no-op otherwise.</summary>
    private static void Probe(string label, Tensor t)
    {
        if (Environment.GetEnvironmentVariable("HARTSY_H3_PROBE") != "1")
        {
            return;
        }
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float* p = (float*)f.DataPointer;
        long n = f.ElementCount;
        float mn = float.MaxValue, mx = float.MinValue;
        double sum = 0, sq = 0;
        long bad = 0;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            if (!float.IsFinite(v)) { bad++; continue; }
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v; sq += (double)v * v;
        }
        Logs.Warning($"[h3-probe] {label}: min={mn:F4} max={mx:F4} mean={sum / n:F4} rms={Math.Sqrt(sq / n):F4} nonfinite={bad} n={n}");
        if (!ReferenceEquals(f, t)) f.Dispose();
    }

    /// <summary>Makes the DiT device-resident when the recipe determined it fits, leaving headroom for activations
    /// and the VAE decode. When it does not fit (the 66 GB bf16 build) the per-call streaming path stands.</summary>
    private bool TryPreloadTransformer()
    {
        if (!_preloadTransformer && DitShardBackend is null)
        {
            Logs.Info("[MiniMaxH3] DiT too large to stay resident — streaming per call.");
            return false;
        }
        try
        {
            if (DitShardBackend is not null)
            {
                // Asymmetric split — shared + [0, split) on the primary, ONLY [split, NumLayers) on the shard
                // backend. EnumerateWeights() on both would replicate instead of pool.
                Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
                Backend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
                DitShardBackend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _config.NumLayers));
            }
            else
            {
                Backend.PreloadWeights(_transformer.EnumerateWeights());
            }
            return true;
        }
        catch (HartsyInference.Core.Exceptions.OutOfVramException ex)
        {
            // Residency is an optimization, never a requirement: PreloadWeights rolls its batch back on OOM, so the
            // per-call streaming path is still correct — degrade instead of failing the generation. On the sharded
            // route, drop whatever partial batches landed so neither card is left holding a half-preloaded range.
            Logs.Warning($"[MiniMaxH3] DiT preload did not fit ({ex.Message}) — streaming per call.");
            if (DitShardBackend is not null)
            {
                Backend.FreeWeights(_transformer.EnumerateSharedWeights());
                Backend.FreeWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
                DitShardBackend.FreeWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _config.NumLayers));
            }
            return false;
        }
    }

    /// <summary>Best-effort weight residency for a phase-scoped component (the VAE decoders, called once at the end
    /// of a generation) — mirrors <see cref="TryPreloadTransformer"/>'s degrade-not-fail contract: an
    /// <see cref="HartsyInference.Core.Exceptions.OutOfVramException"/> during preload just means the phase falls
    /// back to the existing lazy per-op streaming path, never a failed generation.</summary>
    private static bool TryPreloadWeights(IBackend backend, string label, IEnumerable<Tensor> weights)
    {
        try
        {
            backend.PreloadWeights(weights);
            return true;
        }
        catch (HartsyInference.Core.Exceptions.OutOfVramException ex)
        {
            Logs.Warning($"[MiniMaxH3] {label} preload did not fit ({ex.Message}) — streaming per call.");
            return false;
        }
    }

    /// <summary>Writes the raw F32 tensor to <c>$HARTSY_H3_DUMP/&lt;name&gt;.bin</c> for reference comparison; no-op
    /// when the variable is unset.</summary>
    private static void Dump(string name, Tensor t)
    {
        string? dir = Environment.GetEnvironmentVariable("HARTSY_H3_DUMP");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }
        Directory.CreateDirectory(dir);
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        using (FileStream fs = File.Create(Path.Combine(dir, name + ".bin")))
        {
            fs.Write(new ReadOnlySpan<byte>((void*)f.DataPointer, checked((int)(f.ElementCount * 4))));
        }
        Logs.Warning($"[h3-dump] {name} {f.Shape} -> {dir}");
        if (!ReferenceEquals(f, t)) { f.Dispose(); }
    }

    /// <summary><c>z += v * delta</c> in place.</summary>
    private void EulerStep(Tensor z, Tensor velocity, float delta)
    {
        Backend.CfgEulerStep(z, velocity, velocity, 1f, delta);
    }

    protected override void DisposeCore()
    {
        _transformer.Dispose();
    }
}
