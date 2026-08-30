using HartsyInference.Core.Configuration;
using System.Diagnostics;
using System.Linq;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.MiniMaxH3;

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
    private readonly MiniMaxH3PddAdapter? _pddAdapter;
    private readonly float _pddAdapterStrength;
    private readonly VideoSparseAttentionProfileKind? _sparseAttentionProfile;

    /// <param name="preloadTransformer">False for checkpoints too large to stay device-resident (the bf16 build).</param>
    public MiniMaxH3Pipeline(IBackend backend, MiniMaxH3Transformer transformer, MiniMaxH3VideoVaeDecoder videoVae,
        MiniMaxH3AudioVaeDecoder? audioVae, bool preloadTransformer = true,
        MiniMaxH3PddAdapter? pddAdapter = null, float pddAdapterStrength = 1f,
        VideoSparseAttentionProfileKind? sparseAttentionProfile = null) : base(backend)
    {
        _preloadTransformer = preloadTransformer;
        _transformer = transformer;
        _videoVae = videoVae;
        _audioVae = audioVae;
        _config = transformer.Config;
        _pddAdapter = pddAdapter;
        _pddAdapterStrength = pddAdapterStrength;
        _sparseAttentionProfile = sparseAttentionProfile;
    }

    /// <summary>Whether <see cref="DiffusionPipelineBase.VaeBackend"/> is a device the DiT does not use — the operator
    /// placed it there deliberately, so the decoders' weights stay resident between generations rather than being
    /// freed for a DiT that isn't competing for that space. Pointing the decodes at another device needs no peer-copy
    /// machinery: <see cref="MiniMaxH3Latents"/>' unpack step is a host loop into a fresh tensor, so the latent
    /// reaches the decoder with no device association to carry across.</summary>
    private bool VaeIsWarmPlaced => !ReferenceEquals(VaeBackend, Backend);

    /// <summary>Decoded output: RGB frames plus the jointly generated stereo soundtrack.</summary>
    public readonly record struct Result(byte[][] Frames, int Width, int Height, int Seed,
        float[][]? Audio, int AudioSampleRate);

    /// <summary>Bytes the primary backend will hold resident once <see cref="Generate"/> preloads: the full DiT when
    /// unsharded, or only the shared weights plus its own <c>[0, DitShardSplitBlock)</c> block range when
    /// <see cref="DiffusionPipelineBase.DitShardBackend"/> is set — the shard backend holds the remaining blocks (see
    /// <see cref="EstimateShardResidentWeightBytes"/>), so counting the whole set here would double-charge it. Zero
    /// when unsharded and this checkpoint doesn't fit resident (the bf16 build), since it then streams per op
    /// instead. A pre-flight VRAM check run before <see cref="Generate"/> needs this to know how much of current
    /// free VRAM the DiT is about to claim, on top of the activation floor it also needs.</summary>
    public long EstimateResidentWeightBytes()
    {
        if (DitShardBackend is not null)
        {
            return SumBytes(_transformer.EnumerateSharedWeights())
                + SumBytes(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
        }
        return _preloadTransformer ? SumBytes(_transformer.EnumerateWeights()) : 0;
    }

    /// <summary>Bytes the shard backend will hold resident — its own <c>[DitShardSplitBlock, NumLayers)</c> block
    /// range only, since shared weights (patch/time-embed/final-layer projections) always live on the primary
    /// backend. Zero when <see cref="DiffusionPipelineBase.DitShardBackend"/> is not set.</summary>
    public long EstimateShardResidentWeightBytes() => DitShardBackend is not null
        ? SumBytes(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _config.NumLayers)) : 0;

    private static long SumBytes(IEnumerable<Tensor> weights) =>
        weights.Sum(t => t.DType.ComputeByteCount(t.ElementCount));

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
        VideoSparseAttentionPlan? sparsePlan = ResolveSparseAttentionPlan(layout, request);
        int frameRows = (latentH / _config.PatchH) * (latentW / _config.PatchW);
        int videoRowCount = latentT * frameRows;
        int audioRowCount = audioT * 2;

        (int condVideoRows, int condAudioRows) = MiniMaxH3Conditioning.ConditioningRowCounts(layout);
        RequireConditioningRows(request.CondVideoRows, condVideoRows, _config.VideoPatchDim, "video");
        RequireConditioningRows(request.CondAudioRows, condAudioRows, _config.AudioLatentsDim, "audio");

        (IReadOnlyList<float>? videoMaskRows, IReadOnlyList<float>? videoFeatureMaskValues) =
            NormalizeDenoiseMask(
                request.VideoDenoiseMaskRows, request.VideoDenoiseFeatureMaskValues,
                request.VideoDenoiseSourceRows, videoRowCount, _config.VideoPatchDim,
                _config.PatchT * _config.PatchH * _config.PatchW, "video");
        (IReadOnlyList<float>? audioMaskRows, IReadOnlyList<float>? audioFeatureMaskRows) =
            NormalizeDenoiseMask(
                request.AudioDenoiseMaskRows, request.AudioDenoiseFeatureMaskRows,
                request.AudioDenoiseSourceRows, audioRowCount, _config.AudioLatentsDim,
                patchArea: 1, modality: "audio");
        bool hasDenoiseMask = videoFeatureMaskValues is not null || audioFeatureMaskRows is not null;
        if (request.Controls?.Any(static control => control.IsInpaint) == true && hasDenoiseMask)
        {
            throw new NotSupportedException(
                "MiniMax-H3 Fun ControlNet inpainting cannot be combined with video or audio denoise masks.");
        }
        if (_pddAdapter is not null && hasDenoiseMask)
        {
            throw new NotSupportedException(
                "MiniMax-H3 PDD cannot be combined with video or audio denoise masks.");
        }

        Tensor videoLat = SeedGenerator.CreateNoise(new TensorShape(videoRowCount, _config.VideoPatchDim), seed);
        Tensor audioLat = SeedGenerator.CreateNoise(new TensorShape(audioRowCount, _config.AudioLatentsDim), seed ^ 0x5D2B);
        Tensor? videoFixedNoise = null;
        Tensor? videoInjection = null;
        Tensor? videoTokenMask = null;
        Tensor? videoFeatureMask = null;
        Tensor? videoModelScratch = null;
        Tensor? videoDenoisedScratch = null;
        Tensor? audioTokenMask = null;
        Tensor? audioFeatureMask = null;
        Tensor? audioModelScratch = null;
        Tensor? audioDenoisedScratch = null;
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(layout.PositionIds, _transformer.RopeInvFreq(), _config.AttentionHeadDim);
        // The reference re-derives augmented conditioning from the same seeded stream on every forward, so hoisting it
        // out of the loop is the same values for a fraction of the work.
        Tensor? condVideo = NoiseAugment(request.CondVideoRows, request.VisualCondNoiseAug, seed);
        Tensor? condAudio = NoiseAugment(request.CondAudioRows, request.AudioCondNoiseAug, seed + 1);

        double shiftV = request.SigmaShiftVideo, shiftA = request.SigmaShiftAudio;
        MiniMaxH3PddSchedule? pddSchedule = _pddAdapter is null ? null : MiniMaxH3PddSchedule.Create(
            new MiniMaxH3PddExecutionSettings
            {
                Nfe = request.Steps,
                Sampler = request.Sampler,
                CfgScale = request.CfgScale,
                VideoFlowShift = shiftV,
                AudioFlowShift = shiftA,
                Strength = _pddAdapterStrength,
                HasHybrid = request.HybridProfile,
            });
        double[] sigmas = pddSchedule is null
            ? MiniMaxH3Schedule.VideoSigmas(request.Steps, shiftV)
            : pddSchedule.Sigmas.ToArray();
        IReadOnlyList<MiniMaxH3FunControlCondition>?[] controlSchedule = BuildControlSchedule(
            request.Controls, request.Steps);

        using IVideoSparseAttentionSession? sparseAttentionPrimary = sparsePlan is null
            ? null : Backend.CreateVideoSparseAttentionSession(sparsePlan);
        using IVideoSparseAttentionSession? sparseAttentionShard = sparsePlan is null || DitShardBackend is null
            ? null : DitShardBackend.CreateVideoSparseAttentionSession(sparsePlan);
        // Park the DiT on device up front. Without this its weights land in the cache lazily per op and get evicted
        // by whatever else is resident, so every Linear pays a fresh host->device upload. Gated on it actually
        // fitting: the 66 GB bf16 build is larger than the card and must stay on the per-call streaming path.
        bool preloaded = TryPreloadTransformer();
        using PddHeadFusionSession? pddFusion = _pddAdapter is null
            ? null : new PddHeadFusionSession(Backend, _pddAdapter.HeadBank);
        try
        {
            try
            {
                if (videoFeatureMaskValues is not null)
                {
                    videoFixedNoise = SeedGenerator.CreateNoise(videoLat.Shape, seed);
                    videoInjection = new Tensor(videoLat.Shape, DType.F32);
                    BuildVideoMaskInjection(Backend, videoInjection,
                        request.VideoDenoiseSourceRows!, videoFixedNoise);
                    videoTokenMask = RowMaskTensor(videoMaskRows)!;
                    videoFeatureMask = FeatureMaskTensor(
                        videoFeatureMaskValues, videoRowCount,
                        _config.PatchT * _config.PatchH * _config.PatchW);
                    videoModelScratch = new Tensor(videoLat.Shape, DType.F32);
                    videoDenoisedScratch = new Tensor(videoLat.Shape, DType.F32);
                }
                if (audioFeatureMaskRows is not null)
                {
                    audioTokenMask = RowMaskTensor(audioMaskRows)!;
                    audioFeatureMask = RowMaskTensor(audioFeatureMaskRows)!;
                    audioModelScratch = new Tensor(audioLat.Shape, DType.F32);
                    audioDenoisedScratch = new Tensor(audioLat.Shape, DType.F32);
                }

                for (int step = 0; step < request.Steps; step++)
                {
                    Stopwatch sw = Stopwatch.StartNew();
                    double sigma = sigmas[step];
                    double dSigma = sigmas[step + 1] - sigma;
                    (float tVideo, float tAudio) = MiniMaxH3Schedule.Timesteps(sigma, shiftV, shiftA);
                    MiniMaxH3TimestepPlan timestepPlan = MiniMaxH3Conditioning.BuildMaskedTimestepRows(
                        layout, tVideo, tAudio, request.VisualCondNoiseAug, request.AudioCondNoiseAug,
                        videoMaskRows, audioMaskRows);

                    Tensor videoModelTarget = videoModelScratch ?? videoLat;
                    if (videoModelScratch is not null)
                    {
                        BuildMaskedModelInput(
                            Backend, videoModelTarget, videoLat, videoInjection!, videoTokenMask!);
                    }
                    Tensor audioModelTarget = audioModelScratch ?? audioLat;
                    if (audioModelScratch is not null)
                    {
                        BuildMaskedModelInput(
                            Backend, audioModelTarget, audioLat,
                            request.AudioDenoiseSourceRows!, audioTokenMask!);
                    }

                    if (step == 0)
                    {
                        Probe("video latent (noise, pre-step)", videoLat);
                        Probe("audio latent (noise, pre-step)", audioLat);
                    }
                    // Conditioning rows ride every forward at their fixed content and are never integrated, so they are
                    // spliced back in fresh each step rather than living in the state the sampler advances.
                    Tensor videoIn = PackConditioned(condVideo, videoModelTarget);
                    Tensor audioIn = PackConditioned(condAudio, audioModelTarget);
                    try
                    {
                        using PddFusedHeads? pddHeads = pddFusion is null || pddSchedule is null
                            ? null : pddFusion.Fuse(pddSchedule, sigma, sigmas[step + 1]);
                        (Tensor vVideo, Tensor vAudio) = DitShardBackend is not null
                            ? _transformer.ForwardSharded(
                                Backend, DitShardBackend, layout, videoIn, audioIn, textStates, cos, sin,
                                timestepPlan.Timesteps, timestepPlan.RowOf, DitShardSplitBlock,
                                textTagRuns: textTagRuns, pddHeads: pddHeads, controls: controlSchedule[step],
                                videoTimestepRows: timestepPlan.VideoRowOf,
                                audioTimestepRows: timestepPlan.AudioRowOf,
                                sparseAttentionA: sparseAttentionPrimary,
                                sparseAttentionB: sparseAttentionShard)
                            : _transformer.Forward(
                                Backend, layout, videoIn, audioIn, textStates, cos, sin,
                                timestepPlan.Timesteps, timestepPlan.RowOf, textTagRuns: textTagRuns,
                                pddHeads: pddHeads, controls: controlSchedule[step],
                                videoTimestepRows: timestepPlan.VideoRowOf,
                                audioTimestepRows: timestepPlan.AudioRowOf,
                                sparseAttention: sparseAttentionPrimary);
                        try
                        {
                            // Both heads return native-stream flow velocity. Unmasked audio keeps the derivative-scaled
                            // video-sigma integrator; masked audio uses the exact current/next native audio sigmas below.
                            float slope = (float)MiniMaxH3Schedule.ShiftSlope(sigma, shiftV, shiftA);
                            if (step == 0 || step == request.Steps / 2 || step == request.Steps - 1)
                            {
                                Probe($"DiT velocity (video, step {step})", vVideo);
                                Probe($"DiT velocity (audio, step {step}, sigmaV={sigma:F4} tA={tAudio:F4} slope={slope:F4})", vAudio);
                            }
                            double nextSigmaVideo = sigmas[step + 1];
                            if (videoModelScratch is null)
                            {
                                EulerStep(videoLat, vVideo, (float)-dSigma);
                            }
                            else
                            {
                                AdvanceMaskedState(
                                    Backend, videoModelScratch, videoDenoisedScratch!, videoLat,
                                    videoModelTarget, vVideo, request.VideoDenoiseSourceRows!, videoFeatureMask!,
                                    MaskBroadcastLayout.PackedChannelOuter, (float)sigma, (float)nextSigmaVideo);
                                Tensor previousVideoState = videoLat;
                                videoLat = videoModelScratch;
                                videoModelScratch = previousVideoState;
                            }
                            if (audioModelScratch is null)
                            {
                                EulerStep(audioLat, vAudio, (float)(-dSigma * slope));
                            }
                            else
                            {
                                float sigmaAudio = (float)MiniMaxH3Schedule.ShiftSigma(sigma, shiftV, shiftA);
                                float nextSigmaAudio = (float)MiniMaxH3Schedule.ShiftSigma(
                                    nextSigmaVideo, shiftV, shiftA);
                                AdvanceMaskedState(
                                    Backend, audioModelScratch, audioDenoisedScratch!, audioLat,
                                    audioModelTarget, vAudio, request.AudioDenoiseSourceRows!, audioFeatureMask!,
                                    MaskBroadcastLayout.Rows, sigmaAudio, nextSigmaAudio);
                                Tensor previousAudioState = audioLat;
                                audioLat = audioModelScratch;
                                audioModelScratch = previousAudioState;
                            }
                        }
                        finally
                        {
                            vVideo.Dispose();
                            vAudio.Dispose();
                        }
                    }
                    finally
                    {
                        if (!ReferenceEquals(videoIn, videoModelTarget)) { videoIn.Dispose(); }
                        if (!ReferenceEquals(audioIn, audioModelTarget)) { audioIn.Dispose(); }
                    }
                    Backend.Sync();
                    DitShardBackend?.Sync();
                    sw.Stop();
                    Logs.Info($"[minimax-h3] step {step + 1}/{request.Steps}: {sw.ElapsedMilliseconds} ms");
                    if (onProgress is not null)
                    {
                        Tensor previewLatent = MiniMaxH3Latents.UnpackVideo(
                            videoLat, latentT, latentH, latentW, _config);
                        onProgress.Invoke(new GenerationProgress(
                            step + 1, request.Steps, sw.Elapsed.TotalMilliseconds)
                        {
                            Latent = previewLatent,
                            LatentArch = LatentArchitecture.MiniMaxH3,
                        });
                        previewLatent.Dispose();
                    }
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
            }
            finally
            {
                // Persistent VSA route/stat buffers are generation-owned and must leave before the VAE competes
                // for memory. Dispose is idempotent; the using declarations remain the exception-safe fallback.
                sparseAttentionShard?.Dispose();
                sparseAttentionPrimary?.Dispose();
                // Denoising is done (or failed): hand the DiT's ~20 GB back before the VAE needs its own ~5 GB, and
                // so a failed generation doesn't leak it resident for every later request on this cached pipeline —
                // load-bearing for CheckVramFeasibility, which assumes weights are resident only while a Generate
                // call is actually using them. Sharded, the free mirrors the asymmetric preload — the whole-set free
                // would silently no-op on the shard backend's range (frees are per-backend) and leak it; this also
                // returns the second card's share pre-decode.
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
            }

            IBackend vaeBackend = VaeBackend;
            Tensor videoLatent = MiniMaxH3Latents.UnpackVideo(videoLat, latentT, latentH, latentW, _config);
            Tensor rgb;
            bool videoVaePreloaded = TryPreloadWeights(vaeBackend, "video VAE decoder", _videoVae.EnumerateWeights());
            try
            {
                rgb = _videoVae.Decode(vaeBackend, videoLatent);
            }
            finally
            {
                videoLatent.Dispose();
                // The preload flag still gates the free — it is false when PreloadWeights rolled back on OOM and the
                // lazy per-op path ran instead, where there is nothing resident to free.
                if (videoVaePreloaded && !VaeIsWarmPlaced) vaeBackend.FreeWeights(_videoVae.EnumerateWeights());
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
                bool audioVaePreloaded = false;
                Tensor wave = DecodeAudioWithCleanup(audioLatent,
                    () =>
                    {
                        audioVaePreloaded = TryPreloadWeights(
                            vaeBackend, "audio VAE decoder", _audioVae.EnumerateWeights());
                        return _audioVae.Decode(vaeBackend, audioLatent);
                    },
                    () =>
                    {
                        if (audioVaePreloaded && !VaeIsWarmPlaced)
                        {
                            vaeBackend.FreeWeights(_audioVae.EnumerateWeights());
                        }
                    });
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
                    wave.Dispose();
                }
            }
            return new Result(frames, outW, outH, seed, audio, sampleRate);
        }
        finally
        {
            videoLat.Dispose();
            audioLat.Dispose();
            videoModelScratch?.Dispose();
            videoDenoisedScratch?.Dispose();
            audioModelScratch?.Dispose();
            audioDenoisedScratch?.Dispose();
            cos.Dispose();
            sin.Dispose();
            if (!ReferenceEquals(condVideo, request.CondVideoRows)) { condVideo?.Dispose(); }
            if (!ReferenceEquals(condAudio, request.CondAudioRows)) { condAudio?.Dispose(); }
            videoFixedNoise?.Dispose();
            videoInjection?.Dispose();
            videoTokenMask?.Dispose();
            videoFeatureMask?.Dispose();
            audioTokenMask?.Dispose();
            audioFeatureMask?.Dispose();
        }
    }

    /// <summary>Runs audio decode setup/decode while guaranteeing that its unpacked input and any staged weights
    /// are released even when preload or decode throws before a waveform exists.</summary>
    internal static Tensor DecodeAudioWithCleanup(Tensor audioLatent, Func<Tensor> decode,
        Action releaseWeights)
    {
        ArgumentNullException.ThrowIfNull(audioLatent);
        ArgumentNullException.ThrowIfNull(decode);
        ArgumentNullException.ThrowIfNull(releaseWeights);
        try
        {
            return decode();
        }
        finally
        {
            try
            {
                audioLatent.Dispose();
            }
            finally
            {
                releaseWeights();
            }
        }
    }

    /// <summary>Performs the execution-boundary VSA preflight and builds the one layout every main block reuses.
    /// Any mismatch is terminal; a sparse checkpoint is never retried as dense after construction or launch.</summary>
    private VideoSparseAttentionPlan? ResolveSparseAttentionPlan(MiniMaxH3PackedLayout layout,
        MiniMaxH3GenerationRequest request)
    {
        if (_sparseAttentionProfile is not VideoSparseAttentionProfileKind profile)
        {
            return null;
        }
        if (_pddAdapter is not null || request.HybridProfile)
        {
            throw new NotSupportedException("MiniMax-H3 VSA cannot be combined with PDD or Hybrid execution.");
        }
        if (request.Keyframes is { Count: > 0 } || request.Refs is { Count: > 0 }
            || request.CondVideoRows is not null || request.CondAudioRows is not null
            || request.VideoDenoiseMaskRows is not null || request.AudioDenoiseMaskRows is not null
            || request.VideoDenoiseFeatureMaskValues is not null
            || request.AudioDenoiseFeatureMaskRows is not null
            || request.Controls is { Count: > 0 })
        {
            throw new NotSupportedException(
                "MiniMax-H3 VSA is T2VA-only and cannot consume references, guides, masks, or ControlNet.");
        }
        if (request.Steps != 4 || request.CfgScale != 1f
            || !string.Equals(request.Sampler, "euler", StringComparison.OrdinalIgnoreCase)
            || request.SigmaShiftVideo != 12f || request.SigmaShiftAudio != 3f)
        {
            throw new NotSupportedException(
                "MiniMax-H3 VSA requires exactly four Euler evaluations, CFG 1, and video/audio shifts 12/3.");
        }
        if (!Backend.SupportsVideoSparseAttention)
        {
            throw new NotSupportedException(
                $"Backend '{Backend.Capabilities.Name}' cannot execute the required MiniMax-H3 VSA profile.");
        }
        if (DitShardBackend is not null && !DitShardBackend.SupportsVideoSparseAttention)
        {
            throw new NotSupportedException(
                $"DiT shard backend '{DitShardBackend.Capabilities.Name}' cannot execute profile '{profile}'.");
        }
        return _transformer.CreateVideoSparseAttentionPlan(layout, profile);
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

    /// <summary>Validates the token and raw-feature views as one mask. Activity follows the raw view: a patch may
    /// pool to token value one while still preserving some of its individual features.</summary>
    internal static (IReadOnlyList<float>? TokenRows, IReadOnlyList<float>? FeatureValues) NormalizeDenoiseMask(
        IReadOnlyList<float>? tokenRows, IReadOnlyList<float>? featureValues, Tensor? source,
        int expectedRows, int expectedFeatures, int patchArea, string modality)
    {
        if (expectedRows <= 0 || expectedFeatures <= 0 || patchArea <= 0
            || expectedFeatures % patchArea != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(expectedRows),
                "MiniMax-H3 denoise-mask geometry must be positive and patch-aligned.");
        }

        bool tokenAllWhite = ValidateMaskValues(tokenRows, expectedRows, "token", modality);
        int expectedFeatureValues = checked(expectedRows * patchArea);
        bool featureAllWhite = ValidateMaskValues(
            featureValues, expectedFeatureValues, "feature", modality);
        if (featureValues is null)
        {
            if (tokenRows is not null && !tokenAllWhite)
            {
                throw new ArgumentException(
                    $"MiniMax-H3 {modality} denoise mask has active token rows but no raw feature values.",
                    nameof(featureValues));
            }
            return (null, null);
        }
        if (featureAllWhite)
        {
            if (tokenRows is not null && !tokenAllWhite)
            {
                throw new ArgumentException(
                    $"MiniMax-H3 {modality} raw mask is all-white but its token rows are active.",
                    nameof(tokenRows));
            }
            return (null, null);
        }
        if (tokenRows is null)
        {
            throw new ArgumentException(
                $"MiniMax-H3 {modality} denoise mask has raw feature values but no pooled token rows.",
                nameof(tokenRows));
        }
        for (int row = 0; row < expectedRows; row++)
        {
            float maximum = float.NegativeInfinity;
            int offset = row * patchArea;
            for (int patch = 0; patch < patchArea; patch++)
            {
                maximum = Math.Max(maximum, featureValues[offset + patch]);
            }
            float expectedToken = MiniMaxH3Masking.QuantizeTokenMask(maximum);
            if (tokenRows[row] != expectedToken)
            {
                throw new ArgumentException(
                    $"MiniMax-H3 {modality} denoise token row {row} is {tokenRows[row]:R}, but the raw feature "
                    + $"maximum requires {expectedToken:R} on the 1/256 grid.", nameof(tokenRows));
            }
        }
        if (source is null)
        {
            throw new ArgumentException(
                $"MiniMax-H3 {modality} denoise mask preserves rows but has no source latent.");
        }
        TensorShape expectedShape = new TensorShape(expectedRows, expectedFeatures);
        if (source.DType != DType.F32 || source.Shape != expectedShape)
        {
            throw new ArgumentException(
                $"MiniMax-H3 {modality} denoise source must be F32 {expectedShape}; got "
                + $"{source.DType} {source.Shape}.", nameof(source));
        }
        return (tokenRows, featureValues);

        static bool ValidateMaskValues(
            IReadOnlyList<float>? values, int expected, string representation, string modalityName)
        {
            if (values is null)
            {
                return true;
            }
            if (values.Count != expected)
            {
                throw new ArgumentException(
                    $"MiniMax-H3 {modalityName} denoise {representation} mask has {values.Count} value(s), "
                    + $"expected {expected}.");
            }
            bool allWhite = true;
            for (int i = 0; i < values.Count; i++)
            {
                float value = values[i];
                if (!UnitInterval.Contains(value))
                {
                    throw new ArgumentOutOfRangeException(nameof(values), value,
                        $"MiniMax-H3 {modalityName} {representation} mask value {i} must be finite and in [0,1].");
                }
                allWhite &= value == 1f;
            }
            return allWhite;
        }
    }

    private static unsafe Tensor? RowMaskTensor(IReadOnlyList<float>? rows)
    {
        if (rows is null)
        {
            return null;
        }
        Tensor mask = new Tensor(new TensorShape(rows.Count), DType.F32);
        float* destination = (float*)mask.DataPointer;
        for (int i = 0; i < rows.Count; i++)
        {
            destination[i] = rows[i];
        }
        return mask;
    }

    private static unsafe Tensor FeatureMaskTensor(
        IReadOnlyList<float> values, int rows, int patchArea)
    {
        Tensor mask = new Tensor(new TensorShape(rows, patchArea), DType.F32);
        float* destination = (float*)mask.DataPointer;
        for (int i = 0; i < values.Count; i++)
        {
            destination[i] = values[i];
        }
        return mask;
    }

    /// <summary>H3 trains preserved video content at a fixed near-clean condition strength, not at the sampler's
    /// current sigma. The seeded noise stays fixed for the whole request.</summary>
    internal static void BuildVideoMaskInjection(
        IBackend backend, Tensor injection, Tensor source, Tensor fixedNoise)
    {
        float sourceStrength = MiniMaxH3Schedule.VisualCondTimestep;
        backend.AffineMix(
            injection, source, fixedNoise, sourceStrength, 1f - sourceStrength);
    }

    /// <summary>Builds the DiT input <c>q = M*z + (1-M)*I</c> without mutating sampler state <c>z</c>.</summary>
    internal static void BuildMaskedModelInput(
        IBackend backend, Tensor modelInput, Tensor state, Tensor injection, Tensor tokenMask)
    {
        backend.Scale(modelInput, state, 1f);
        backend.MaskedAffineMixInPlace(
            modelInput, injection, null, tokenMask,
            sourceScale: 1f, noiseScale: 0f, layout: MaskBroadcastLayout.Rows);
    }

    /// <summary>Translates Comfy's masked x0 wrapper into Hartsy's positive-velocity Euler convention. First
    /// <c>Dmodel=q+sigma*V</c>, then raw <c>m</c> produces <c>D=m*Dmodel+(1-m)*S</c>, and the exact native-sigma
    /// Euler step is <c>zNext=(sigmaNext/sigma)*z+(1-sigmaNext/sigma)*D</c>. <paramref name="nextState"/> may be
    /// the same tensor as <paramref name="modelInput"/> because the latter is consumed before the final mix.</summary>
    internal static void AdvanceMaskedState(
        IBackend backend, Tensor nextState, Tensor denoisedScratch, Tensor state, Tensor modelInput,
        Tensor velocity, Tensor source, Tensor featureMask, MaskBroadcastLayout featureLayout,
        float nativeSigma, float nextNativeSigma)
    {
        if (!float.IsFinite(nativeSigma) || !float.IsFinite(nextNativeSigma)
            || nativeSigma <= 0f || nextNativeSigma < 0f || nextNativeSigma > nativeSigma)
        {
            throw new ArgumentOutOfRangeException(nameof(nativeSigma), nativeSigma,
                $"MiniMax-H3 masked Euler sigmas must satisfy 0 <= next <= current; got "
                + $"{nextNativeSigma} and {nativeSigma}.");
        }
        backend.AffineMix(denoisedScratch, modelInput, velocity, 1f, nativeSigma);
        backend.MaskedAffineMixInPlace(
            denoisedScratch, source, null, featureMask,
            sourceScale: 1f, noiseScale: 0f, layout: featureLayout);
        float stateStrength = nextNativeSigma / nativeSigma;
        backend.AffineMix(
            nextState, state, denoisedScratch, stateStrength, 1f - stateStrength);
    }

    /// <summary>Resolves strength windows once before sampling, keeping managed allocations out of the denoise loop.</summary>
    private static IReadOnlyList<MiniMaxH3FunControlCondition>?[] BuildControlSchedule(
        IReadOnlyList<MiniMaxH3FunControlCondition>? controls, int steps)
    {
        IReadOnlyList<MiniMaxH3FunControlCondition>?[] schedule =
            new IReadOnlyList<MiniMaxH3FunControlCondition>?[steps];
        if (controls is null || controls.Count == 0)
        {
            return schedule;
        }
        for (int step = 0; step < steps; step++)
        {
            List<MiniMaxH3FunControlCondition>? active = null;
            foreach (MiniMaxH3FunControlCondition control in controls)
            {
                if (control.IsActive(step, steps))
                {
                    (active ??= new List<MiniMaxH3FunControlCondition>()).Add(control);
                }
            }
            schedule[step] = active;
        }
        return schedule;
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
            throw new HartsyInferenceException(
                $"MiniMax-H3 layout expects {expectedRows} conditioning {stream} row(s), got {actual}.");
        }
        if (rows is not null && rows.Shape[1] != expectedWidth)
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 conditioning {stream} rows are {rows.Shape[1]} wide, expected {expectedWidth}.");
        }
    }

    /// <summary>Debug switches, read once at type init rather than per tensor: <see cref="Probe"/> and <see cref="Dump"/> are called several times per denoise step, so a per-call environment lookup would be wasted work.</summary>
    private static readonly bool ProbeEnabled = EngineKnobs.H3Probe.Value;

    private static readonly string? DumpDir = EngineKnobs.H3Dump.Value;

    /// <summary>Logs min/max/mean/rms under <c>HARTSY_H3_PROBE=1</c>; no-op otherwise.</summary>
    private static void Probe(string label, Tensor t)
    {
        if (!ProbeEnabled)
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
        catch (OutOfVramException ex)
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
    /// <see cref="OutOfVramException"/> during preload just means the phase falls
    /// back to the existing lazy per-op streaming path, never a failed generation.</summary>
    private static bool TryPreloadWeights(IBackend backend, string label, IEnumerable<Tensor> weights)
    {
        try
        {
            backend.PreloadWeights(weights);
            return true;
        }
        catch (OutOfVramException ex)
        {
            Logs.Warning($"[MiniMaxH3] {label} preload did not fit ({ex.Message}) — streaming per call.");
            return false;
        }
    }

    /// <summary>Writes the raw F32 tensor to <c>$HARTSY_H3_DUMP/&lt;name&gt;.bin</c> for reference comparison; no-op
    /// when the variable is unset.</summary>
    private static void Dump(string name, Tensor t)
    {
        string? dir = DumpDir;
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
        // The decoders keep their weights resident across generations when warm-placed, so disposal is the only
        // point that releases them — otherwise a model switch strands them on the placement card.
        if (VaeIsWarmPlaced)
        {
            VaeBackend.FreeWeights(_videoVae.EnumerateWeights());
            if (_audioVae is not null) { VaeBackend.FreeWeights(_audioVae.EnumerateWeights()); }
        }
        _transformer.Dispose();
        _pddAdapter?.Dispose();
        _videoVae.Dispose();
        _audioVae?.Dispose();
    }
}
