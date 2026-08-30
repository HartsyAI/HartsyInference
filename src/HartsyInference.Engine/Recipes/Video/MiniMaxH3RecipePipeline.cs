using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Audio;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Vision;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed MiniMax-H3 pipeline driven from the native <see cref="VideoRequest"/>. H3 emits a stereo soundtrack with every clip, so the result carries both streams.</summary>
public sealed unsafe class MiniMaxH3RecipePipeline : IVideoRecipePipeline
{
    /// <summary>Reference caps the model was trained under. A reference video's own soundtrack is capped separately from the standalone clips, as the reference node does, so both lists may carry three.</summary>
    private const int MaxReferenceImages = 9, MaxReferenceAudios = 3, MaxReferenceVideos = 3;

    /// <summary>Pixels per latent cell on H/W; a reference block's grid is stated in latent cells.</summary>
    private const int VaeSpatialRatio = 16;

    private readonly MiniMaxH3Pipeline _pipeline;
    private readonly MiniMaxH3Config _config;
    private readonly IBackend _backend;
    /// <summary>Where the Qwen3-VL encode runs; equal to <see cref="_backend"/> unless placement moved it.</summary>
    private readonly IBackend _textEncoderBackend;
    /// <summary>Where every VAE ENCODE runs (keyframes, references). The decodes are the pipeline's own <c>VaeBackend</c>, set from the same placement so both halves of the VAE land on one device.</summary>
    private readonly IBackend _vaeBackend;
    private readonly MiniMaxH3TextEncoder _textEncoder;
    private readonly Qwen2Tokenizer _tokenizer;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly MiniMaxH3VideoVaeEncoder? _videoVaeEncoder;
    private readonly MiniMaxH3AudioVaeEncoder? _audioVaeEncoder;
    private readonly MergedLoraStack? _loraStack;
    private readonly MergedLoraStack? _pddLoraStack;
    private readonly bool _supportsHybridConditioning;
    private readonly IReadOnlyDictionary<string, int> _funControlModelIndices;
    private int _disposed;

    /// <summary>Takes ownership of the pipeline, the pre-encoded conditioning, and every loader backing the weights. The encoders are null for decode-only VAEs, which disables keyframe and reference conditioning respectively.</summary>
    public MiniMaxH3RecipePipeline(IBackend backend, MiniMaxH3Pipeline pipeline, MiniMaxH3Config config,
        MiniMaxH3TextEncoder textEncoder, Qwen2Tokenizer tokenizer, List<SafeTensorsLoader> loaders,
        MiniMaxH3VideoVaeEncoder? videoVaeEncoder = null, MiniMaxH3AudioVaeEncoder? audioVaeEncoder = null,
        MergedLoraStack? loraStack = null, IBackend? textEncoderBackend = null, IBackend? vaeBackend = null)
        : this(backend, pipeline, config, textEncoder, tokenizer, loaders, false,
            videoVaeEncoder, audioVaeEncoder, loraStack, textEncoderBackend, vaeBackend, null, null)
    {
    }

    /// <summary>Takes ownership of all model components while explicitly binding whether their detected profile can combine guide and reference conditioning.</summary>
    internal MiniMaxH3RecipePipeline(IBackend backend, MiniMaxH3Pipeline pipeline, MiniMaxH3Config config,
        MiniMaxH3TextEncoder textEncoder, Qwen2Tokenizer tokenizer, List<SafeTensorsLoader> loaders,
        bool supportsHybridConditioning, MiniMaxH3VideoVaeEncoder? videoVaeEncoder = null,
        MiniMaxH3AudioVaeEncoder? audioVaeEncoder = null, MergedLoraStack? loraStack = null,
        IBackend? textEncoderBackend = null, IBackend? vaeBackend = null, MergedLoraStack? pddLoraStack = null,
        IReadOnlyDictionary<string, int>? funControlModelIndices = null)
    {
        _backend = backend;
        _textEncoderBackend = textEncoderBackend ?? backend;
        _vaeBackend = vaeBackend ?? backend;
        _pipeline = pipeline;
        _config = config;
        _textEncoder = textEncoder;
        _tokenizer = tokenizer;
        _loaders = loaders;
        _videoVaeEncoder = videoVaeEncoder;
        _audioVaeEncoder = audioVaeEncoder;
        _loraStack = loraStack;
        _pddLoraStack = pddLoraStack;
        _supportsHybridConditioning = supportsHybridConditioning;
        _funControlModelIndices = funControlModelIndices is null
            ? new Dictionary<string, int>(VideoArtifactPath.Comparer)
            : new Dictionary<string, int>(funControlModelIndices, VideoArtifactPath.Comparer);
    }

    /// <summary>Tier 3.8's <c>&lt;refcrop:&gt;</c> backend — a pipeline-owned cache (mirrors <c>ImagesService</c>'s own <c>ClipSegSegmenter</c> instance) so a prompt with no <c>&lt;refcrop:&gt;</c> tags never loads it.</summary>
    private readonly ClipSegSegmenter _clipSeg = new();

    /// <summary>One merged visual/audio guide after its signed anchor and both VAE streams have been resolved.</summary>
    private sealed record Keyframe
    {
        public required int FrameIndex { get; init; }
        public Tensor? VideoRows { get; init; }
        public Tensor? AudioRows { get; init; }
        public int VideoLatentFrames { get; init; }
        public int AudioLatentFrames { get; init; }

        /// <summary>Legacy start/end guides keep their existing Qwen vision presentation. Arbitrary guides are
        /// already represented by packed condition rows and therefore leave this null.</summary>
        public Tensor? VisionRgb { get; init; }

        public int VisionTokens { get; init; }
    }

    /// <summary>Canonical guide media before VAE encoding. Legacy and arbitrary inputs have already been merged.</summary>
    private sealed record PreparedGuide
    {
        public required int FrameIndex { get; init; }
        public IReadOnlyList<byte[]>? VisualFrames { get; init; }
        public AudioClip? Audio { get; init; }
        public bool PresentToVisionEncoder { get; init; }
    }

    private sealed class GuideBuilder
    {
        public required int FrameIndex { get; init; }
        public IReadOnlyList<byte[]>? VisualFrames { get; set; }
        public AudioClip? Audio { get; set; }
        public bool PresentToVisionEncoder { get; set; }
    }

    private sealed record PreparedDenoiseMasks : IDisposable
    {
        public float[]? VideoRows { get; init; }
        public float[]? VideoFeatureValues { get; init; }
        public Tensor? VideoSourceRows { get; init; }
        public float[]? AudioRows { get; init; }
        public float[]? AudioFeatureRows { get; init; }
        public Tensor? AudioSourceRows { get; init; }

        public void Dispose()
        {
            VideoSourceRows?.Dispose();
            AudioSourceRows?.Dispose();
        }
    }

    /// <summary>Keeps each conditioning encoder resident from its first actual use through the complete conditioning
    /// phase, then releases it before the DiT is made resident.</summary>
    private sealed class ConditioningWeightResidency(MiniMaxH3RecipePipeline owner) : IDisposable
    {
        private bool _videoLoaded;
        private bool _audioLoaded;

        /// <summary>Uploads the video encoder at most once for this generation.</summary>
        public void EnsureVideo()
        {
            if (_videoLoaded)
            {
                return;
            }
            MiniMaxH3VideoVaeEncoder encoder = owner._videoVaeEncoder
                ?? throw new InvalidOperationException(
                    "MiniMax-H3 visual conditioning requires a video VAE that carries its encoder half.");
            owner._vaeBackend.PreloadWeights(encoder.EnumerateWeights());
            _videoLoaded = true;
        }

        /// <summary>Uploads the audio encoder at most once for this generation.</summary>
        public void EnsureAudio()
        {
            if (_audioLoaded)
            {
                return;
            }
            MiniMaxH3AudioVaeEncoder encoder = owner._audioVaeEncoder
                ?? throw new InvalidOperationException(
                    "MiniMax-H3 audio conditioning requires an audio VAE that carries its encoder half.");
            owner._vaeBackend.PreloadWeights(encoder.EnumerateWeights());
            _audioLoaded = true;
        }

        public void Dispose()
        {
            if (_audioLoaded)
            {
                owner.ReleaseComponentWeights(owner._vaeBackend, owner._audioVaeEncoder!.EnumerateWeights());
            }
            if (_videoLoaded)
            {
                owner.ReleaseComponentWeights(owner._vaeBackend, owner._videoVaeEncoder!.EnumerateWeights());
            }
        }
    }

    /// <summary>One ref2va reference resolved for both paths. The rows land in the stream its block kind names, so the order these are produced in has to match the order the packed layout emits their segments. A soundtracked video carries two conditions — its <c>&lt;Audio j&gt;</c> label then its <c>&lt;Video k&gt;</c> — behind one block.</summary>
    private sealed record Reference(MiniMaxH3RefBlock Block, IReadOnlyList<MiniMaxH3TextEncoding.Condition> Conditions)
    {
        public Tensor? VideoRows { get; init; }
        public Tensor? AudioRows { get; init; }

        /// <summary>What the vision tower presents, one entry per spliced block: a <c>[3, H, W]</c> still for an image or a <c>[2, 3, H, W]</c> frame stack per video block. Empty for an audio reference, which is label-only.</summary>
        public IReadOnlyList<Tensor> Rgb { get; init; } = [];
    }

    /// <summary>A reference clip decoded, truncated onto the frame grid, and resized onto its canvas — everything both encode passes need, resolved before either VAE is made resident.</summary>
    private sealed record PreparedVideo(IReadOnlyList<byte[]> Frames, int Width, int Height, AudioClip? Soundtrack);

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        // H3's canonical joint solver is Euler over its native "normal" schedule. One DiT forward returns BOTH
        // stream velocities, and they are integrated over different deltas — video over -dSigma, audio over -dSigma
        // scaled by the schedule map's derivative — whereas ISampler.Step advances a single latent per evaluation.
        // Accept the canonical names that profile planning materializes, but refuse every alternative rather than
        // silently substituting it. Splitting an alternative solver into two samplers would double the forwards.
        if (!IsCanonicalSamplingSelection(request.Sampler, "euler")
            || !IsCanonicalSamplingSelection(request.Scheduler, "normal"))
        {
            throw new NotSupportedException(
                $"Sampler/schedule '{request.Sampler ?? request.Scheduler}' is not available on MiniMax-H3: one DiT "
                + "forward drives both the video and the audio latent on different schedules, which the single-latent "
                + "sampler seam cannot express. Use sampler 'euler' and schedule 'normal', or leave both unset.");
        }
        // Tier 3.8: <refcrop:N,query[,threshold]> auto-crops reference image N to a CLIPSeg-matched region before
        // it reaches EncodeReferences below. Must run before request.Prompt is read anywhere (line ~191's text
        // encode) — an un-stripped tag left in the prompt is exactly the base-prompt tag-leak class of bug Tier
        // 3.2 fixed. Identity path (same request instance) when the prompt carries no <refcrop:> tags at all.
        request = ReferenceCropResolver.Apply(request, _backend, _clipSeg, cancel);
        if (request.Fps is int requestedFps && requestedFps != MiniMaxH3Geometry.Fps)
        {
            // The model always denoises at MiniMaxH3Geometry.Fps — VideoService.GenerateAsync resolves the final
            // container fps as request.Fps ?? result.Fps ?? resolved.Fps, so a differing request value only changes
            // the muxed playback rate (slow/fast motion over the same generated frames), never the model's cadence.
            Logs.Warning($"[MiniMaxH3RecipePipeline] Requested fps {requestedFps} differs from H3's native "
                + $"{MiniMaxH3Geometry.Fps} — the video generates at {MiniMaxH3Geometry.Fps} fps and is muxed at "
                + $"{requestedFps} fps (slow/fast motion), not resampled.");
        }
        int requestedFrames = request.Frames ?? 124;
        // H3's grids are coarse and non-obvious: frames snap to 17k+5, latent frames are NOT frames/4, and each pixel
        // axis rounds to 32 (a multiple of 16 alone leaves an odd latent axis and the 2x2 patchifier drops its last
        // row/column). Audio length follows the ALIGNED frame count so the two streams end together.
        int frames = MiniMaxH3Geometry.AlignFrameCount(requestedFrames);
        int requestedWidth = request.Width ?? 1344;
        int requestedHeight = request.Height ?? 768;
        (int width, int height) = MiniMaxH3Geometry.ClampToMaxArea(requestedWidth, requestedHeight);
        if (frames != requestedFrames || width != requestedWidth || height != requestedHeight)
        {
            Logs.Info($"[MiniMaxH3RecipePipeline] Geometry snapped to H3's grid: "
                + $"{requestedWidth}x{requestedHeight}x{requestedFrames}f -> {width}x{height}x{frames}f.");
        }
        if ((long)requestedWidth * requestedHeight > MiniMaxH3Geometry.MaxPixels)
        {
            // Denoising above the trained area costs proportionally more compute for no quality gain, so the clamp
            // is a correctness fix rather than a memory one — the memory question is CheckVramFeasibility's.
            Logs.Warning($"[MiniMaxH3RecipePipeline] {requestedWidth}x{requestedHeight} is "
                + $"{(long)requestedWidth * requestedHeight / 1000}k pixels, above MiniMax-H3's trained area of "
                + $"{MiniMaxH3Geometry.MaxPixels / 1000}k — scaled to {width}x{height} preserving aspect.");
        }
        if (frames > MiniMaxH3Geometry.TrainedFrameEnvelope)
        {
            Logs.Warning($"[MiniMaxH3RecipePipeline] {frames} frames is "
                + $"{(double)frames / MiniMaxH3Geometry.Fps:F1} s, past MiniMax-H3's trained envelope of "
                + $"{MiniMaxH3Geometry.TrainedFrameEnvelope} frames (~{(double)MiniMaxH3Geometry.TrainedFrameEnvelope / MiniMaxH3Geometry.Fps:F0} s) "
                + "— generating anyway; motion coherence and audio sync may drift past that length.");
        }

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);

        List<Keyframe> keyframes = [];
        List<Reference> references = [];
        List<MiniMaxH3FunControlCondition> controls = [];
        try
        {
            CheckVramFeasibility(width, height, frames);
            using PreparedDenoiseMasks denoiseMasks = PrepareConditioning(
                request, width, height, frames, cancel, controls, keyframes, references);
            if (keyframes.Count > 0 && references.Count > 0 && !_supportsHybridConditioning)
            {
                throw new ArgumentException(
                    "This MiniMax-H3 profile cannot combine start/end guides with reference inputs. Select a "
                    + "Hybrid-capable checkpoint profile to use both conditioning families together.");
            }

            List<Tensor> videoRowParts =
                [.. keyframes.Where(k => k.VideoRows is not null).Select(k => k.VideoRows!)];
            videoRowParts.AddRange(references.Where(r => r.VideoRows is not null).Select(r => r.VideoRows!));
            List<Tensor> audioRowParts =
                [.. keyframes.Where(k => k.AudioRows is not null).Select(k => k.AudioRows!)];
            audioRowParts.AddRange(references.Where(r => r.AudioRows is not null).Select(r => r.AudioRows!));
            Tensor? condVideoRows = videoRowParts.Count == 0 ? null : ConcatRows(videoRowParts);
            Tensor? condAudioRows = audioRowParts.Count == 0 ? null : ConcatRows(audioRowParts);
            try
            {
                MiniMaxH3GenerationRequest inner = new MiniMaxH3GenerationRequest
                {
                    Width = width,
                    Height = height,
                    LatentFrames = MiniMaxH3Geometry.VideoLatentFrames(frames),
                    AudioLatentFrames = MiniMaxH3Geometry.AudioLatentFrames(frames),
                    Steps = request.Steps ?? 30,
                    Seed = (int)(RecipeRequestMapper.MapSeed(request.Seed) ?? 0),
                    SigmaShiftVideo = request.FlowShift ?? _config.SigmaShiftVideo,
                    SigmaShiftAudio = request.AudioFlowShift ?? _config.SigmaShiftAudio,
                    Sampler = request.Sampler ?? "euler",
                    CfgScale = request.CfgScale ?? 1f,
                    HybridProfile = _supportsHybridConditioning,
                    Keyframes = keyframes.Count == 0 ? null
                        : keyframes.Select(k => new MiniMaxH3Keyframe
                        {
                            ResolvedFrameIndex = k.FrameIndex,
                            VideoLatentFrames = k.VideoLatentFrames,
                            AudioLatentFrames = k.AudioLatentFrames,
                        }).ToList(),
                    Refs = references.Count == 0 ? null : references.Select(r => r.Block).ToList(),
                    FrameCount = frames,
                    CondVideoRows = condVideoRows,
                    CondAudioRows = condAudioRows,
                    VideoDenoiseMaskRows = denoiseMasks.VideoRows,
                    VideoDenoiseFeatureMaskValues = denoiseMasks.VideoFeatureValues,
                    VideoDenoiseSourceRows = denoiseMasks.VideoSourceRows,
                    AudioDenoiseMaskRows = denoiseMasks.AudioRows,
                    AudioDenoiseFeatureMaskRows = denoiseMasks.AudioFeatureRows,
                    AudioDenoiseSourceRows = denoiseMasks.AudioSourceRows,
                    Controls = controls.Count == 0 ? null : controls,
                };

                // Both lists walk references in presentation order and flatten per reference; grouping by kind here
                // would silently pair the wrong vision block with the wrong label.
                List<MiniMaxH3TextEncoding.Condition> conditions =
                    [.. keyframes.Where(k => k.VisionRgb is not null).Select(k => ImageCondition(k.VisionTokens)),
                        .. references.SelectMany(r => r.Conditions)];
                List<Tensor> visionInputs =
                    [.. keyframes.Where(k => k.VisionRgb is not null).Select(k => k.VisionRgb!),
                        .. references.SelectMany(r => r.Rgb)];

                // Preload/free around the encode, as every other video recipe does: the encoder and the DiT cannot both
                // be device-resident on a 24 GB card. (This is hygiene, not the perf fix — measurement showed the
                // encoder's weights were never the thing occupying VRAM during denoise.)
                MiniMaxH3TextEncoder.Result encoded;
                _textEncoderBackend.PreloadWeights(_textEncoder.EnumerateWeights());
                try
                {
                    // Keyframes are presented to the vision tower exactly as reference images are — the reference
                    // labels them <Picture 1>/<Picture 2> ahead of the prompt — so the two conditioning paths agree.
                    encoded = _textEncoder.Encode(_textEncoderBackend, _tokenizer, request.Prompt,
                        conditions.Count == 0 ? null : conditions, visionInputs.Count == 0 ? null : visionInputs);
                    // Load-bearing when the encoder sits on another device: the DiT's first read of these hidden
                    // states faults them back from here, and a fault does not await this device's stream.
                    _textEncoderBackend.Sync();
                }
                finally
                {
                    ReleaseComponentWeights(_textEncoderBackend, _textEncoder.EnumerateWeights());
                }

                MiniMaxH3Pipeline.Result result;
                try
                {
                    result = _pipeline.Generate(encoded.HiddenStates, inner, encoded.TagRuns, bridge);
                }
                finally
                {
                    encoded.HiddenStates.Dispose();
                }
                return Finish(result, request);
            }
            finally
            {
                condVideoRows?.Dispose();
                condAudioRows?.Dispose();
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[MiniMaxH3RecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            foreach (Keyframe k in keyframes)
            {
                k.VideoRows?.Dispose();
                k.AudioRows?.Dispose();
                k.VisionRgb?.Dispose();
            }
            foreach (Reference r in references)
            {
                DisposeReference(r);
            }
            foreach (MiniMaxH3FunControlCondition control in controls)
            {
                control.ControlRows.Dispose();
            }
        }
    }

    private static bool IsCanonicalSamplingSelection(string? selection, string canonical)
    {
        return string.IsNullOrWhiteSpace(selection)
            || string.Equals(selection.Trim(), canonical, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Refuses instantly, before any encode/allocate work, when the requested geometry's activation floor cannot fit even with the DiT streamed per-op and nothing else resident — turning the mid-generation OOM a 481-frame/1280x704 request hit into an immediate, actionable error instead. Chunking (<see cref="MiniMaxH3ChunkPolicy"/>) cannot rescue a request past this floor, so it is checked ahead of any allocation rather than left to surface as an <see cref="OutOfVramException"/> mid-denoise; see <see cref="MiniMaxH3ActivationEstimate"/> for what the floor accounts for. When DiT sharding is active, the shard backend runs the exact same full-sequence forward for its own block range (only the WEIGHT range splits, not the sequence), so it needs the identical floor and is checked too — a split that leaves the smaller card's share too thin would otherwise only surface as a mid-denoise OOM on that backend.</summary>
    private void CheckVramFeasibility(int width, int height, int frames)
    {
        int seq = SequenceLengthFor(width, height, frames);
        long floorBytes = MiniMaxH3ActivationEstimate.EstimateFloorBytes(seq, _config, DType.F32);

        // Every backend running block ranges needs the same per-block floor, so when sharding is on, BOTH have to
        // clear it. Report whichever is furthest short rather than whichever is checked first: the tightest one is
        // what the user actually has to fix, and on this box that is usually the smaller shard card.
        List<(IBackend Backend, string Label, long Weights)> stages =
            [(_backend, "primary", _pipeline.EstimateResidentWeightBytes())];
        if (_pipeline.DitShardBackend is not null)
        {
            stages.Add((_pipeline.DitShardBackend, "shard", _pipeline.EstimateShardResidentWeightBytes()));
        }
        OutOfVramException? worst = null;
        long worstDeficit = 0, worstBudget = 0;
        foreach ((IBackend backend, string label, long weights) in stages)
        {
            if (CheckOneBackend(backend, label, floorBytes, weights, frames, width, height, seq,
                    out long deficit, out long budget) is OutOfVramException failure && deficit > worstDeficit)
            {
                worst = failure;
                worstDeficit = deficit;
                worstBudget = budget;
            }
        }
        if (worst is not null)
        {
            // Name a length that WOULD work. "Lower the frame count" without a number leaves the user bisecting by
            // hand against a check that only answers yes/no.
            int feasible = LargestFeasibleFrameCount(width, height, frames, worstBudget);
            string advice = feasible > 0
                ? $" At {width}x{height} the longest clip that fits is {feasible} frames "
                    + $"({(double)feasible / MiniMaxH3Geometry.Fps:F1} s)."
                : $" Not even the shortest clip fits at {width}x{height} — the resolution is the limit here, "
                    + "not the length.";
            throw new OutOfVramException(worst.Message + advice);
        }
    }

    private static OutOfVramException? CheckOneBackend(IBackend backend, string label, long floorBytes,
        long residentWeightBytes, int frames, int width, int height, int seq, out long deficit, out long budget)
    {
        deficit = 0;
        budget = 0;
        // Pooled cuMemFreeAsync reservations don't return to cuMemGetInfo's free count until trimmed — the same
        // staleness VramPlanner.TrimBeforeQuery exists to counteract for other families' checks.
        backend.TrimMemoryPool();
        (long freeBytes, _) = backend.GetVramInfo();
        if (freeBytes <= 0)
        {
            // GetVramInfo() defaults to (0, 0) on backends that don't report live VRAM (e.g. CPU) — nothing to check.
            return null;
        }
        long availableForActivations = freeBytes - residentWeightBytes;
        budget = availableForActivations;
        if (floorBytes <= availableForActivations)
        {
            return null;
        }
        deficit = floorBytes - availableForActivations;
        return new OutOfVramException(
            $"MiniMax-H3/{frames}f@{width}x{height} (seq~{seq}) cannot run on this device's {label} backend: it "
            + $"needs at least {ByteFormat.Mb(floorBytes)} of activations and workspace on top of {ByteFormat.Mb(residentWeightBytes)} "
            + $"of resident DiT weights, but only {ByteFormat.Mb(freeBytes)} is free ({ByteFormat.Mb(deficit)} short). Weight streaming "
            + "cannot reduce this — lower the resolution or frame count, use a device with more VRAM, or adjust the "
            + "DiT shard split.");
    }

    /// <summary>Packed sequence length for a geometry, without allocating anything. Text/reference rows are only known after encoding, but they are a small bounded addition next to a geometry large enough to be at risk here, so a fixed conservative allowance keeps this usable as a pre-flight.</summary>
    private int SequenceLengthFor(int width, int height, int frames)
    {
        int latentH = height / VaeSpatialRatio, latentW = width / VaeSpatialRatio;
        int videoRows = MiniMaxH3Geometry.VideoLatentFrames(frames) * (latentH / 2) * (latentW / 2);
        int audioRows = MiniMaxH3Geometry.AudioLatentFrames(frames) * 2;
        const int approxNonVideoRows = 512;
        return approxNonVideoRows + videoRows + audioRows;
    }

    /// <summary>The longest clip that WOULD fit at this resolution, so the refusal can name a length that works instead of only the one that doesn't. Walks the 17k+5 grid down from the request rather than solving: the floor is not linear in frames (video and audio rows advance on different grids) and the search is at most a few dozen arithmetic steps with no allocation. Returns 0 when even the shortest clip doesn't fit, which means the resolution is the problem, not the length.</summary>
    private int LargestFeasibleFrameCount(int width, int height, int frames, long budgetBytes)
    {
        for (int candidate = frames - 17; candidate >= 5; candidate -= 17)
        {
            if (MiniMaxH3ActivationEstimate.EstimateFloorBytes(
                SequenceLengthFor(width, height, candidate), _config, DType.F32) <= budgetBytes)
            {
                return candidate;
            }
        }
        return 0;
    }


    /// <summary>Encodes every guide, mask source, reference, and control while each required VAE encoder is uploaded
    /// only once. The returned mask rows outlive this residency scope; model weights do not.</summary>
    private PreparedDenoiseMasks PrepareConditioning(VideoRequest request, int width, int height, int frames,
        CancellationToken cancel, List<MiniMaxH3FunControlCondition> controls, List<Keyframe> keyframes,
        List<Reference> references)
    {
        PreparedDenoiseMasks? masks = null;
        using ConditioningWeightResidency weights = new ConditioningWeightResidency(this);
        try
        {
            masks = PrepareDenoiseMasks(request, width, height, frames, cancel, weights);
            EncodeControls(request, width, height, frames, cancel, controls, weights);
            EncodeGuides(request, width, height, frames, cancel, keyframes, weights);
            EncodeReferences(request, width, height, frames, cancel, references, weights);
            return masks;
        }
        catch
        {
            masks?.Dispose();
            throw;
        }
    }

    /// <summary>Frees a component's weights after use — unless placement put that component on a device the DiT does not use, in which case they stay resident so the next generation skips the re-upload. The opt-in is the placement itself: on the primary the free is load-bearing (the DiT needs that room back), and off it there is nothing competing for the space. Warm weights still go on <c>FreeMemory()</c>, model switch, and disposal, which release every backend's weight set regardless. Only WEIGHTS are held — activations are freed normally, so nothing a later generation faults back to host is kept alive by this.</summary>
    private void ReleaseComponentWeights(IBackend backend, IEnumerable<Tensor> weights)
    {
        if (ReferenceEquals(backend, _backend))
        {
            backend.FreeWeights(weights);
        }
    }

    /// <summary>Encodes ref2va references in the order the presentation and the packed layout both expect: images, then videos, then standalone audio. Each becomes one <see cref="MiniMaxH3RefBlock"/> plus its presentation label(s). The work is phased so each VAE is made resident exactly once even though a soundtracked video needs both of them.</summary>
    private void EncodeReferences(VideoRequest request, int width, int height, int frameCount,
        CancellationToken cancel, List<Reference> into, ConditioningWeightResidency weights)
    {
        IReadOnlyList<ImageData> images = request.ReferenceImages ?? [];
        IReadOnlyList<ReferenceVideo> videos = request.ReferenceVideos ?? [];
        IReadOnlyList<AudioClip> audios = request.ReferenceAudios ?? [];
        if (images.Count == 0 && videos.Count == 0 && audios.Count == 0)
        {
            return;
        }
        if (images.Count > MaxReferenceImages)
        {
            throw new ArgumentException(
                $"MiniMax-H3 takes at most {MaxReferenceImages} reference images, got {images.Count}.");
        }
        if (videos.Count > MaxReferenceVideos)
        {
            throw new ArgumentException(
                $"MiniMax-H3 takes at most {MaxReferenceVideos} reference videos, got {videos.Count}.");
        }
        if (audios.Count > MaxReferenceAudios)
        {
            throw new ArgumentException(
                $"MiniMax-H3 takes at most {MaxReferenceAudios} standalone reference audio clips, got {audios.Count}.");
        }
        bool needsAudioVae = audios.Count > 0 || videos.Any(v => v.Audio is not null);
        if ((images.Count > 0 || videos.Count > 0) && _videoVaeEncoder is null)
        {
            throw new InvalidOperationException(
                "Reference images and videos need a video VAE that carries its encoder half.");
        }
        if (needsAudioVae && _audioVaeEncoder is null)
        {
            throw new InvalidOperationException("Reference audio needs an audio VAE that carries its encoder half.");
        }

        List<PreparedVideo> prepared = new List<PreparedVideo>(videos.Count);
        VideoReferenceSizing referenceSizing = request.ReferenceSizing ?? VideoReferenceSizing.Native;
        foreach (ReferenceVideo video in videos)
        {
            prepared.Add(PrepareReferenceVideo(video, frameCount, width, height, referenceSizing, cancel));
        }

        // Nothing reaches the caller's list until the whole set is assembled, so every partial result is disposed here
        // rather than leaking on a mid-phase failure.
        List<Reference> assembled = new List<Reference>(images.Count + prepared.Count + audios.Count);
        Tensor?[] videoRows = new Tensor?[prepared.Count];
        int[] videoLatentT = new int[prepared.Count];
        Tensor?[] soundtrackRows = new Tensor?[prepared.Count];
        int[] soundtrackT = new int[prepared.Count];
        try
        {
            if (images.Count > 0 || prepared.Count > 0)
            {
                weights.EnsureVideo();
                foreach (ImageData image in images)
                {
                    assembled.Add(EncodeReferenceImage(image, width, height, referenceSizing));
                }
                for (int i = 0; i < prepared.Count; i++)
                {
                    using Tensor latent = _videoVaeEncoder!.EncodeRgbClip(
                        _vaeBackend, prepared[i].Frames, prepared[i].Width, prepared[i].Height);
                    videoRows[i] = MiniMaxH3Latents.PackVideo(latent, _config);
                    videoLatentT[i] = (int)latent.Shape[2];
                }
                _vaeBackend.Sync();
            }

            List<Reference> audioRefs = new List<Reference>(audios.Count);
            if (needsAudioVae)
            {
                weights.EnsureAudio();
                for (int i = 0; i < prepared.Count; i++)
                {
                    if (prepared[i].Soundtrack is null)
                    {
                        continue;
                    }
                    (Tensor rows, int refAudioT) = EncodeAudioRows(prepared[i].Soundtrack!);
                    soundtrackRows[i] = rows;
                    soundtrackT[i] = refAudioT;
                }
                foreach (AudioClip clip in audios)
                {
                    audioRefs.Add(EncodeReferenceAudio(clip));
                }
                _vaeBackend.Sync();
            }

            for (int i = 0; i < prepared.Count; i++)
            {
                Reference reference = BuildVideoReference(
                    prepared[i], videoRows[i]!, videoLatentT[i], soundtrackRows[i], soundtrackT[i]);
                videoRows[i] = null;
                soundtrackRows[i] = null;
                assembled.Add(reference);
            }
            assembled.AddRange(audioRefs);
        }
        catch
        {
            foreach (Reference reference in assembled)
            {
                DisposeReference(reference);
            }
            foreach (Tensor? rows in videoRows)
            {
                rows?.Dispose();
            }
            foreach (Tensor? rows in soundtrackRows)
            {
                rows?.Dispose();
            }
            throw;
        }

        into.AddRange(assembled);
        Logs.Info($"[MiniMaxH3RecipePipeline] ref2va: {images.Count} image(s), {prepared.Count} video(s), "
            + $"{audios.Count} standalone audio clip(s).");
    }

    private static void DisposeReference(Reference reference)
    {
        reference.VideoRows?.Dispose();
        reference.AudioRows?.Dispose();
        foreach (Tensor rgb in reference.Rgb)
        {
            rgb.Dispose();
        }
    }

    /// <summary>Decodes a reference clip, truncates it onto the model's frame grid, and resizes it onto its canvas. Truncation runs before the resize: the discarded frames would otherwise be resampled for nothing, and a long HD clip is gigabytes of them.</summary>
    private static PreparedVideo PrepareReferenceVideo(ReferenceVideo reference, int frameCount,
        int targetWidth, int targetHeight, VideoReferenceSizing sizing, CancellationToken cancel)
    {
        FfmpegProcessDecoder decoder = new FfmpegProcessDecoder();
        FfmpegProcessDecoder.Result decoded =
            decoder.DecodeAsync(reference.Video.Data, reference.Video.Format, cancel).GetAwaiter().GetResult();
        int kept = Math.Min(decoded.Frames.Count, frameCount);
        if (kept < 5)
        {
            throw new ArgumentException(
                $"A MiniMax-H3 reference video needs at least 5 frames (~0.2 s at 24 fps); got {kept}.");
        }
        kept = MiniMaxH3Geometry.SnapFrameCountDown(kept);
        (int canvasWidth, int canvasHeight) = ReferenceCanvas(
            decoded.Width, decoded.Height, targetWidth, targetHeight, sizing);

        List<byte[]> resized = new List<byte[]>(kept);
        for (int i = 0; i < kept; i++)
        {
            cancel.ThrowIfCancellationRequested();
            ImageData frame = new ImageData { Rgb = decoded.Frames[i], Width = decoded.Width, Height = decoded.Height };
            resized.Add(VideoRecipeUtils.ResizeRgb24(frame, canvasWidth, canvasHeight));
        }
        decoded.Frames.Clear();
        Logs.Info($"[MiniMaxH3RecipePipeline] Reference clip {decoded.Width}x{decoded.Height} -> "
            + $"{canvasWidth}x{canvasHeight}, {kept} frame(s).");
        return new PreparedVideo(resized, canvasWidth, canvasHeight, reference.Audio);
    }

    /// <summary>Assembles a prepared clip into its reference block, presentation labels, and the 2 fps frame stacks the vision tower sees. The stack count must equal what <see cref="MiniMaxH3TextEncoding.VideoBlocks"/> produces — a mismatch only surfaces as the vision tower's token-count assertion once real weights are loaded.</summary>
    private Reference BuildVideoReference(PreparedVideo video, Tensor videoRows, int latentT,
        Tensor? audioRows, int refAudioT)
    {
        IReadOnlyList<int> sampled = MiniMaxH3Geometry.RefVideoSampleIndices(video.Frames.Count);
        int tokensPerBlock = _textEncoder.VisionTokenCount(video.Height, video.Width);
        List<MiniMaxH3TextEncoding.Condition> conditions = new List<MiniMaxH3TextEncoding.Condition>(2);
        if (audioRows is not null)
        {
            // The soundtrack's <Audio j> label is emitted before its <Video k>, so the audio ordinal increments first.
            conditions.Add(MiniMaxH3TextEncoding.Audio());
        }
        conditions.Add(MiniMaxH3TextEncoding.Video(sampled.Count, tokensPerBlock));

        int padded = sampled.Count + (sampled.Count % 2);
        List<Tensor> stacks = new List<Tensor>(padded / 2);
        for (int i = 0; i < padded; i += 2)
        {
            byte[] first = video.Frames[sampled[i]];
            byte[] second = video.Frames[sampled[Math.Min(i + 1, sampled.Count - 1)]];
            stacks.Add(RgbPairToTensor(first, second, video.Width, video.Height));
        }
        return new Reference(
            new MiniMaxH3RefBlock
            {
                Kind = audioRows is not null ? "video_audio" : "video",
                LatentT = latentT,
                LatentH = video.Height / VaeSpatialRatio,
                LatentW = video.Width / VaeSpatialRatio,
                RefAudioT = refAudioT,
            },
            conditions)
        {
            VideoRows = videoRows,
            AudioRows = audioRows,
            Rgb = stacks,
        };
    }

    /// <summary>Resolves the profile-controlled reference canvas. Match-target scales down (never up) to the
    /// generation area while native uses H3's trained reference canvas.</summary>
    internal static (int Width, int Height) ReferenceCanvas(int sourceWidth, int sourceHeight,
        int targetWidth, int targetHeight, VideoReferenceSizing sizing)
    {
        if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sourceWidth), "Reference and target dimensions must be positive.");
        }
        if (sizing == VideoReferenceSizing.Native)
        {
            return MiniMaxH3Geometry.RefVideoCanvas(sourceWidth, sourceHeight);
        }
        if (sizing != VideoReferenceSizing.MatchTarget)
        {
            throw new ArgumentOutOfRangeException(nameof(sizing), sizing, "Unknown MiniMax-H3 reference sizing policy.");
        }
        if (sourceWidth < MiniMaxH3Geometry.CanvasMultiple || sourceHeight < MiniMaxH3Geometry.CanvasMultiple)
        {
            throw new ArgumentOutOfRangeException(nameof(sourceWidth),
                $"MiniMax-H3 reference axes must each be at least {MiniMaxH3Geometry.CanvasMultiple} pixels.");
        }
        if (targetWidth < MiniMaxH3Geometry.CanvasMultiple || targetHeight < MiniMaxH3Geometry.CanvasMultiple)
        {
            throw new ArgumentOutOfRangeException(nameof(targetWidth),
                $"MiniMax-H3 target axes must each be at least {MiniMaxH3Geometry.CanvasMultiple} pixels.");
        }
        double scale = Math.Min(1.0,
            Math.Sqrt((double)targetWidth * targetHeight / ((double)sourceWidth * sourceHeight)));
        int width = MiniMaxH3Geometry.Floor(sourceWidth * scale);
        int height = MiniMaxH3Geometry.Floor(sourceHeight * scale);
        long targetArea = (long)targetWidth * targetHeight;

        // Flooring itself cannot exceed the scaled area, but Floor deliberately clamps an axis to one 32-pixel
        // patch. For a very wide or tall reference that minimum can push the result back above target area
        // (32x1024 matched to 32x32 used to become 32x160). Reduce the other axis on-grid; neither axis may fall
        // below one patch and neither can exceed its source because scale is capped at one.
        while ((long)width * height > targetArea
            && (width > MiniMaxH3Geometry.CanvasMultiple || height > MiniMaxH3Geometry.CanvasMultiple))
        {
            if (width == MiniMaxH3Geometry.CanvasMultiple)
            {
                height -= MiniMaxH3Geometry.CanvasMultiple;
            }
            else if (height == MiniMaxH3Geometry.CanvasMultiple)
            {
                width -= MiniMaxH3Geometry.CanvasMultiple;
            }
            else if ((double)width / sourceWidth >= (double)height / sourceHeight)
            {
                width -= MiniMaxH3Geometry.CanvasMultiple;
            }
            else
            {
                height -= MiniMaxH3Geometry.CanvasMultiple;
            }
        }
        return (width, height);
    }

    /// <summary>VAE-encodes one reference image at the profile-controlled canvas while preserving its aspect.</summary>
    private Reference EncodeReferenceImage(ImageData image, int width, int height, VideoReferenceSizing sizing)
    {
        (int tw, int th) = ReferenceCanvas(image.Width, image.Height, width, height, sizing);
        byte[] rgb = VideoRecipeUtils.ResizeRgb24(image, tw, th);
        Tensor latent = _videoVaeEncoder!.EncodeRgbFrame(_vaeBackend, rgb, tw, th);
        try
        {
            return new Reference(
                new MiniMaxH3RefBlock { Kind = "image", LatentH = th / VaeSpatialRatio, LatentW = tw / VaeSpatialRatio },
                [MiniMaxH3TextEncoding.Image(_textEncoder.VisionTokenCount(th, tw))])
            {
                VideoRows = MiniMaxH3Latents.PackVideo(latent, _config),
                Rgb = [RgbToTensor(rgb, tw, th)],
            };
        }
        finally
        {
            latent.Dispose();
        }
    }

    /// <summary>VAE-encodes a clip to packed audio rows plus its latent length. Shared by standalone reference audio and by a reference video's soundtrack, which folds into that video's block instead of becoming its own.</summary>
    private (Tensor Rows, int RefAudioT) EncodeAudioRows(AudioClip clip)
    {
        (float[] left, float[] right) = AudioClipCodec.DecodeStereo(clip, _audioVaeEncoder!.Config.SampleRate);
        using Tensor wave = new Tensor(new TensorShape(1, 2, left.Length), DType.F32);
        float* wp = (float*)wave.DataPointer;
        for (int i = 0; i < left.Length; i++)
        {
            wp[i] = left[i];
            wp[left.Length + i] = right[i];
        }
        using Tensor latent = _audioVaeEncoder.Encode(_vaeBackend, wave);
        return (MiniMaxH3Latents.PackAudio(latent, _config), (int)latent.Shape[3]);
    }

    /// <summary>Encodes a standalone reference clip. It carries no vision block — the presentation is the <c>&lt;Audio j&gt;</c> label alone.</summary>
    private Reference EncodeReferenceAudio(AudioClip clip)
    {
        (Tensor rows, int refAudioT) = EncodeAudioRows(clip);
        return new Reference(
            new MiniMaxH3RefBlock { Kind = "audio", RefAudioT = refAudioT },
            [MiniMaxH3TextEncoding.Audio()])
        {
            AudioRows = rows,
        };
    }

    private static VideoGenerationResult Finish(MiniMaxH3Pipeline.Result result, VideoRequest request)
    {
        AudioBuffer audio = AudioBuffer.FromChannels(result.Audio, result.AudioSampleRate);
        Logs.Info($"[MiniMaxH3RecipePipeline] {result.Frames.Length} frames {result.Width}x{result.Height}"
            + (audio.IsEmpty ? "." : $" plus a {audio.SampleRate} Hz {audio.ChannelCount}ch soundtrack."));
        return VideoRecipeUtils.ToResult(result.Frames, result.Width, result.Height, request,
            audio.IsEmpty ? null : audio);
    }

    private static MiniMaxH3TextEncoding.Condition ImageCondition(int visionTokens) =>
        new MiniMaxH3TextEncoding.Condition
        {
            Kind = MiniMaxH3TextEncoding.ConditionKind.Image,
            Blocks = [new MiniMaxH3TextEncoding.VisionBlock(visionTokens)],
        };

    /// <summary>Resizes request masks into native target-row order and VAE-encodes explicit preservation sources.
    /// White masks return no rows and therefore skip both source encoding and every sampler-loop mask operation.</summary>
    private PreparedDenoiseMasks PrepareDenoiseMasks(
        VideoRequest request, int width, int height, int frames, CancellationToken cancel,
        ConditioningWeightResidency weights)
    {
        if ((request.VideoDenoiseMask is not null || request.AudioDenoiseMask is not null)
            && request.Controls?.Any(static control => control.Kind == VideoControlKind.Inpaint) == true)
        {
            throw new ArgumentException(
                "MiniMax-H3 sampler denoise masks cannot run together with Fun ControlNet inpainting.",
                nameof(request));
        }

        if (request.VideoDenoiseMask is VideoDenoiseMask videoMask)
        {
            if ((videoMask.MaskImage is null) == (videoMask.MaskVideo is null))
            {
                throw new ArgumentException(
                    "MiniMax-H3 VideoDenoiseMask must provide exactly one of maskImage or maskVideo.",
                    nameof(request));
            }
            if (videoMask.SourceImage is not null && videoMask.SourceVideo is not null)
            {
                throw new ArgumentException(
                    "MiniMax-H3 VideoDenoiseMask sourceImage and sourceVideo are mutually exclusive.",
                    nameof(request));
            }
        }

        int latentT = MiniMaxH3Geometry.VideoLatentFrames(frames);
        int latentH = height / VaeSpatialRatio;
        int latentW = width / VaeSpatialRatio;
        int audioT = MiniMaxH3Geometry.AudioLatentFrames(frames);
        float[]? videoMaskRows = null;
        float[]? videoFeatureMaskValues = null;
        if (request.VideoDenoiseMask is not null)
        {
            videoMaskRows = PrepareVideoMaskRows(request.VideoDenoiseMask, latentT, latentH, latentW,
                cancel, out videoFeatureMaskValues);
        }
        float[]? audioMaskRows = null;
        float[]? audioFeatureMaskRows = null;
        if (request.AudioDenoiseMask is not null)
        {
            audioMaskRows = MiniMaxH3Masking.ResampleAudioMask(request.AudioDenoiseMask.Values,
                request.AudioDenoiseMask.Rate, audioT, out audioFeatureMaskRows);
        }

        Tensor? videoSourceRows = null;
        Tensor? audioSourceRows = null;
        try
        {
            if (videoMaskRows is not null)
            {
                VideoDenoiseMask mask = request.VideoDenoiseMask!;
                if (mask.SourceImage is null && mask.SourceVideo is null)
                {
                    throw new ArgumentException(
                        "A MiniMax-H3 video mask with preserved rows requires sourceImage or sourceVideo.",
                        nameof(request));
                }
                if (_videoVaeEncoder is null)
                {
                    throw new InvalidOperationException(
                        "MiniMax-H3 video denoise masks require a video VAE that carries its encoder half.");
                }
                IReadOnlyList<byte[]> sourceFrames = PrepareVideoMaskSource(mask, width, height, frames, cancel);
                weights.EnsureVideo();
                using Tensor latent = _videoVaeEncoder.EncodeRgbClip(
                    _vaeBackend, sourceFrames, width, height);
                videoSourceRows = MiniMaxH3Latents.PackVideo(latent, _config);
                _vaeBackend.Sync();
                if (videoSourceRows.Shape[0] != videoMaskRows.Length)
                {
                    throw new InvalidOperationException(
                        $"MiniMax-H3 mask source encoded to {videoSourceRows.Shape[0]} video rows, but the target "
                        + $"mask has {videoMaskRows.Length} rows.");
                }
            }

            if (audioMaskRows is not null)
            {
                AudioDenoiseMask mask = request.AudioDenoiseMask!;
                if (mask.Source is null)
                {
                    throw new ArgumentException(
                        "A MiniMax-H3 audio mask with preserved rows requires source audio.", nameof(request));
                }
                if (_audioVaeEncoder is null)
                {
                    throw new InvalidOperationException(
                        "MiniMax-H3 audio denoise masks require an audio VAE that carries its encoder half.");
                }
                weights.EnsureAudio();
                audioSourceRows = EncodeAudioMaskSource(mask.Source, audioT);
                _vaeBackend.Sync();
                if (audioSourceRows.Shape[0] != audioMaskRows.Length)
                {
                    throw new InvalidOperationException(
                        $"MiniMax-H3 mask source encoded to {audioSourceRows.Shape[0]} audio rows, but the target "
                        + $"mask has {audioMaskRows.Length} rows.");
                }
            }

            PreparedDenoiseMasks prepared = new PreparedDenoiseMasks
            {
                VideoRows = videoMaskRows,
                VideoFeatureValues = videoFeatureMaskValues,
                VideoSourceRows = videoSourceRows,
                AudioRows = audioMaskRows,
                AudioFeatureRows = audioFeatureMaskRows,
                AudioSourceRows = audioSourceRows,
            };
            videoSourceRows = null;
            audioSourceRows = null;
            return prepared;
        }
        finally
        {
            videoSourceRows?.Dispose();
            audioSourceRows?.Dispose();
        }
    }

    private static float[]? PrepareVideoMaskRows(VideoDenoiseMask mask, int latentT, int latentH, int latentW,
        CancellationToken cancel, out float[]? featureMaskValues)
    {
        List<byte[]> grayscale;
        if (mask.MaskImage is not null)
        {
            grayscale = [FeatureImaging.ResizeGrayscale(mask.MaskImage, latentW, latentH)];
        }
        else if (mask.MaskVideo is not null)
        {
            FfmpegProcessDecoder decoder = new FfmpegProcessDecoder();
            FfmpegProcessDecoder.Result decoded = decoder.DecodeAsync(mask.MaskVideo.Data, mask.MaskVideo.Format,
                maxFrames: null, scaleWidth: latentW, scaleHeight: latentH, cancel).GetAwaiter().GetResult();
            if (decoded.Frames.Count == 0)
            {
                throw new ArgumentException("MiniMax-H3 video mask clip decoded to zero frames.", nameof(mask));
            }
            grayscale = new List<byte[]>(decoded.Frames.Count);
            foreach (byte[] frame in decoded.Frames)
            {
                cancel.ThrowIfCancellationRequested();
                grayscale.Add(FeatureImaging.ResizeGrayscale(new ImageData
                {
                    Rgb = frame,
                    Width = latentW,
                    Height = latentH,
                }, latentW, latentH));
            }
            decoded.Frames.Clear();
        }
        else
        {
            throw new ArgumentException("MiniMax-H3 VideoDenoiseMask must provide maskImage or maskVideo.",
                nameof(mask));
        }

        float[] latentMask = new float[checked(latentT * latentH * latentW)];
        int plane = checked(latentH * latentW);
        for (int frame = 0; frame < latentT; frame++)
        {
            double position = grayscale.Count == 1 || latentT == 1 ? 0.0
                : frame * (grayscale.Count - 1.0) / (latentT - 1.0);
            int leftFrame = (int)Math.Floor(position);
            int rightFrame = Math.Min(grayscale.Count - 1, leftFrame + 1);
            float fraction = (float)(position - leftFrame);
            byte[] left = grayscale[leftFrame];
            byte[] right = grayscale[rightFrame];
            int offset = frame * plane;
            for (int i = 0; i < plane; i++)
            {
                latentMask[offset + i] = (left[i] + (right[i] - left[i]) * fraction) / 255f;
            }
        }
        return MiniMaxH3Masking.PackVideoMaskRows(latentMask, latentT, latentH, latentW,
            out featureMaskValues, patchHeight: 2, patchWidth: 2);
    }

    private static IReadOnlyList<byte[]> PrepareVideoMaskSource(
        VideoDenoiseMask mask, int width, int height, int frames, CancellationToken cancel)
    {
        if (mask.SourceImage is not null)
        {
            byte[] still = VideoRecipeUtils.ResizeRgb24(mask.SourceImage, width, height);
            byte[][] tiled = new byte[frames][];
            Array.Fill(tiled, still);
            return tiled;
        }
        if (mask.SourceVideo is null)
        {
            throw new ArgumentException("MiniMax-H3 video mask source is missing.", nameof(mask));
        }

        FfmpegProcessDecoder decoder = new FfmpegProcessDecoder();
        FfmpegProcessDecoder.Result decoded = decoder.DecodeAsync(mask.SourceVideo.Data, mask.SourceVideo.Format,
            maxFrames: frames, scaleWidth: width, scaleHeight: height, cancel).GetAwaiter().GetResult();
        if (decoded.Frames.Count == 0)
        {
            throw new ArgumentException("MiniMax-H3 video mask source decoded to zero frames.", nameof(mask));
        }
        List<byte[]> sourceFrames = new List<byte[]>(frames);
        for (int i = 0; i < frames; i++)
        {
            sourceFrames.Add(decoded.Frames[Math.Min(i, decoded.Frames.Count - 1)]);
        }
        decoded.Frames.Clear();
        return sourceFrames;
    }

    private unsafe Tensor EncodeAudioMaskSource(AudioClip source, int targetAudioT)
    {
        (float[] left, float[] right) = AudioClipCodec.DecodeStereo(source, _audioVaeEncoder!.Config.SampleRate);
        int samples = checked(targetAudioT * _audioVaeEncoder.SamplesPerLatentFrame);
        using Tensor waveform = new Tensor(new TensorShape(1, 2, samples), DType.F32);
        float* destination = (float*)waveform.DataPointer;
        int copied = Math.Min(samples, Math.Min(left.Length, right.Length));
        for (int i = 0; i < copied; i++)
        {
            destination[i] = left[i];
            destination[samples + i] = right[i];
        }
        using Tensor latent = _audioVaeEncoder.Encode(_vaeBackend, waveform);
        int encodedT = checked((int)latent.Shape[3]);
        Tensor rows = MiniMaxH3Latents.PackAudio(latent, _config);
        if (encodedT == targetAudioT)
        {
            return rows;
        }
        if (encodedT < targetAudioT)
        {
            rows.Dispose();
            throw new InvalidOperationException(
                $"MiniMax-H3 audio-mask source encoded to {encodedT} frames, expected {targetAudioT}.");
        }
        Tensor cropped = CropChannelMajorRows(rows, encodedT, targetAudioT);
        rows.Dispose();
        return cropped;
    }

    /// <summary>Translates legacy start/end inputs and arbitrary signed anchors into one merged guide table, then
    /// VAE-encodes each modality in a single residency phase. Visual and audio payloads at the same frame merge;
    /// duplicate payloads of the same modality fail before weights are loaded.</summary>
    private void EncodeGuides(VideoRequest request, int width, int height, int frames,
        CancellationToken cancel, List<Keyframe> into, ConditioningWeightResidency weights)
    {
        IReadOnlyList<PreparedGuide> guides = PrepareGuides(request, width, height, frames, cancel);
        if (guides.Count == 0)
        {
            return;
        }
        bool hasVisual = guides.Any(static guide => guide.VisualFrames is not null);
        bool hasAudio = guides.Any(static guide => guide.Audio is not null);
        if (hasVisual && _videoVaeEncoder is null)
        {
            throw new InvalidOperationException(
                "MiniMax-H3 visual guides require a video VAE that carries its encoder half.");
        }
        if (hasAudio && _audioVaeEncoder is null)
        {
            throw new InvalidOperationException(
                "MiniMax-H3 audio guides require an audio VAE that carries its encoder half.");
        }

        Tensor?[] videoRows = new Tensor?[guides.Count];
        Tensor?[] audioRows = new Tensor?[guides.Count];
        Tensor?[] visionRgb = new Tensor?[guides.Count];
        int[] videoLatentFrames = new int[guides.Count];
        int[] audioLatentFrames = new int[guides.Count];
        try
        {
            if (hasVisual)
            {
                weights.EnsureVideo();
                for (int i = 0; i < guides.Count; i++)
                {
                    IReadOnlyList<byte[]>? visual = guides[i].VisualFrames;
                    if (visual is null)
                    {
                        continue;
                    }
                    using Tensor latent = visual.Count == 1
                        ? _videoVaeEncoder!.EncodeRgbFrame(_vaeBackend, visual[0], width, height)
                        : _videoVaeEncoder!.EncodeRgbClip(_vaeBackend, visual, width, height);
                    videoRows[i] = MiniMaxH3Latents.PackVideo(latent, _config);
                    videoLatentFrames[i] = checked(((int)latent.Shape[2] + _config.PatchT - 1) / _config.PatchT);
                    if (guides[i].PresentToVisionEncoder)
                    {
                        visionRgb[i] = RgbToTensor(visual[0], width, height);
                    }
                }
                _vaeBackend.Sync();
            }

            if (hasAudio)
            {
                weights.EnsureAudio();
                int targetAudioT = MiniMaxH3Geometry.AudioLatentFrames(frames);
                for (int i = 0; i < guides.Count; i++)
                {
                    if (guides[i].Audio is null)
                    {
                        continue;
                    }
                    (Tensor encodedRows, int encodedT) = EncodeAudioRows(guides[i].Audio!);
                    int remaining = MiniMaxH3Masking.GuideAudioLatentFrames(
                        targetAudioT, guides[i].FrameIndex);
                    int keep = Math.Min(encodedT, remaining);
                    if (keep <= 0)
                    {
                        encodedRows.Dispose();
                        throw new ArgumentException(
                            $"MiniMax-H3 audio guide at frame {guides[i].FrameIndex} starts after the target's "
                            + "remaining audio duration.");
                    }
                    if (keep == encodedT)
                    {
                        audioRows[i] = encodedRows;
                    }
                    else
                    {
                        audioRows[i] = CropChannelMajorRows(encodedRows, encodedT, keep);
                        encodedRows.Dispose();
                    }
                    audioLatentFrames[i] = keep;
                }
                _vaeBackend.Sync();
            }

            for (int i = 0; i < guides.Count; i++)
            {
                into.Add(new Keyframe
                {
                    FrameIndex = guides[i].FrameIndex,
                    VideoRows = videoRows[i],
                    AudioRows = audioRows[i],
                    VideoLatentFrames = videoLatentFrames[i],
                    AudioLatentFrames = audioLatentFrames[i],
                    VisionRgb = visionRgb[i],
                    VisionTokens = visionRgb[i] is null ? 0 : _textEncoder.VisionTokenCount(height, width),
                });
                videoRows[i] = null;
                audioRows[i] = null;
                visionRgb[i] = null;
            }
        }
        finally
        {
            foreach (Tensor? rows in videoRows)
            {
                rows?.Dispose();
            }
            foreach (Tensor? rows in audioRows)
            {
                rows?.Dispose();
            }
            foreach (Tensor? rgb in visionRgb)
            {
                rgb?.Dispose();
            }
        }

        Logs.Info($"[MiniMaxH3RecipePipeline] guides: {into.Count} merged anchor(s) at frame "
            + string.Join(", ", into.Select(static guide => guide.FrameIndex)) + ".");
    }

    private static IReadOnlyList<PreparedGuide> PrepareGuides(VideoRequest request, int width, int height,
        int frames, CancellationToken cancel)
    {
        SortedDictionary<int, GuideBuilder> merged = new SortedDictionary<int, GuideBuilder>();
        if (request.InitImage is not null)
        {
            AddVisual(0, [FitLegacyGuideFrame(request.InitImage, width, height)], true);
        }
        if (request.VideoEndFrame is not null)
        {
            AddVisual(frames - 1, [FitLegacyGuideFrame(request.VideoEndFrame, width, height)], true);
        }

        foreach (VideoGuide guide in request.Guides ?? [])
        {
            cancel.ThrowIfCancellationRequested();
            int frameIndex = MiniMaxH3Masking.ResolveFrameIndex(guide.FrameIndex, frames);
            if (guide.Image is not null && guide.Video is not null)
            {
                throw new ArgumentException(
                    $"MiniMax-H3 guide at frame {guide.FrameIndex} cannot carry both image and video payloads.",
                    nameof(request));
            }
            if (guide.Image is null && guide.Video is null && guide.Audio is null)
            {
                throw new ArgumentException(
                    $"MiniMax-H3 guide at frame {guide.FrameIndex} has no visual or audio payload.",
                    nameof(request));
            }
            if (guide.Image is not null)
            {
                AddVisual(frameIndex, [VideoRecipeUtils.FitGuideFrame(guide.Image, width, height, guide.FitMode)], false);
            }
            else if (guide.Video is not null)
            {
                FfmpegProcessDecoder decoder = new FfmpegProcessDecoder();
                FfmpegProcessDecoder.Result decoded = decoder.DecodeAsync(
                    guide.Video.Data, guide.Video.Format, cancel).GetAwaiter().GetResult();
                int keep = MiniMaxH3Masking.GuideFrameCount(decoded.Frames.Count);
                List<byte[]> visual = new List<byte[]>(keep);
                for (int i = 0; i < keep; i++)
                {
                    cancel.ThrowIfCancellationRequested();
                    visual.Add(VideoRecipeUtils.FitGuideFrame(new ImageData
                    {
                        Rgb = decoded.Frames[i],
                        Width = decoded.Width,
                        Height = decoded.Height,
                    }, width, height, guide.FitMode));
                }
                decoded.Frames.Clear();
                AddVisual(frameIndex, visual, false);
            }
            if (guide.Audio is not null)
            {
                GuideBuilder builder = Get(frameIndex);
                if (builder.Audio is not null)
                {
                    throw new ArgumentException(
                        $"Multiple MiniMax-H3 audio guide payloads resolve to frame {frameIndex}.",
                        nameof(request));
                }
                builder.Audio = guide.Audio;
            }
        }

        return merged.Values.Select(static builder => new PreparedGuide
        {
            FrameIndex = builder.FrameIndex,
            VisualFrames = builder.VisualFrames,
            Audio = builder.Audio,
            PresentToVisionEncoder = builder.PresentToVisionEncoder,
        }).ToArray();

        GuideBuilder Get(int frameIndex)
        {
            if (!merged.TryGetValue(frameIndex, out GuideBuilder? builder))
            {
                builder = new GuideBuilder { FrameIndex = frameIndex };
                merged.Add(frameIndex, builder);
            }
            return builder;
        }

        void AddVisual(int frameIndex, IReadOnlyList<byte[]> visual, bool presentToVisionEncoder)
        {
            MiniMaxH3Masking.ValidateGuideFrameSpan(frameIndex, visual.Count, frames);
            GuideBuilder builder = Get(frameIndex);
            if (builder.VisualFrames is not null)
            {
                throw new ArgumentException(
                    $"Multiple MiniMax-H3 visual guide payloads resolve to frame {frameIndex}.", nameof(request));
            }
            builder.VisualFrames = visual;
            builder.PresentToVisionEncoder = presentToVisionEncoder;
        }
    }

    /// <summary>Preserves the original H3 init/end behavior: both legacy anchors were directly resized to the
    /// target, even when their aspect ratio differed. Arbitrary guides use their explicit fit mode instead.</summary>
    internal static byte[] FitLegacyGuideFrame(ImageData image, int width, int height) =>
        VideoRecipeUtils.FitGuideFrame(image, width, height, VideoGuideFitMode.Stretch);

    private static unsafe Tensor CropChannelMajorRows(Tensor rows, int sourceFrames, int keepFrames)
    {
        int channels = checked((int)rows.Shape[0] / sourceFrames);
        int features = checked((int)rows.Shape[1]);
        Tensor cropped = new Tensor(new TensorShape((long)channels * keepFrames, features), DType.F32);
        float* source = (float*)rows.DataPointer;
        float* destination = (float*)cropped.DataPointer;
        long bytesPerChannel = (long)keepFrames * features * sizeof(float);
        for (int channel = 0; channel < channels; channel++)
        {
            Buffer.MemoryCopy(source + (long)channel * sourceFrames * features,
                destination + (long)channel * keepFrames * features, bytesPerChannel, bytesPerChannel);
        }
        return cropped;
    }

    /// <summary>Turns already-annotated request clips into the exact Fun branch row contract. Model weights are
    /// deduplicated at recipe construction; this phase deliberately retains one condition per request stream so
    /// strengths and normalized windows stay independent.</summary>
    private void EncodeControls(VideoRequest request, int width, int height, int frames,
        CancellationToken cancel, List<MiniMaxH3FunControlCondition> into,
        ConditioningWeightResidency weights)
    {
        IReadOnlyList<VideoControl> requested = request.Controls ?? [];
        if (requested.Count == 0)
        {
            return;
        }

        List<(VideoControl Control, int ModelIndex)> active = new List<(VideoControl, int)>(requested.Count);
        foreach (VideoControl control in requested)
        {
            int modelIndex = ValidateControl(control);
            if (control.Strength != 0.0)
            {
                active.Add((control, modelIndex));
            }
        }
        if (active.Count == 0)
        {
            return;
        }
        if (_videoVaeEncoder is null)
        {
            throw new InvalidOperationException(
                "MiniMax-H3 Fun ControlNet requires a video VAE that carries its encoder half.");
        }

        int targetLatentT = MiniMaxH3Geometry.VideoLatentFrames(frames);
        int targetLatentH = height / VaeSpatialRatio;
        int targetLatentW = width / VaeSpatialRatio;
        weights.EnsureVideo();
        foreach ((VideoControl control, int modelIndex) in active)
        {
            cancel.ThrowIfCancellationRequested();
            IReadOnlyList<byte[]> controlFrames = DecodeControlFrames(
                control.Video, width, height, frames, cancel);
            using Tensor controlLatent = _videoVaeEncoder.EncodeRgbClip(
                _vaeBackend, controlFrames, width, height);
            ValidateControlLatent(controlLatent, targetLatentT, targetLatentH, targetLatentW, "control");

            Tensor? visibility = null;
            Tensor? sourceLatent = null;
            Tensor? rows = null;
            try
            {
                if (control.Kind == VideoControlKind.Inpaint)
                {
                    visibility = DecodeVisibilityLatent(control.VisibilityMask!, frames,
                        targetLatentT, targetLatentH, targetLatentW, cancel);
                    IReadOnlyList<byte[]> sourceFrames = DecodeControlFrames(
                        control.MaskedSource!, width, height, frames, cancel);
                    sourceLatent = _videoVaeEncoder.EncodeRgbClip(
                        _vaeBackend, sourceFrames, width, height);
                    ValidateControlLatent(sourceLatent, targetLatentT, targetLatentH, targetLatentW,
                        "masked source");
                }

                rows = MiniMaxH3FunControlInputBuilder.Build(controlLatent, visibility, sourceLatent,
                    _config.PatchT, _config.PatchH, _config.PatchW);
                into.Add(new MiniMaxH3FunControlCondition
                {
                    ModelIndex = modelIndex,
                    ControlRows = rows,
                    Strength = (float)control.Strength,
                    Start = (float)control.Start,
                    End = (float)control.End,
                    IsInpaint = control.Kind == VideoControlKind.Inpaint,
                });
                rows = null;
            }
            finally
            {
                rows?.Dispose();
                sourceLatent?.Dispose();
                visibility?.Dispose();
            }
        }
        _vaeBackend.Sync();

        Logs.Info($"[MiniMaxH3RecipePipeline] Fun ControlNet: {into.Count} active stream(s), "
            + $"{into.Select(static condition => condition.ModelIndex).Distinct().Count()} deduplicated branch(es).");
    }

    private int ValidateControl(VideoControl control)
    {
        ArgumentNullException.ThrowIfNull(control);
        if (string.IsNullOrWhiteSpace(control.Model))
        {
            throw new ArgumentException("A MiniMax-H3 control stream has no model path.", nameof(control));
        }
        string modelPath = VideoArtifactPath.Canonicalize(control.Model);
        if (!_funControlModelIndices.TryGetValue(modelPath, out int modelIndex))
        {
            throw new InvalidOperationException(
                $"MiniMax-H3 control model '{modelPath}' was not registered by the resolved VideoPlan. "
                + "Plan and construct the request with the same control inputs.");
        }
        if (control.Video.Data.Length == 0)
        {
            throw new ArgumentException("A MiniMax-H3 control stream has an empty video payload.", nameof(control));
        }
        if (!VideoControlValidation.IsValidStrength(control.Strength))
        {
            throw new ArgumentOutOfRangeException(nameof(control), control.Strength,
                "MiniMax-H3 control strength must be finite, non-negative, and representable as F32.");
        }
        if (!VideoControlValidation.IsValidWindow(control.Start, control.End))
        {
            throw new ArgumentOutOfRangeException(nameof(control),
                "MiniMax-H3 control windows must satisfy 0 <= start <= end <= 1.");
        }
        bool inpaint = control.Kind == VideoControlKind.Inpaint;
        if (inpaint && (control.VisibilityMask is null || control.MaskedSource is null))
        {
            throw new ArgumentException(
                "MiniMax-H3 Fun inpaint requires both visibility-mask and masked-source videos.", nameof(control));
        }
        if (!inpaint && (control.VisibilityMask is not null || control.MaskedSource is not null))
        {
            throw new ArgumentException(
                "Visibility-mask and masked-source payloads are valid only for a MiniMax-H3 Inpaint control.",
                nameof(control));
        }
        if (control.VisibilityMask is not null && control.VisibilityMask.Data.Length == 0
            || control.MaskedSource is not null && control.MaskedSource.Data.Length == 0)
        {
            throw new ArgumentException("MiniMax-H3 Fun inpaint payloads cannot be empty.", nameof(control));
        }
        return modelIndex;
    }

    /// <summary>Fits an already-preprocessed stream to the target canvas, truncating long clips and holding the
    /// last frame for short clips. Control kind never enters this path: it is provenance metadata, not a weight or
    /// pixel transform selector.</summary>
    private static IReadOnlyList<byte[]> DecodeControlFrames(VideoClip clip, int width, int height, int frames,
        CancellationToken cancel)
    {
        FfmpegProcessDecoder decoder = new FfmpegProcessDecoder();
        FfmpegProcessDecoder.Result decoded = decoder.DecodeAsync(clip.Data, clip.Format,
            maxFrames: frames, scaleWidth: width, scaleHeight: height, cancel).GetAwaiter().GetResult();
        if (decoded.Frames.Count == 0)
        {
            throw new ArgumentException("MiniMax-H3 control video decoded to zero frames.", nameof(clip));
        }
        byte[] last = decoded.Frames[^1];
        while (decoded.Frames.Count < frames)
        {
            decoded.Frames.Add(last);
        }
        return decoded.Frames;
    }

    /// <summary>Resizes white-is-visible mask values to the control latent grid with continuous temporal
    /// interpolation. Spatial scaling happens in ffmpeg before bytes cross the pipe.</summary>
    private static Tensor DecodeVisibilityLatent(VideoClip clip, int pixelFrames, int latentFrames,
        int latentHeight, int latentWidth, CancellationToken cancel)
    {
        FfmpegProcessDecoder decoder = new FfmpegProcessDecoder();
        FfmpegProcessDecoder.Result decoded = decoder.DecodeAsync(clip.Data, clip.Format,
            maxFrames: pixelFrames, scaleWidth: latentWidth, scaleHeight: latentHeight, cancel)
            .GetAwaiter().GetResult();
        if (decoded.Frames.Count == 0)
        {
            throw new ArgumentException("MiniMax-H3 inpaint visibility video decoded to zero frames.", nameof(clip));
        }

        Tensor visibility = new Tensor(
            new TensorShape([1L, 1, latentFrames, latentHeight, latentWidth]), DType.F32);
        float* output = (float*)visibility.DataPointer;
        int pixels = checked(latentHeight * latentWidth);
        for (int targetFrame = 0; targetFrame < latentFrames; targetFrame++)
        {
            double sourcePosition = latentFrames == 1 ? 0.0
                : targetFrame * (pixelFrames - 1.0) / (latentFrames - 1.0);
            int lowIndex = Math.Min((int)Math.Floor(sourcePosition), decoded.Frames.Count - 1);
            int highIndex = Math.Min(lowIndex + 1, decoded.Frames.Count - 1);
            float mix = (float)(sourcePosition - Math.Floor(sourcePosition));
            byte[] low = decoded.Frames[lowIndex];
            byte[] high = decoded.Frames[highIndex];
            float* frame = output + (long)targetFrame * pixels;
            for (int pixel = 0; pixel < pixels; pixel++)
            {
                int offset = pixel * 3;
                float lowValue = (low[offset] + low[offset + 1] + low[offset + 2]) / (3f * 255f);
                float highValue = (high[offset] + high[offset + 1] + high[offset + 2]) / (3f * 255f);
                frame[pixel] = lowValue + (highValue - lowValue) * mix;
            }
        }
        return visibility;
    }

    private void ValidateControlLatent(Tensor latent, int frames, int height, int width, string label)
    {
        TensorShape expected = new TensorShape([1L, _config.LatentsDim, frames, height, width]);
        if (latent.DType != DType.F32 || latent.Shape != expected)
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 Fun {label} encoded to {latent.DType} {latent.Shape}, expected F32 {expected}.");
        }
    }

    /// <summary>Interleaved RGB24 to the <c>[3, H, W]</c> tensor in [0, 1] the vision tower takes.</summary>
    private static unsafe Tensor RgbToTensor(byte[] rgb, int width, int height)
    {
        Tensor outT = new Tensor(new TensorShape(3, height, width), DType.F32);
        float* p = (float*)outT.DataPointer;
        long plane = (long)width * height;
        for (long pix = 0; pix < plane; pix++)
        {
            for (int c = 0; c < 3; c++)
            {
                p[c * plane + pix] = rgb[pix * 3 + c] / 255f;
            }
        }
        return outT;
    }

    /// <summary>Two interleaved-RGB24 frames as the <c>[2, 3, H, W]</c> stack in [0, 1] that fills one temporal patch.</summary>
    private static unsafe Tensor RgbPairToTensor(byte[] first, byte[] second, int width, int height)
    {
        Tensor outT = new Tensor(new TensorShape(2, 3, height, width), DType.F32);
        float* p = (float*)outT.DataPointer;
        long plane = (long)width * height;
        byte[][] pair = [first, second];
        for (int f = 0; f < pair.Length; f++)
        {
            byte[] rgb = pair[f];
            float* frame = p + f * 3 * plane;
            for (long pix = 0; pix < plane; pix++)
            {
                for (int c = 0; c < 3; c++)
                {
                    frame[c * plane + pix] = rgb[pix * 3 + c] / 255f;
                }
            }
        }
        return outT;
    }

    private static unsafe Tensor ConcatRows(IReadOnlyList<Tensor> parts)
    {
        long rows = 0;
        foreach (Tensor p in parts)
        {
            rows += p.Shape[0];
        }
        Tensor outT = new Tensor(new TensorShape(rows, parts[0].Shape[1]), DType.F32);
        float* dst = (float*)outT.DataPointer;
        long cursor = 0;
        foreach (Tensor p in parts)
        {
            long count = p.ElementCount;
            Buffer.MemoryCopy((void*)p.DataPointer, dst + cursor, count * 4, count * 4);
            cursor += count;
        }
        return outT;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        // Disposal can race with cache eviction/shutdown, and the placement backend's FreeWeights path is not an
        // ownership-neutral operation. Guard the complete transitive teardown, not only the inner diffusion
        // pipeline, so a second call cannot release warm-placed encoder weights twice.
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;

        // Warm-placed components skip the per-generation free, so this is the ONLY thing that hands their device
        // weights back. Before the host tensors are disposed, since those tensors are the device cache's keys.
        if (!ReferenceEquals(_textEncoderBackend, _backend))
        {
            _textEncoderBackend.FreeWeights(_textEncoder.EnumerateWeights());
        }
        if (!ReferenceEquals(_vaeBackend, _backend))
        {
            if (_videoVaeEncoder is not null) { _vaeBackend.FreeWeights(_videoVaeEncoder.EnumerateWeights()); }
            if (_audioVaeEncoder is not null) { _vaeBackend.FreeWeights(_audioVaeEncoder.EnumerateWeights()); }
        }
        if (!ReferenceEquals(_textEncoderBackend, _backend) || !ReferenceEquals(_vaeBackend, _backend))
        {
            Logs.Info("[MiniMaxH3RecipePipeline] Released warm-placed component weights from their placement "
                + "backends — those devices are back to baseline.");
        }
        _pipeline.Dispose();
        _textEncoder.Dispose();
        _audioVaeEncoder?.Dispose();
        _clipSeg.Dispose();
        // After the transformer that reads them: the merged tensors are the DiT's weights, not copies.
        _loraStack?.Dispose();
        _pddLoraStack?.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
