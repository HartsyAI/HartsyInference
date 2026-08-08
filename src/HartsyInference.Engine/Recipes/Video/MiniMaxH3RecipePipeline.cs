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
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed MiniMax-H3 pipeline driven from the native <see cref="VideoRequest"/>. H3 emits a stereo
/// soundtrack with every clip, so the result carries both streams.</summary>
public sealed unsafe class MiniMaxH3RecipePipeline : IVideoRecipePipeline
{
    /// <summary>Reference caps the model was trained under. A reference video's own soundtrack is capped separately
    /// from the standalone clips, as the reference node does, so both lists may carry three.</summary>
    private const int MaxReferenceImages = 9, MaxReferenceAudios = 3, MaxReferenceVideos = 3;

    /// <summary>Pixels per latent cell on H/W; a reference block's grid is stated in latent cells.</summary>
    private const int VaeSpatialRatio = 16;

    private readonly MiniMaxH3Pipeline _pipeline;
    private readonly MiniMaxH3Config _config;
    private readonly IBackend _backend;
    private readonly MiniMaxH3TextEncoder _textEncoder;
    private readonly Qwen2Tokenizer _tokenizer;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly MiniMaxH3VideoVaeEncoder? _videoVaeEncoder;
    private readonly MiniMaxH3AudioVaeEncoder? _audioVaeEncoder;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Takes ownership of the pipeline, the pre-encoded conditioning, and every loader backing the weights.
    /// The encoders are null for decode-only VAEs, which disables keyframe and reference conditioning respectively.</summary>
    public MiniMaxH3RecipePipeline(IBackend backend, MiniMaxH3Pipeline pipeline, MiniMaxH3Config config,
        MiniMaxH3TextEncoder textEncoder, Qwen2Tokenizer tokenizer, List<SafeTensorsLoader> loaders,
        MiniMaxH3VideoVaeEncoder? videoVaeEncoder = null, MiniMaxH3AudioVaeEncoder? audioVaeEncoder = null,
        MergedLoraStack? loraStack = null)
    {
        _backend = backend;
        _pipeline = pipeline;
        _config = config;
        _textEncoder = textEncoder;
        _tokenizer = tokenizer;
        _loaders = loaders;
        _videoVaeEncoder = videoVaeEncoder;
        _audioVaeEncoder = audioVaeEncoder;
        _loraStack = loraStack;
    }

    /// <summary>One keyframe resolved into everything the two conditioning paths need: the DiT's packed rows, the
    /// anchor that pins it to the clip's first or last frame, and the RGB the vision tower presents.</summary>
    private readonly record struct Keyframe(int FrameIndex, Tensor Rows, Tensor Rgb, int VisionTokens);

    /// <summary>One ref2va reference resolved for both paths. The rows land in the stream its block kind names, so the
    /// order these are produced in has to match the order the packed layout emits their segments. A soundtracked video
    /// carries two conditions — its <c>&lt;Audio j&gt;</c> label then its <c>&lt;Video k&gt;</c> — behind one block.</summary>
    private sealed record Reference(MiniMaxH3RefBlock Block, IReadOnlyList<MiniMaxH3TextEncoding.Condition> Conditions)
    {
        public Tensor? VideoRows { get; init; }
        public Tensor? AudioRows { get; init; }

        /// <summary>What the vision tower presents, one entry per spliced block: a <c>[3, H, W]</c> still for an image
        /// or a <c>[2, 3, H, W]</c> frame stack per video block. Empty for an audio reference, which is label-only.</summary>
        public IReadOnlyList<Tensor> Rgb { get; init; } = [];
    }

    /// <summary>A reference clip decoded, truncated onto the frame grid, and resized onto its canvas — everything both
    /// encode passes need, resolved before either VAE is made resident.</summary>
    private sealed record PreparedVideo(IReadOnlyList<byte[]> Frames, int Width, int Height, AudioClip? Soundtrack);

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
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
        int width = MiniMaxH3Geometry.Round(request.Width ?? 1344);
        int height = MiniMaxH3Geometry.Round(request.Height ?? 768);
        if (frames != requestedFrames || width != (request.Width ?? 1344) || height != (request.Height ?? 768))
        {
            Logs.Info($"[MiniMaxH3RecipePipeline] Geometry snapped to H3's grid: "
                + $"{request.Width}x{request.Height}x{requestedFrames}f -> {width}x{height}x{frames}f.");
        }

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        List<Keyframe> keyframes = [];
        List<Reference> references = [];
        try
        {
            CheckVramFeasibility(width, height, frames);
            EncodeKeyframes(request, width, height, frames, keyframes);
            EncodeReferences(request, width, height, frames, cancel, references);
            if (keyframes.Count > 0 && references.Count > 0)
            {
                // The layout restarts its position cursor for reference blocks, so keyframe and reference rows would
                // occupy overlapping coordinates. The two tasks also ship as separate checkpoints upstream.
                throw new ArgumentException(
                    "MiniMax-H3 cannot combine start/end frames with references — fl2va and ref2va are separate tasks.");
            }

            List<Tensor> videoRowParts = [.. keyframes.Select(k => k.Rows)];
            videoRowParts.AddRange(references.Where(r => r.VideoRows is not null).Select(r => r.VideoRows!));
            List<Tensor> audioRowParts = [.. references.Where(r => r.AudioRows is not null).Select(r => r.AudioRows!)];
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
                    Keyframes = keyframes.Count == 0
                        ? null
                        : keyframes.Select(k => new MiniMaxH3Keyframe { ResolvedFrameIndex = k.FrameIndex }).ToList(),
                    Refs = references.Count == 0 ? null : references.Select(r => r.Block).ToList(),
                    FrameCount = frames,
                    CondVideoRows = condVideoRows,
                    CondAudioRows = condAudioRows,
                };

                // Both lists walk references in presentation order and flatten per reference; grouping by kind here
                // would silently pair the wrong vision block with the wrong label.
                List<MiniMaxH3TextEncoding.Condition> conditions =
                    [.. keyframes.Select(k => ImageCondition(k.VisionTokens)), .. references.SelectMany(r => r.Conditions)];
                List<Tensor> visionInputs =
                    [.. keyframes.Select(k => k.Rgb), .. references.SelectMany(r => r.Rgb)];

                // Preload/free around the encode, as every other video recipe does: the encoder and the DiT cannot both
                // be device-resident on a 24 GB card. (This is hygiene, not the perf fix — measurement showed the
                // encoder's weights were never the thing occupying VRAM during denoise.)
                MiniMaxH3TextEncoder.Result encoded;
                _backend.PreloadWeights(_textEncoder.EnumerateWeights());
                try
                {
                    // Keyframes are presented to the vision tower exactly as reference images are — the reference
                    // labels them <Picture 1>/<Picture 2> ahead of the prompt — so the two conditioning paths agree.
                    encoded = _textEncoder.Encode(_backend, _tokenizer, request.Prompt,
                        conditions.Count == 0 ? null : conditions,
                        visionInputs.Count == 0 ? null : visionInputs);
                    _backend.Sync();
                }
                finally
                {
                    _backend.FreeWeights(_textEncoder.EnumerateWeights());
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
                k.Rows.Dispose();
                k.Rgb.Dispose();
            }
            foreach (Reference r in references)
            {
                DisposeReference(r);
            }
        }
    }

    /// <summary>Refuses instantly, before any encode/allocate work, when the requested geometry's activation floor
    /// cannot fit even with the DiT streamed per-op and nothing else resident — turning the mid-generation OOM a
    /// 481-frame/1280x704 request hit into an immediate, actionable error instead. Chunking
    /// (<see cref="MiniMaxH3ChunkPolicy"/>) cannot rescue a request past this floor, so it is checked ahead of any
    /// allocation rather than left to surface as an <see cref="OutOfVramException"/> mid-denoise; see
    /// <see cref="MiniMaxH3ActivationEstimate"/> for what the floor accounts for. When DiT sharding is active, the
    /// shard backend runs the exact same full-sequence forward for its own block range (only the WEIGHT range
    /// splits, not the sequence), so it needs the identical floor and is checked too — a split that leaves the
    /// smaller card's share too thin would otherwise only surface as a mid-denoise OOM on that backend.</summary>
    private void CheckVramFeasibility(int width, int height, int frames)
    {
        int videoLatentFrames = MiniMaxH3Geometry.VideoLatentFrames(frames);
        int audioLatentFrames = MiniMaxH3Geometry.AudioLatentFrames(frames);
        int latentH = height / VaeSpatialRatio, latentW = width / VaeSpatialRatio;
        int videoRows = videoLatentFrames * (latentH / 2) * (latentW / 2);
        int audioRows = audioLatentFrames * 2;
        // Text/reference rows are only known after encoding, but they're a small, bounded addition next to a
        // geometry large enough to be at risk here — a fixed conservative estimate keeps this check allocation-free.
        const int approxNonVideoRows = 512;
        int seq = approxNonVideoRows + videoRows + audioRows;
        long floorBytes = MiniMaxH3ActivationEstimate.EstimateFloorBytes(seq, _config, DType.F32);

        CheckOneBackend(_backend, "primary", floorBytes, _pipeline.EstimateResidentWeightBytes(), frames, width, height, seq);
        if (_pipeline.DitShardBackend is not null)
        {
            CheckOneBackend(_pipeline.DitShardBackend, "shard", floorBytes,
                _pipeline.EstimateShardResidentWeightBytes(), frames, width, height, seq);
        }
    }

    private static void CheckOneBackend(IBackend backend, string label, long floorBytes, long residentWeightBytes,
        int frames, int width, int height, int seq)
    {
        // Pooled cuMemFreeAsync reservations don't return to cuMemGetInfo's free count until trimmed — the same
        // staleness VramPlanner.TrimBeforeQuery exists to counteract for other families' checks.
        backend.TrimMemoryPool();
        (long freeBytes, _) = backend.GetVramInfo();
        if (freeBytes <= 0)
        {
            // GetVramInfo() defaults to (0, 0) on backends that don't report live VRAM (e.g. CPU) — nothing to check.
            return;
        }
        long availableForActivations = freeBytes - residentWeightBytes;
        if (floorBytes > availableForActivations)
        {
            throw new OutOfVramException(
                $"MiniMax-H3/{frames}f@{width}x{height} (seq~{seq}) cannot run on this device's {label} backend: it "
                + $"needs at least {Mb(floorBytes)} of activations and workspace on top of {Mb(residentWeightBytes)} "
                + $"of resident DiT weights, but only {Mb(freeBytes)} is free. Weight streaming cannot reduce this "
                + "— lower the resolution or frame count, use a device with more VRAM, or adjust the DiT shard "
                + "split.");
        }
    }

    private static string Mb(long bytes) => $"{bytes / (1024 * 1024)} MB";

    /// <summary>Encodes ref2va references in the order the presentation and the packed layout both expect: images,
    /// then videos, then standalone audio. Each becomes one <see cref="MiniMaxH3RefBlock"/> plus its presentation
    /// label(s). The work is phased so each VAE is made resident exactly once even though a soundtracked video needs
    /// both of them.</summary>
    private void EncodeReferences(VideoRequest request, int width, int height, int frameCount,
        CancellationToken cancel, List<Reference> into)
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
        foreach (ReferenceVideo video in videos)
        {
            prepared.Add(PrepareReferenceVideo(video, frameCount, cancel));
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
                _backend.PreloadWeights(_videoVaeEncoder!.EnumerateWeights());
                try
                {
                    foreach (ImageData image in images)
                    {
                        assembled.Add(EncodeReferenceImage(image, width, height));
                    }
                    for (int i = 0; i < prepared.Count; i++)
                    {
                        using Tensor latent = _videoVaeEncoder.EncodeRgbClip(
                            _backend, prepared[i].Frames, prepared[i].Width, prepared[i].Height);
                        videoRows[i] = MiniMaxH3Latents.PackVideo(latent, _config);
                        videoLatentT[i] = (int)latent.Shape[2];
                    }
                    _backend.Sync();
                }
                finally
                {
                    _backend.FreeWeights(_videoVaeEncoder!.EnumerateWeights());
                }
            }

            List<Reference> audioRefs = new List<Reference>(audios.Count);
            if (needsAudioVae)
            {
                _backend.PreloadWeights(_audioVaeEncoder!.EnumerateWeights());
                try
                {
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
                    _backend.Sync();
                }
                finally
                {
                    _backend.FreeWeights(_audioVaeEncoder!.EnumerateWeights());
                }
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

    /// <summary>Decodes a reference clip, truncates it onto the model's frame grid, and resizes it onto its canvas.
    /// Truncation runs before the resize: the discarded frames would otherwise be resampled for nothing, and a long
    /// HD clip is gigabytes of them.</summary>
    private static PreparedVideo PrepareReferenceVideo(ReferenceVideo reference, int frameCount, CancellationToken cancel)
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
        (int canvasWidth, int canvasHeight) = MiniMaxH3Geometry.RefVideoCanvas(decoded.Width, decoded.Height);

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

    /// <summary>Assembles a prepared clip into its reference block, presentation labels, and the 2 fps frame stacks the
    /// vision tower sees. The stack count must equal what <see cref="MiniMaxH3TextEncoding.VideoBlocks"/> produces —
    /// a mismatch only surfaces as the vision tower's token-count assertion once real weights are loaded.</summary>
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

    /// <summary>Scales a reference down (never up) to the generation's pixel area, keeping its own aspect — the
    /// reference's "match" policy. A reference is not the canvas, so it keeps its shape rather than being stretched.</summary>
    private Reference EncodeReferenceImage(ImageData image, int width, int height)
    {
        double scale = Math.Min(1.0, Math.Sqrt((double)width * height / ((double)image.Width * image.Height)));
        int tw = MiniMaxH3Geometry.Round((int)Math.Round(image.Width * scale));
        int th = MiniMaxH3Geometry.Round((int)Math.Round(image.Height * scale));
        byte[] rgb = VideoRecipeUtils.ResizeRgb24(image, tw, th);
        Tensor latent = _videoVaeEncoder!.EncodeRgbFrame(_backend, rgb, tw, th);
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

    /// <summary>VAE-encodes a clip to packed audio rows plus its latent length. Shared by standalone reference audio
    /// and by a reference video's soundtrack, which folds into that video's block instead of becoming its own.</summary>
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
        using Tensor latent = _audioVaeEncoder.Encode(_backend, wave);
        return (MiniMaxH3Latents.PackAudio(latent, _config), (int)latent.Shape[3]);
    }

    /// <summary>Encodes a standalone reference clip. It carries no vision block — the presentation is the
    /// <c>&lt;Audio j&gt;</c> label alone.</summary>
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

    /// <summary>VAE-encodes the start and end images into keyframe conditioning. The first frame is stretched to the
    /// canvas because it anchors the geometry; the last frame is cover-cropped so it does not distort what follows.</summary>
    private void EncodeKeyframes(VideoRequest request, int width, int height, int frames, List<Keyframe> into)
    {
        if (request.InitImage is null && request.VideoEndFrame is null)
        {
            return;
        }
        if (_videoVaeEncoder is null)
        {
            Logs.Warning("[MiniMaxH3RecipePipeline] A start/end frame was supplied but this VAE carries no encoder — "
                + "generating from the prompt alone.");
            return;
        }
        _backend.PreloadWeights(_videoVaeEncoder.EnumerateWeights());
        try
        {
            if (request.InitImage is not null)
            {
                into.Add(EncodeKeyframe(request.InitImage, width, height, 0));
            }
            if (request.VideoEndFrame is not null)
            {
                into.Add(EncodeKeyframe(request.VideoEndFrame, width, height, frames - 1));
            }
            _backend.Sync();
        }
        finally
        {
            _backend.FreeWeights(_videoVaeEncoder.EnumerateWeights());
        }
        Logs.Info($"[MiniMaxH3RecipePipeline] fl2va: {into.Count} keyframe(s) at frame "
            + string.Join(", ", into.Select(k => k.FrameIndex)) + ".");
    }

    private Keyframe EncodeKeyframe(ImageData image, int width, int height, int frameIndex)
    {
        byte[] rgb = VideoRecipeUtils.ResizeRgb24(image, width, height);
        Tensor latent = _videoVaeEncoder!.EncodeRgbFrame(_backend, rgb, width, height);
        try
        {
            return new Keyframe(frameIndex, MiniMaxH3Latents.PackVideo(latent, _config),
                RgbToTensor(rgb, width, height), _textEncoder.VisionTokenCount(height, width));
        }
        finally
        {
            latent.Dispose();
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
        _pipeline.Dispose();
        _textEncoder.Dispose();
        // After the transformer that reads them: the merged tensors are the DiT's weights, not copies.
        _loraStack?.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
