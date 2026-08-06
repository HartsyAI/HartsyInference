using HartsyInference.Core.Backends;
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
using HartsyInference.Video.Pipelines;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed MiniMax-H3 pipeline driven from the native <see cref="VideoRequest"/>. H3 emits a stereo
/// soundtrack with every clip, so the result carries both streams.</summary>
public sealed unsafe class MiniMaxH3RecipePipeline : IVideoRecipePipeline
{
    /// <summary>Reference caps the model was trained under.</summary>
    private const int MaxReferenceImages = 9, MaxReferenceAudios = 3;

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
    /// order these are produced in has to match the order the packed layout emits their segments.</summary>
    private sealed record Reference(MiniMaxH3RefBlock Block, MiniMaxH3TextEncoding.Condition Condition)
    {
        public Tensor? VideoRows { get; init; }
        public Tensor? AudioRows { get; init; }

        /// <summary>The RGB the vision tower presents; null for an audio reference, which is label-only.</summary>
        public Tensor? Rgb { get; init; }
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        int fps = request.Fps ?? MiniMaxH3Geometry.Fps;
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
            EncodeKeyframes(request, width, height, frames, keyframes);
            EncodeReferences(request, width, height, references);
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

                List<MiniMaxH3TextEncoding.Condition> conditions =
                    [.. keyframes.Select(k => ImageCondition(k.VisionTokens)), .. references.Select(r => r.Condition)];
                List<Tensor> visionInputs =
                    [.. keyframes.Select(k => k.Rgb), .. references.Where(r => r.Rgb is not null).Select(r => r.Rgb!)];

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
                r.VideoRows?.Dispose();
                r.AudioRows?.Dispose();
                r.Rgb?.Dispose();
            }
        }
    }

    /// <summary>Encodes ref2va references in the order the presentation and the packed layout both expect: images
    /// first, then standalone audio. Each becomes one <see cref="MiniMaxH3RefBlock"/> plus its presentation label.</summary>
    private void EncodeReferences(VideoRequest request, int width, int height, List<Reference> into)
    {
        IReadOnlyList<ImageData> images = request.ReferenceImages ?? [];
        IReadOnlyList<AudioClip> audios = request.ReferenceAudios ?? [];
        if (request.ReferenceVideos is { Count: > 0 })
        {
            // A video reference presents as timestamped 2-frame vision blocks, and the Qwen3-VL processor currently
            // takes one frame per block — the DiT side is ready, the presentation side is not.
            throw new NotSupportedException(
                "MiniMax-H3 reference videos are not supported yet: the vision processor takes a single frame per "
                + "block, while a video reference presents paired frames. Use reference images or audio.");
        }
        if (images.Count == 0 && audios.Count == 0)
        {
            return;
        }
        if (images.Count > MaxReferenceImages)
        {
            throw new ArgumentException(
                $"MiniMax-H3 takes at most {MaxReferenceImages} reference images, got {images.Count}.");
        }
        if (audios.Count > MaxReferenceAudios)
        {
            throw new ArgumentException(
                $"MiniMax-H3 takes at most {MaxReferenceAudios} reference audio clips, got {audios.Count}.");
        }
        if (images.Count > 0 && _videoVaeEncoder is null)
        {
            throw new InvalidOperationException("Reference images need a video VAE that carries its encoder half.");
        }
        if (audios.Count > 0 && _audioVaeEncoder is null)
        {
            throw new InvalidOperationException("Reference audio needs an audio VAE that carries its encoder half.");
        }

        if (images.Count > 0)
        {
            _backend.PreloadWeights(_videoVaeEncoder!.EnumerateWeights());
            try
            {
                foreach (ImageData image in images)
                {
                    into.Add(EncodeReferenceImage(image, width, height));
                }
                _backend.Sync();
            }
            finally
            {
                _backend.FreeWeights(_videoVaeEncoder!.EnumerateWeights());
            }
        }
        if (audios.Count > 0)
        {
            _backend.PreloadWeights(_audioVaeEncoder!.EnumerateWeights());
            try
            {
                foreach (AudioClip clip in audios)
                {
                    into.Add(EncodeReferenceAudio(clip));
                }
                _backend.Sync();
            }
            finally
            {
                _backend.FreeWeights(_audioVaeEncoder!.EnumerateWeights());
            }
        }
        Logs.Info($"[MiniMaxH3RecipePipeline] ref2va: {images.Count} image(s), {audios.Count} audio clip(s).");
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
                MiniMaxH3TextEncoding.Image(_textEncoder.VisionTokenCount(th, tw)))
            {
                VideoRows = MiniMaxH3Latents.PackVideo(latent, _config),
                Rgb = RgbToTensor(rgb, tw, th),
            };
        }
        finally
        {
            latent.Dispose();
        }
    }

    /// <summary>Encodes a reference clip to audio rows. It carries no vision block — the presentation is the
    /// <c>&lt;Audio j&gt;</c> label alone.</summary>
    private Reference EncodeReferenceAudio(AudioClip clip)
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
        return new Reference(
            new MiniMaxH3RefBlock { Kind = "audio", RefAudioT = (int)latent.Shape[3] },
            MiniMaxH3TextEncoding.Audio())
        {
            AudioRows = MiniMaxH3Latents.PackAudio(latent, _config),
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
