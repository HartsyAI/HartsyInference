using HartsyInference.Core.Backends;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.Models.MossVl;
using HartsyInference.LLM.Multimodal.Streaming;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.Engine.Services;

/// <summary>A MOSS-VL model loaded for the text service: the text side (<see cref="MossVlModel"/>) and the vision tower, both on the
/// slot's backend. A GGUF carries its own tower; a checkpoint folder's tower comes from the same shards.</summary>
internal sealed class MossVlRuntime : IDisposable
{
    private readonly IBackend _backend;
    private int _disposed;

    private MossVlRuntime(MossVlModel model, MossVlVisionTower tower, IBackend backend, bool lowVramQuant)
    {
        Model = model;
        Tower = tower;
        _backend = backend;
        LowVramQuant = lowVramQuant;
    }

    public MossVlModel Model { get; }

    public MossVlVisionTower Tower { get; }

    /// <summary>Whether the decoder was loaded with the low-VRAM quantized Linear path.</summary>
    public bool LowVramQuant { get; }

    /// <summary>Loads both halves from <paramref name="path"/> (a GGUF or a checkpoint folder) for <paramref name="backend"/> and makes
    /// their weights resident.</summary>
    public static MossVlRuntime Load(string path, IBackend backend, bool lowVramQuant)
    {
        MossVlModel model = MossVlModel.Open(path, new MossVlLoadOptions { Backend = backend, LowVramQuant = lowVramQuant });
        MossVlVisionTower? tower = null;
        try
        {
            tower = model.IsGguf ? MossVlVisionTower.OpenGguf(path, backend) : MossVlVisionTower.Open(path, backend);
            // Resident up front: without it a device backend uploads each weight on the op that first reads it.
            backend.PreloadWeightGroups(model.Decoder.Model.EnumerateWeightGroups(includeRedundantSplits: false));
            backend.PreloadWeights(tower.Encoder.EnumerateWeights());
            return new MossVlRuntime(model, tower, backend, lowVramQuant);
        }
        catch
        {
            tower?.Dispose();
            model.Dispose();
            throw;
        }
    }

    /// <summary>Answers <paramref name="question"/> about <paramref name="images"/> (in prompt order). The images share the model's
    /// batch pixel budget, run through the tower one at a time, and feed one reply streamed through <paramref name="onDelta"/>.</summary>
    public MossVlGeneration Generate(string question, IReadOnlyList<ImageData> images, int maxTokens, SamplingOptions sampling,
        Action<string>? onDelta, CancellationToken cancel) =>
        Generate(question, images, video: [], videoSampling: null, maxTokens, sampling, onDelta, cancel);

    /// <summary>Answers <paramref name="question"/> about still <paramref name="images"/> and a <paramref name="video"/> of timed frames
    /// (already sampled; in time order). The images share the batch pixel budget; the video's frames share one size from
    /// <paramref name="videoSampling"/> (the model's settings when null). Each frame runs through the tower on its own.</summary>
    public MossVlGeneration Generate(string question, IReadOnlyList<ImageData> images, IReadOnlyList<TimedImage> video,
        VideoFrameSamplingOptions? videoSampling, int maxTokens, SamplingOptions sampling, Action<string>? onDelta, CancellationToken cancel)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (images.Count == 0 && video.Count == 0)
            throw new Core.Exceptions.HartsyInferenceException(
                "MOSS-VL-Realtime answers about images and video: attach at least one image or a video to the user message.");
        VisionPatchGrid[] stills = Model.Processor.PreprocessBatch(images.Select(i => new RawRgbFrame(i.Rgb, i.Width, i.Height)).ToArray());
        VisionPatchGrid[] frames = [];
        List<Tensor> merged = [];
        try
        {
            frames = Model.Processor.PreprocessVideo(video.Select(f => new RawRgbFrame(f.Image.Rgb, f.Image.Width, f.Image.Height)).ToArray(),
                videoSampling ?? Model.VideoSampling);
            List<MossVlImageFeatures> stillFeatures = [.. stills.Select(g => Encode(g, merged, cancel))];
            List<VisionFrameFeatures> frameFeatures = new(frames.Length);
            for (int i = 0; i < frames.Length; i++)
            {
                MossVlImageFeatures f = Encode(frames[i], merged, cancel);
                frameFeatures.Add(new VisionFrameFeatures(f.MergedTokens, f.MergedHeight, f.MergedWidth, video[i].TimestampSeconds));
            }
            MossVlGenerator generator = new(Model.Decoder, Model.Config, Model.Tokenizer, _backend);
            return frameFeatures.Count == 0
                ? generator.Generate(question, stillFeatures, maxTokens, sampling, onDelta, cancel)
                : generator.GenerateVideo(question, frameFeatures, maxTokens, sampling, onDelta, cancel, stillFeatures);
        }
        finally
        {
            foreach (Tensor t in merged) t.Dispose();
            foreach (VisionPatchGrid g in stills) g.Dispose();
            foreach (VisionPatchGrid g in frames) g.Dispose();
        }
    }

    /// <summary>Runs one preprocessed image or frame through the tower; the merged tokens are added to <paramref name="owned"/>.</summary>
    private MossVlImageFeatures Encode(VisionPatchGrid grid, List<Tensor> owned, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        Qwen3VlVisionEncoder.VisionOutput output = Tower.Encoder.Forward(_backend, grid.PixelValues, grid.GridT, grid.GridH, grid.GridW);
        // MOSS's unified merger yields no deepstack features; dispose any a different tower config would produce.
        foreach (Tensor d in output.DeepstackFeatures) d.Dispose();
        Tensor tokens = output.MergedTokens;
        owned.Add(tokens);
        return new MossVlImageFeatures(tokens, grid.MergedHeight, grid.MergedWidth);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Tower.Dispose();
        Model.Dispose();
    }
}
