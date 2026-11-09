using HartsyInference.Core.Backends;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.Models.MossVl;
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
        Action<string>? onDelta, CancellationToken cancel)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (images.Count == 0)
            throw new Core.Exceptions.HartsyInferenceException("MOSS-VL-Realtime answers about images: attach at least one image to the user message.");
        VisionPatchGrid[] grids = Model.Processor.PreprocessBatch(images.Select(i => new RawRgbFrame(i.Rgb, i.Width, i.Height)).ToArray());
        List<Tensor> merged = [];
        try
        {
            List<MossVlImageFeatures> features = new(grids.Length);
            foreach (VisionPatchGrid grid in grids)
            {
                cancel.ThrowIfCancellationRequested();
                Qwen3VlVisionEncoder.VisionOutput output = Tower.Encoder.Forward(_backend, grid.PixelValues, grid.GridT, grid.GridH, grid.GridW);
                // MOSS's unified merger yields no deepstack features; dispose any a different tower config would produce.
                foreach (Tensor d in output.DeepstackFeatures) d.Dispose();
                Tensor tokens = output.MergedTokens;
                merged.Add(tokens);
                features.Add(new MossVlImageFeatures(tokens, grid.MergedHeight, grid.MergedWidth));
            }
            MossVlGenerator generator = new(Model.Decoder, Model.Config, Model.Tokenizer, _backend);
            return generator.Generate(question, features, maxTokens, sampling, onDelta, cancel);
        }
        finally
        {
            foreach (Tensor t in merged) t.Dispose();
            foreach (VisionPatchGrid g in grids) g.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Tower.Dispose();
        Model.Dispose();
    }
}
