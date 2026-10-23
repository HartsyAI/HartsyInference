using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>What Fibo 2.0 editing needs beyond text-to-image: the Qwen3-VL vision tower (read from the same text-encoder file the
/// recipe already loads) and the FLUX.2 VAE encoder. Built on the first edit, so text-to-image never pays for the tower. Owns the
/// tower and its file mapping; the language tower, VAE and path are borrowed.</summary>
internal sealed class Fibo2EditSupport(LlamaStyleEncoder language, string encoderPath, Flux2VaeComponents vae) : IDisposable
{
    /// <summary>Qwen3-VL-4B's deepstack taps (ComfyUI <c>QWEN3VL_VISION["qwen3vl_4b"]</c>); the weights carry no config to read them from.</summary>
    private static readonly int[] DeepstackTaps = [5, 11, 17];

    /// <summary>ComfyUI's <c>to_qwen_grid</c> pixel bounds, and the processor's 0.5 mean and std.</summary>
    private const int MinPixels = 3136, MaxPixels = 1605632;

    private readonly object _gate = new();
    private Qwen3VlVisionEncoder? _vision;
    private CheckpointSource? _source;
    private Loaded? _loaded;
    private int _disposed;

    /// <summary>The tower's smart-resize and patch layout.</summary>
    public Qwen3VlImageProcessor Processor => Load().Processor;

    /// <summary>The language tower with the vision tokens spliced in.</summary>
    public Qwen3VlMultimodalEncoder Encoder => Load().Encoder;

    /// <summary>Pixels per side one vision token covers: patch size times the spatial merge.</summary>
    public int VisionFactor => Load().Factor;

    /// <summary>The FLUX.2 VAE encoder.</summary>
    /// <exception cref="InvalidOperationException">The VAE file ships no encoder.</exception>
    public VaeEncoder VaeEncoder => vae.Encoder
        ?? throw new InvalidOperationException("Fibo 2.0 editing needs the FLUX.2 VAE encoder, but the VAE file holds only a decoder.");

    private Loaded Load()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        lock (_gate)
        {
            if (_loaded is not null)
                return _loaded;
            (Dictionary<string, Tensor> bare, CheckpointSource source) = ComponentLoader.Load(encoderPath, "Fibo2Recipe", BareVisionKey,
                applyFp8Dequant: true);
            Qwen3VlVisionEncoder? vision = null;
            try
            {
                if (bare.Count == 0)
                    throw new InvalidDataException($"Fibo 2.0 editing needs the Qwen3-VL vision tower, but '{Path.GetFileName(encoderPath)}' holds no visual.* tensors.");
                Qwen3VlVisionConfig config = Qwen3VlVisionConfig.FromWeights(bare, numHeads: 16) with
                {
                    DeepstackVisualIndexes = DeepstackTaps,
                    MergerExactGelu = true,
                };
                vision = new Qwen3VlVisionEncoder(config);
                vision.LoadWeights(bare);
                Qwen3VlImageProcessor processor = new(config, MinPixels, MaxPixels, imageMean: [0.5f, 0.5f, 0.5f], imageStd: [0.5f, 0.5f, 0.5f]);
                Qwen3VlMultimodalEncoder encoder = new(language, vision, processor, config);
                Logs.Info($"[Fibo2Recipe] Qwen3-VL vision tower loaded for editing: depth {config.Depth}, hidden {config.HiddenSize}.");
                // Kept only once everything loaded, so a failed first edit leaves nothing behind for a retry to overwrite.
                _source = source;
                _vision = vision;
                return _loaded = new Loaded(processor, encoder, config.PatchSize * config.SpatialMergeSize);
            }
            catch
            {
                vision?.Dispose();
                source.Dispose();
                throw;
            }
        }
    }

    /// <summary>The tower's tensors under bare names (<c>blocks.0...</c>), from <c>model.visual.</c> or <c>visual.</c>; null drops the rest.</summary>
    private static string? BareVisionKey(string key)
    {
        const string marker = "visual.";
        int at = key.IndexOf(marker, StringComparison.Ordinal);
        bool rooted = at == 0 || (at > 0 && key[..at] == "model.");
        return rooted ? key[(at + marker.Length)..] : null;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _vision?.Dispose();
        _source?.Dispose();
    }

    private sealed record Loaded(Qwen3VlImageProcessor Processor, Qwen3VlMultimodalEncoder Encoder, int Factor);
}
