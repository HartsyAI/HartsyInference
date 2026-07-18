using HartsyInference.Engine;
using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Dispatch.Handlers;

/// <summary>A loaded LTX-Video stack: the T5-XXL encoder + tokenizer feeding the video pipeline, plus the loaders that
/// own the mmap-backed weights.</summary>
public sealed class VideoRunner : IModalityRunner
{
    private readonly IReadOnlyList<IDisposable> _owned;

    /// <summary>Creates a runner over the assembled LTX-Video components.</summary>
    public VideoRunner(string modelId, LtxVideoPipeline pipeline, T5TextEncoder textEncoder, T5Tokenizer tokenizer,
        LtxVideoConfig config, IBackend backend, IReadOnlyList<IDisposable> owned)
    {
        ModelId = modelId;
        Pipeline = pipeline;
        TextEncoder = textEncoder;
        Tokenizer = tokenizer;
        Config = config;
        Backend = backend;
        _owned = owned;
    }

    /// <inheritdoc/>
    public Modality Modality => Modality.Video;

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>The LTX-Video pipeline.</summary>
    public LtxVideoPipeline Pipeline { get; }

    /// <summary>The T5-XXL prompt encoder.</summary>
    public T5TextEncoder TextEncoder { get; }

    /// <summary>The T5 tokenizer.</summary>
    public T5Tokenizer Tokenizer { get; }

    /// <summary>The pipeline config (VAE compression constraints).</summary>
    public LtxVideoConfig Config { get; }

    /// <summary>The borrowed compute backend.</summary>
    public IBackend Backend { get; }

    /// <inheritdoc/>
    public void Dispose()
    {
        Pipeline.Dispose();
        foreach (IDisposable d in _owned)
            d.Dispose();
    }
}
