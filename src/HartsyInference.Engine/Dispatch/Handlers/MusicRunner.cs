using HartsyInference.Audio.Pipelines;
using HartsyInference.Engine;
using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Dispatch.Handlers;

/// <summary>A loaded MusicGen stack: the T5 prompt encoder + tokenizer feeding the decoder+EnCodec pipeline, plus the
/// loaders whose mmap-backed weights must outlive them.</summary>
public sealed class MusicRunner : IModalityRunner
{
    private readonly IReadOnlyList<IDisposable> _owned;

    /// <summary>Creates a runner over the assembled MusicGen components.</summary>
    public MusicRunner(string modelId, T5TextEncoder textEncoder, T5Tokenizer tokenizer, MusicGenPipeline pipeline,
        int sampleRate, IBackend backend, IReadOnlyList<IDisposable> owned)
    {
        ModelId = modelId;
        TextEncoder = textEncoder;
        Tokenizer = tokenizer;
        Pipeline = pipeline;
        SampleRate = sampleRate;
        Backend = backend;
        _owned = owned;
    }

    /// <inheritdoc/>
    public Modality Modality => Modality.Music;

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>The T5 prompt encoder.</summary>
    public T5TextEncoder TextEncoder { get; }

    /// <summary>The T5 tokenizer.</summary>
    public T5Tokenizer Tokenizer { get; }

    /// <summary>The MusicGen decoder + codec pipeline.</summary>
    public MusicGenPipeline Pipeline { get; }

    /// <summary>Output sample rate in Hz.</summary>
    public int SampleRate { get; }

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
