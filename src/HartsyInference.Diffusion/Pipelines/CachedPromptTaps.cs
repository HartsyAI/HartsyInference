using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>A multi-layer text encoder's prompt features behind an <see cref="EncodedPromptCache"/>, for the FLUX.2-style pipelines that encode one prompt per slot.</summary>
internal sealed unsafe class CachedPromptTaps : IDisposable
{
    private readonly IBackend _backend;
    private readonly LlamaStyleEncoder _encoder;
    private readonly int[] _layers;
    private readonly int _contextDim;
    private readonly string _family;
    private readonly EncodedPromptCache _cache;

    /// <summary>Encodes through <paramref name="layers"/> of <paramref name="encoder"/> on <paramref name="backend"/>; features must be <paramref name="contextDim"/> wide.</summary>
    public CachedPromptTaps(IBackend backend, LlamaStyleEncoder encoder, int[] layers, int contextDim, string family, int slots = 2)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentNullException.ThrowIfNull(family);
        _backend = backend;
        _encoder = encoder;
        _layers = layers;
        _contextDim = contextDim;
        _family = family;
        _cache = new EncodedPromptCache(slots);
    }

    /// <summary>The <c>[1, L, context]</c> features of <paramref name="tokens"/> in <paramref name="slot"/>, owned by this cache and reused while the slot's ids and valid length repeat.
    /// A miss frees <paramref name="denoiser"/> first, since the encoder and a resident denoiser need not fit together; <paramref name="padding"/> masks a right-padded prompt as transformers does.</summary>
    public Tensor Encode(int slot, int[] tokens, TextEncoderPadding? padding, ResidentWeights denoiser)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(denoiser);
        int validLength = padding?.ValidLength ?? 0;
        if (_cache.TryGet(slot, tokens, validLength, out Tensor? cached))
            return cached!;
        denoiser.Release(force: true);
        Tensor hidden;
        try
        {
            hidden = _encoder.EncodeMultiLayer(_backend, [tokens], _layers, interleavedLayout: false, padding);
        }
        finally
        {
            _backend.FreeWeights(_encoder.EnumerateWeights());
        }
        // Host-materialize before the reclaim, so the cached features survive every later activation sweep.
        _ = hidden.DataPointer;
        _backend.FreeActivations();
        if (hidden.Shape[2] != _contextDim)
        {
            long got = hidden.Shape[2];
            hidden.Dispose();
            throw new HartsyInferenceException($"{_family} text features are {got} wide; the transformer expects {_contextDim}.");
        }
        _cache.Store(slot, tokens, validLength, hidden);
        return hidden;
    }

    /// <inheritdoc/>
    public void Dispose() => _cache.Dispose();
}
