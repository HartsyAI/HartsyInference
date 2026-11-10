using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The SANA-WM text path over a loaded Gemma-2 encoder: <see cref="SanaWmTextInputs"/> builds the CHI-prefixed
/// condition and the plain negative. Borrows the encoder and tokenizer; their owner releases them.</summary>
public sealed class SanaWmGemmaTextEncoder : ISanaWmTextEncoder
{
    private readonly IBackend _backend;
    private readonly LlamaStyleEncoder _encoder;
    private readonly GemmaTokenizer _tokenizer;

    /// <summary>Wraps a loaded encoder and its tokenizer.</summary>
    public SanaWmGemmaTextEncoder(IBackend backend, LlamaStyleEncoder encoder, GemmaTokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(tokenizer);
        _backend = backend;
        _encoder = encoder;
        _tokenizer = tokenizer;
    }

    /// <inheritdoc/>
    public SanaWmTextConditioning Encode(string prompt, string negativePrompt)
    {
        (Tensor condition, Tensor conditionMask) = SanaWmTextInputs.EncodeCondition(_backend, _encoder, _tokenizer, prompt);
        try
        {
            (Tensor negative, Tensor negativeMask) = SanaWmTextInputs.EncodeNegative(_backend, _encoder, _tokenizer, negativePrompt);
            return new SanaWmTextConditioning(condition, conditionMask, negative, negativeMask);
        }
        catch
        {
            condition.Dispose();
            conditionMask.Dispose();
            throw;
        }
    }
}
