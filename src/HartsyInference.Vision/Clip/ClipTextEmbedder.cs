using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Tokenizers;
using HartsyInference.Vision.Embeddings;

namespace HartsyInference.Vision.Clip;

/// <summary>Standalone CLIP text encoder. Tokenizes a string with the CLIP BPE, runs the causal transformer, extracts the hidden state at the EOS position, projects it through <c>text_projection</c>, and L2-normalizes the result. The output is a unit vector that can be cosine-compared with an <see cref="ImageEmbedding"/>.</summary>
/// <remarks>The wrapped <see cref="ClipTextEncoder"/> and <see cref="ClipTokenizer"/> are not owned by this class; <see cref="ClipModelLoader"/> manages their lifetimes.</remarks>
public sealed unsafe class ClipTextEmbedder
{
    private readonly ClipTextEncoder _encoder;
    private readonly ClipTokenizer _tokenizer;

    /// <summary>Text-tower configuration in use.</summary>
    public ClipTextEncoderConfig Config => _encoder.Config;

    /// <summary>Output embedding dimensionality (equal to <see cref="ClipTextEncoderConfig.ProjectionDim"/>).</summary>
    public int EmbeddingDim => _encoder.Config.ProjectionDim;

    /// <summary>Wraps a pre-loaded text encoder and tokenizer. Both must already be ready to use — weights loaded (including <c>text_projection</c>) and the BPE merges parsed.</summary>
    public ClipTextEmbedder(ClipTextEncoder encoder, ClipTokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (encoder.Config.ProjectionDim <= 0)
            throw new ArgumentException("Text encoder must have ProjectionDim > 0 for standalone CLIP scoring — a checkpoint without text_projection cannot produce a comparable embedding.", nameof(encoder));
        _encoder = encoder;
        _tokenizer = tokenizer;
    }

    /// <summary>Encodes a single string into an L2-normalized <see cref="TextEmbedding"/>.</summary>
    public TextEmbedding Encode(IBackend backend, string text)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(text);

        Tensor pooled = EncodeRaw(backend, text);
        ClipImageEncoder.L2NormalizeInPlace(pooled);
        return new TextEmbedding(pooled, text);
    }

    /// <summary>Encodes a batch of strings. Each text is tokenized independently, but the transformer runs on the full <c>[B, 77]</c> token tensor in one pass.</summary>
    public IReadOnlyList<TextEmbedding> EncodeBatch(IBackend backend, IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(texts);
        if (texts.Count == 0)
            return Array.Empty<TextEmbedding>();

        int batch = texts.Count;
        int[][] tokenIds = new int[batch][];
        int[] eosPositions = new int[batch];

        for (int b = 0; b < batch; b++)
        {
            tokenIds[b] = _tokenizer.Encode(texts[b]);
            int eos = ClipTokenizer.FindEosPosition(tokenIds[b]);
            if (eos < 0)
                throw new InvalidOperationException($"Tokenizer returned no EOS token for text index {b} — should be impossible since {nameof(ClipTokenizer.Encode)} always appends one.");
            eosPositions[b] = eos;
        }

        (Tensor penultimate, Tensor? pooled) = _encoder.EncodePenultimate(backend, tokenIds, eosPositions);
        penultimate.Dispose();
        if (pooled is null)
            throw new InvalidOperationException("Text encoder did not return a pooled output. Make sure ProjectionDim > 0 and text_projection.weight was loaded.");

        int dim = EmbeddingDim;
        TextEmbedding[] results = new TextEmbedding[batch];
        float* batchPtr = (float*)pooled.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            Tensor slice = new Tensor(new TensorShape(1, dim), DType.F32);
            float* slicePtr = (float*)slice.DataPointer;
            ReadOnlySpan<float> src = new ReadOnlySpan<float>(batchPtr + b * dim, dim);
            Span<float> dst = new Span<float>(slicePtr, dim);
            src.CopyTo(dst);
            ClipImageEncoder.L2NormalizeInPlace(slice);
            results[b] = new TextEmbedding(slice, texts[b]);
        }
        pooled.Dispose();
        return results;
    }

    /// <summary>Tokenizes + encodes a single string, returning the raw <c>[1, dim]</c> pooled tensor (un-normalized). Caller owns and must dispose the returned tensor.</summary>
    public Tensor EncodeRaw(IBackend backend, string text)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(text);

        int[] tokenIds = _tokenizer.Encode(text);
        int eos = ClipTokenizer.FindEosPosition(tokenIds);
        if (eos < 0)
            throw new InvalidOperationException("Tokenizer returned no EOS token — should be impossible since ClipTokenizer.Encode always appends one.");

        int[][] batchIds = [tokenIds];
        int[] eosBatch = [eos];

        (Tensor penultimate, Tensor? pooled) = _encoder.EncodePenultimate(backend, batchIds, eosBatch);
        penultimate.Dispose();
        if (pooled is null)
            throw new InvalidOperationException("Text encoder did not return a pooled output. Make sure ProjectionDim > 0 and text_projection.weight was loaded.");
        return pooled;
    }
}
