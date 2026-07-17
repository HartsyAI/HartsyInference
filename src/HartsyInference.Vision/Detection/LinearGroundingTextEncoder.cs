using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Minimal stand-in for the BERT text tower: a learned embedding table (<c>vocab × embedDim</c>) followed
/// by an optional linear projection to <see cref="TextDim"/>. It produces shape-correct, deterministic token
/// embeddings so the Grounding DINO forward pass is runnable and testable without a real BERT.
///
/// <para>This is NOT a language model — there is no attention, so the embeddings are context-free. It exists only
/// to unblock structural bring-up; swap it for a BERT-backed <see cref="IGroundingTextEncoder"/> (see that
/// interface's remarks on hoisting <c>BertModel</c> into a shared package) for real open-vocabulary quality.</para></summary>
public sealed class LinearGroundingTextEncoder : IGroundingTextEncoder
{
    private readonly int _vocabSize;
    private readonly int _embedDim;
    private readonly int _textDim;
    private readonly bool _project;

    private Tensor? _embedding;   // [vocab, embedDim]
    private Tensor? _projWeight;  // [textDim, embedDim] (only when embedDim != textDim)
    private Tensor? _projBias;    // [textDim]

    public int TextDim => _textDim;

    /// <summary>Creates the stub. When <paramref name="embedDim"/> equals <paramref name="textDim"/> the projection
    /// is skipped and the embedding rows are returned directly.</summary>
    public LinearGroundingTextEncoder(int vocabSize, int embedDim, int textDim)
    {
        if (vocabSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabSize), vocabSize, "Vocab size must be positive.");
        if (embedDim <= 0 || textDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(embedDim), "Dims must be positive.");
        _vocabSize = vocabSize;
        _embedDim = embedDim;
        _textDim = textDim;
        _project = embedDim != textDim;
    }

    /// <summary>Resolves the embedding table (and projection, if any) through the shared weight bag.</summary>
    public void Build(IGroundingWeightBag bag, string prefix)
    {
        ArgumentNullException.ThrowIfNull(bag);
        _embedding = bag.Get($"{prefix}.embedding.weight", _vocabSize, _embedDim);
        if (_project)
        {
            _projWeight = bag.Get($"{prefix}.projection.weight", _textDim, _embedDim);
            _projBias = bag.Get($"{prefix}.projection.bias", _textDim);
        }
    }

    /// <summary>Convenience loader for a real safetensors dict.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "text_encoder")
        => Build(new DictionaryWeightBag(weights), prefix);

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedding is not null) yield return _embedding;
        if (_projWeight is not null) yield return _projWeight;
        if (_projBias is not null) yield return _projBias;
    }

    public unsafe Tensor Encode(IBackend backend, ReadOnlySpan<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (_embedding is null)
            throw new InvalidOperationException("LinearGroundingTextEncoder weights not loaded.");
        int t = tokenIds.Length;
        if (t == 0)
            throw new ArgumentException("Token id list must be non-empty.", nameof(tokenIds));

        // Gather the embedding rows for the requested ids. This is pure table assembly (a host copy of constant
        // rows), not model math, so a direct pointer gather is appropriate here.
        Tensor gathered = new Tensor(new TensorShape(1, t, _embedDim), DType.F32);
        float* dst = (float*)gathered.DataPointer;
        float* table = (float*)_embedding.DataPointer;
        for (int i = 0; i < t; i++)
        {
            int id = tokenIds[i];
            if (id < 0 || id >= _vocabSize)
                id = 0;
            Buffer.MemoryCopy(table + (long)id * _embedDim, dst + (long)i * _embedDim, (long)_embedDim * 4, (long)_embedDim * 4);
        }

        if (!_project)
            return gathered;

        Tensor projected = new Tensor(new TensorShape(1, t, _textDim), DType.F32);
        backend.Linear(projected, gathered, _projWeight!, _projBias);
        gathered.Dispose();
        return projected;
    }
}
