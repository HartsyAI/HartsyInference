using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Embeddings;

/// <summary>A decoder-based text-embedding model (Qwen3-Embedding, gte-Qwen2, e5-mistral, LLM2Vec, …): runs a
/// causal LLM decoder over the input and pools the final-layer hidden states (last-token by default, or mean) into
/// an L2-normalized sentence embedding. Reuses the verified <see cref="GenericTransformer"/> decoder + its
/// tokenizer — no new architecture, just a pooling head.</summary>
public sealed unsafe class DecoderEmbeddingModel : IDisposable
{
    private readonly GgufLanguageModel _lm;
    private int _disposed;

    /// <summary>0 = none, 1 = mean, 3 = last-token (llama.cpp pooling type). Decoder embedders default to last-token.</summary>
    public int PoolingType { get; }
    public int Hidden => _lm.Config.HiddenSize;
    public ILlmTokenizer Tokenizer => _lm.Tokenizer;

    private DecoderEmbeddingModel(GgufLanguageModel lm, int pooling) { _lm = lm; PoolingType = pooling; }

    public static DecoderEmbeddingModel Load(string path, int poolingType = 3)
        => new(GgufLanguageModel.Load(path), poolingType);

    /// <summary>Encodes token ids into a pooled, L2-normalized embedding of length <see cref="Hidden"/>.</summary>
    public float[] Encode(IBackend backend, IReadOnlyList<int> ids)
    {
        GenericTransformer model = _lm.Transformer;
        int seq = ids.Count, hidden = Hidden;

        using Tensor embeds = new(new TensorShape(1, seq, hidden), DType.F32);
        model.EmbedLookup(embeds, ids is int[] a ? a : ids.ToArray());

        using FixedKvCache cache = new(model.Config.NumLayers, 1, model.Config.NumKvHeads, model.Config.HeadDim, seq);
        using Tensor h = model.ForwardEmbeds(backend, embeds, seq, 0, cache, applyFinalNorm: true);
        backend.Sync();
        float* hp = (float*)h.DataPointer;   // [1, seq, hidden]

        float[] outv = new float[hidden];
        if (PoolingType == 1)   // mean over tokens
        {
            for (int s = 0; s < seq; s++) for (int c = 0; c < hidden; c++) outv[c] += hp[(long)s * hidden + c];
            for (int c = 0; c < hidden; c++) outv[c] /= seq;
        }
        else                     // last-token (causal models summarize the sequence at the final position)
        {
            float* lastTok = hp + (long)(seq - 1) * hidden;
            for (int c = 0; c < hidden; c++) outv[c] = lastTok[c];
        }
        double norm = 0; for (int c = 0; c < hidden; c++) norm += (double)outv[c] * outv[c];
        float inv = (float)(1.0 / Math.Sqrt(norm + 1e-12));
        for (int c = 0; c < hidden; c++) outv[c] *= inv;
        return outv;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
    }
}
