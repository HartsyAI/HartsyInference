using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ernie5;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Glm5Next;

/// <summary>The glm5_next text decoder (GLM-5.3-Flash, Sage S2.1 Flash) on <see cref="IBackend"/> ops with host F32 activations: embedding broadcast to
/// <c>hc_mult</c> streams, the decoder layers, an unweighted mean over the streams, <c>model.norm</c> and the LM head. The MTP layer is not run.</summary>
public sealed class Glm5NextModel : IDisposable
{
    private readonly IBackend _backend;
    private readonly Glm5NextConfig _c;
    private readonly Glm5NextLayer[] _layers;
    private readonly Tensor _embedding, _finalNorm, _lmHead;

    /// <summary>Loads every decoder layer, the embedding, the final norm and the LM head from <paramref name="source"/>. The model keeps the source's
    /// tensors; the source must outlive it.</summary>
    public Glm5NextModel(IBackend backend, Glm5NextConfig config, IErnie5WeightSource source)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(source);
        _backend = backend;
        _c = config;
        _layers = new Glm5NextLayer[config.Layers];
        Tensor? embedding = null, finalNorm = null, lmHead = null;
        try
        {
            embedding = source.Read(Glm5NextWeightNames.Embedding);
            RequireShape(embedding, config.Vocab, config.Hidden, "embed_tokens");
            // The final norm is read as F32 whatever the checkpoint stores: the norm kernel reads its weight as F32.
            finalNorm = Glm5NextLayer.F32(source, Glm5NextWeightNames.FinalNorm, config.Hidden);
            lmHead = source.Read(Glm5NextWeightNames.LmHead);
            RequireShape(lmHead, config.Vocab, config.Hidden, "lm_head");
            for (int i = 0; i < config.Layers; i++) _layers[i] = new Glm5NextLayer(backend, config, i, source);
        }
        catch
        {
            // Everything this constructor took ownership of is released here; the fields are assigned only once the load has succeeded.
            embedding?.Dispose();
            finalNorm?.Dispose();
            lmHead?.Dispose();
            foreach (Glm5NextLayer? l in _layers) l?.Dispose();
            throw;
        }
        _embedding = embedding;
        _finalNorm = finalNorm;
        _lmHead = lmHead;
    }

    /// <summary>The decoder's shape.</summary>
    public Glm5NextConfig Config => _c;

    /// <summary>Called with each layer's residual stream <c>[tokens, hc, hidden]</c> after it ran; for parity checks.</summary>
    public Action<int, float[]>? LayerObserver { get; set; }

    /// <summary>Token embeddings of <paramref name="ids"/> as <c>[tokens, hidden]</c>.</summary>
    public float[] Embed(ReadOnlySpan<int> ids)
    {
        int d = _c.Hidden;
        float[] rows = new float[ids.Length * d];
        for (int t = 0; t < ids.Length; t++)
        {
            if ((uint)ids[t] >= (uint)_c.Vocab) throw new ArgumentOutOfRangeException(nameof(ids), $"Token id {ids[t]} is outside the {_c.Vocab}-entry vocabulary.");
            using Tensor row = _embedding.SliceRows(ids[t], 1);
            Tensor f32 = TensorCasts.EnsureF32(row);
            try
            {
                f32.AsReadOnlySpan<float>().CopyTo(rows.AsSpan(t * d, d));
            }
            finally
            {
                if (!ReferenceEquals(f32, row)) f32.Dispose();
            }
        }
        return rows;
    }

    /// <summary>Runs <paramref name="embeddings"/> <c>[tokens, hidden]</c> through the decoder at positions <c>state.Length ..</c>, advancing the state, and
    /// returns the final normed hidden rows <c>[tokens, hidden]</c>.</summary>
    public float[] ForwardHidden(ReadOnlySpan<float> embeddings, Glm5NextState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        int d = _c.Hidden, hc = _c.HcMult;
        if (embeddings.Length == 0 || embeddings.Length % d != 0) throw new ArgumentException("embeddings must hold whole [hidden] rows.", nameof(embeddings));
        int tokens = embeddings.Length / d, start = state.Length;
        if (start + tokens > state.Capacity) throw new InvalidOperationException("The state's cache is full; allocate one for a longer sequence.");

        float[] stream = new float[tokens * hc * d];
        for (int t = 0; t < tokens; t++)
            for (int s = 0; s < hc; s++) embeddings.Slice(t * d, d).CopyTo(stream.AsSpan((t * hc + s) * d, d));
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].Forward(stream, tokens, start, state);
            LayerObserver?.Invoke(i, stream);
        }
        state.Length = start + tokens;

        // The stream read-out is the plain mean of the copies (no head projection), then the final RMS norm.
        float[] hidden = new float[tokens * d];
        for (int t = 0; t < tokens; t++)
            for (int s = 0; s < hc; s++)
                for (int j = 0; j < d; j++) hidden[t * d + j] += stream[(t * hc + s) * d + j] / hc;
        HostOps.RmsNorm(_backend, hidden, d, _finalNorm, _c.RmsNormEps);
        return hidden;
    }

    /// <summary>Logits <c>[tokens, vocab]</c> for the rows <paramref name="hidden"/> <c>[tokens, hidden]</c>.</summary>
    public float[] Project(ReadOnlySpan<float> hidden)
    {
        int rows = hidden.Length / _c.Hidden;
        return HostOps.Project(_backend, hidden, rows, _c.Hidden, _lmHead, _c.Vocab);
    }

    /// <summary>Embeds <paramref name="ids"/>, runs them and returns every position's logits <c>[tokens, vocab]</c>.</summary>
    public float[] Forward(ReadOnlySpan<int> ids, Glm5NextState state) => Project(ForwardHidden(Embed(ids), state));

    private static void RequireShape(Tensor t, int rows, int cols, string name)
    {
        if (t.Shape.Rank != 2 || t.Shape[0] != rows || t.Shape[1] != cols) throw new InvalidDataException($"{name} must be [{rows}, {cols}]; got {t.Shape}.");
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (Glm5NextLayer? l in _layers) l?.Dispose();
        _embedding?.Dispose();
        _finalNorm?.Dispose();
        _lmHead?.Dispose();
    }
}
