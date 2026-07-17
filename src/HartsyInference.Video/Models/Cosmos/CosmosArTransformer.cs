using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Video.Models.Cosmos;

/// <summary>The Cosmos-Predict1 Video2World autoregressive backbone: a Llama3-shape discrete-token decoder with 3D
/// RoPE over the <c>(T,H,W)</c> video latent, GQA + QK-norm self-attention, per-layer non-causal T5
/// cross-attention, an additive 3D absolute-position input signal, SwiGLU FFN, and RMSNorm — driven by a fixed
/// KV cache for prefill + per-token decode. A distinct, reusable type (the Phase-10 action-conditioned world
/// models instantiate the same class); it is NOT an LLM <c>GenericTransformer</c> config.
///
/// <para>Scope: batch = 1, F32 activations, eager. Every matmul/attention op routes through <see cref="IBackend"/>.
/// Weights are the raw loaded tensors (stable references) for <see cref="IBackend.PreloadWeights"/>.</para></summary>
public sealed unsafe class CosmosArTransformer : IDisposable
{
    private readonly CosmosArConfig _cfg;
    private readonly CosmosArBlock[] _blocks;
    private readonly CosmosAbsPos3D _absPos;
    private Tensor? _embed;        // [vocab, dim] F32 (host gather)
    private Tensor? _finalNorm;    // [dim]
    private Tensor? _output;       // [vocab, dim] (untied head)
    private Tensor? _ropeCos, _ropeSin;   // [1, N, head_dim] for the active grid
    private int _gridN;
    private int _disposed;

    /// <summary>The architecture this backbone was built for.</summary>
    public CosmosArConfig Config => _cfg;

    /// <summary>Creates the backbone for the given config (weights loaded separately).</summary>
    public CosmosArTransformer(CosmosArConfig cfg)
    {
        _cfg = cfg;
        _blocks = new CosmosArBlock[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) _blocks[i] = new CosmosArBlock(cfg, i);
        _absPos = new CosmosAbsPos3D(cfg.Dim);
    }

    /// <summary>Loads weights from a Cosmos-named state dict (produced by <c>CosmosArCheckpointConverter</c>):
    /// <c>tok_embeddings.weight</c>, <c>layers.{i}.*</c>, <c>norm.weight</c>, <c>output.weight</c>, and (V2W)
    /// <c>pos_emb.weight</c> when the learned abs-pos table ships.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        ThrowIfDisposed();
        _embed = EnsureF32(w["tok_embeddings.weight"]);
        _finalNorm = EnsureF32(w["norm.weight"]);
        _output = w["output.weight"];
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"layers.{i}");
        if (_cfg.ApplyAbsPosEmb && w.TryGetValue("pos_emb.weight", out Tensor? posTable)
            && posTable.Shape.Rank == 2 && (int)posTable.Shape[1] == _cfg.Dim)
            _absPos.LoadLearnedTable(EnsureF32(posTable));
    }

    /// <summary>All weight tensors (stable references) for <see cref="IBackend.PreloadWeights"/> /
    /// <see cref="IBackend.FreeWeights"/>. The embed table is host-only (gather) and skipped.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_finalNorm is not null) yield return _finalNorm;
        if (_output is not null) yield return _output;
        foreach (CosmosArBlock b in _blocks)
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Sets the video latent grid <c>(T,H,W)</c>: builds the 3D-RoPE cos/sin tables and the abs-pos table
    /// (learned if present, else sin-cos). Call once before prefill; <c>N = T·H·W</c> is the AR sequence length.</summary>
    public void SetGrid(int latentT, int latentH, int latentW)
    {
        ThrowIfDisposed();
        _ropeCos?.Dispose(); _ropeSin?.Dispose();
        Cosmos3DRoPE rope = new(_cfg.HeadDim, _cfg.RopeTheta, _cfg.RopeSection);
        (_ropeCos, _ropeSin) = rope.BuildTable(latentT, latentH, latentW);
        _gridN = latentT * latentH * latentW;
        if (_cfg.ApplyAbsPosEmb && _absPos.Length == 0) _absPos.BuildSincos(latentT, latentH, latentW);
    }

    /// <summary>Total AR sequence length for the active grid (<c>T·H·W</c>).</summary>
    public int SequenceLength => _gridN;

    /// <summary>Runs the backbone over <paramref name="tokenIds"/> at absolute positions
    /// <c>[posStart, posStart+t)</c>, appending to <paramref name="cache"/>, cross-attending
    /// <paramref name="context"/> <c>[1, ctxLen, context_dim]</c> (null for the base 4B path). Returns the final
    /// post-norm hidden <c>[1, t, dim]</c> (feed to <see cref="ProjectLogits"/>).</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int posStart, IKvCache cache, Tensor? context, int ctxLen)
    {
        ThrowIfDisposed();
        if (_ropeCos is null) throw new InvalidOperationException("Call SetGrid before Forward.");
        int t = tokenIds.Length;

        Tensor embeds = new(new TensorShape(1, t, _cfg.Dim), DType.F32);
        EmbedLookup(embeds, tokenIds);
        if (_cfg.ApplyAbsPosEmb) _absPos.AddSlice(embeds, posStart, t);

        (Tensor cos, Tensor sin) = SliceRope(posStart, t);
        Tensor hidden = embeds;
        try
        {
            for (int i = 0; i < _blocks.Length; i++)
            {
                Tensor next = _blocks[i].Forward(backend, hidden, t, posStart, cache, i, cos, sin, context, ctxLen);
                hidden.Dispose();
                hidden = next;
            }
            cache.AdvanceLength(t);
            Tensor normed = new(new TensorShape(1, t, _cfg.Dim), DType.F32);
            backend.RmsNorm(normed, hidden, _finalNorm!, _cfg.NormEps);
            hidden.Dispose();
            return normed;
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    /// <summary>Projects hidden <c>[1, t, dim]</c> → logits <c>[1, t, vocab]</c> via the untied output head.</summary>
    public Tensor ProjectLogits(IBackend backend, Tensor hidden, int t)
    {
        ThrowIfDisposed();
        Tensor logits = new(new TensorShape(1, t, _cfg.VocabSize), DType.F32);
        backend.Linear(logits, hidden, _output!, null);
        return logits;
    }

    private (Tensor Cos, Tensor Sin) SliceRope(int posStart, int t)
    {
        int d = _cfg.HeadDim;
        if (posStart + t > _gridN)
            throw new ArgumentOutOfRangeException(nameof(posStart), $"[{posStart},{posStart + t}) exceeds grid N={_gridN}.");
        Tensor cos = new(new TensorShape(1, t, d), DType.F32);
        Tensor sin = new(new TensorShape(1, t, d), DType.F32);
        long bytes = (long)t * d * sizeof(float);
        Buffer.MemoryCopy((float*)_ropeCos!.DataPointer + (long)posStart * d, (float*)cos.DataPointer, bytes, bytes);
        Buffer.MemoryCopy((float*)_ropeSin!.DataPointer + (long)posStart * d, (float*)sin.DataPointer, bytes, bytes);
        return (cos, sin);
    }

    private void EmbedLookup(Tensor output, ReadOnlySpan<int> tokenIds)
    {
        int dim = _cfg.Dim;
        float* op = (float*)output.DataPointer;
        float* ep = (float*)_embed!.DataPointer;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int id = tokenIds[s];
            if ((uint)id >= (uint)_cfg.VocabSize)
                throw new ArgumentException($"token id {id} out of range [0, {_cfg.VocabSize}).");
            Buffer.MemoryCopy(ep + (long)id * dim, op + (long)s * dim, (long)dim * 4, (long)dim * 4);
        }
    }

    private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(CosmosArTransformer));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _ropeCos?.Dispose(); _ropeSin?.Dispose();
        _ropeCos = null; _ropeSin = null;
        _embed = null; _finalNorm = null; _output = null;
    }
}
