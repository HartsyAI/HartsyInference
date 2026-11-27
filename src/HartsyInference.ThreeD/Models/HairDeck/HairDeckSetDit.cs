using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>The HairDeck set DiT (<c>hairdeck/models/setdit.py</c>) for one sample. Every block self-attends over the card
/// tokens without a mask or positions, cross-attends to the condition, and runs a tanh-GELU MLP, each behind a 9-way adaLN
/// modulation. The conditioning is the timestep embedding MLP plus the pooled condition MLP.</summary>
public sealed unsafe class HairDeckSetDit
{
    private const float NormEps = 1e-6f;

    private readonly HairDeckSetDitCheckpointConverter.Weights _w;
    private readonly float _attentionScale;

    public HairDeckSetDit(HairDeckSetDitCheckpointConverter.Weights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _w = weights;
        _attentionScale = 1f / MathF.Sqrt(weights.Dim / weights.Heads);
    }

    /// <summary>Model width.</summary>
    public int Dim => _w.Dim;

    /// <summary>Width of one token (the card token or chain token).</summary>
    public int TokenDim => _w.TokenDim;

    /// <summary>Rows of the learned null condition (the context length the model was trained with).</summary>
    public int ContextTokens => _w.ContextTokens;

    /// <summary>Velocity of the tokens <paramref name="x"/> <c>[N, TokenDim]</c> at time <paramref name="t"/> given the encoded
    /// condition <paramref name="ctx"/> <c>[M, Dim]</c>; <paramref name="uncond"/> replaces it by the null condition.</summary>
    public Tensor Forward(IBackend backend, Tensor x, float t, Tensor ctx, bool uncond)
    {
        ValidateTokens(x);
        Tensor condition = uncond ? _w.NullContext : ctx;
        if (condition.Shape[1] != _w.Dim)
            throw new ArgumentException($"HairDeck context width {condition.Shape[1]} does not match the model width {_w.Dim}.", nameof(ctx));

        Tensor c = Conditioning(backend, t, condition);
        Tensor h = HairDeckNumerics.Project(backend, x, _w.Input);
        for (int i = 0; i < _w.Blocks.Count; i++)
        {
            Tensor next = Block(backend, _w.Blocks[i], h, c, condition);
            h.Dispose();
            h = next;
        }
        Tensor velocity = Head(backend, h, c);
        h.Dispose();
        c.Dispose();
        return velocity;
    }

    /// <summary>The conditioning <c>t_mlp(timestep(t)) + pool_mlp(mean(ctx))</c> as <c>[1, Dim]</c>; the caller disposes it.</summary>
    public Tensor Conditioning(IBackend backend, float t, Tensor ctx)
    {
        using Tensor embedding = HairDeckNumerics.Timestep(t);
        using Tensor timeIn = HairDeckNumerics.Project(backend, embedding, _w.TimeMlpIn);
        using Tensor timeAct = HairDeckNumerics.Silu(backend, timeIn);
        Tensor time = HairDeckNumerics.Project(backend, timeAct, _w.TimeMlpOut);

        using Tensor pooled = HairDeckNumerics.MeanRows(ctx);
        using Tensor poolIn = HairDeckNumerics.Project(backend, pooled, _w.PoolMlpIn);
        using Tensor poolAct = HairDeckNumerics.Silu(backend, poolIn);
        using Tensor poolOut = HairDeckNumerics.Project(backend, poolAct, _w.PoolMlpOut);

        Tensor c = new(time.Shape, DType.F32);
        backend.Add(c, time, poolOut);
        time.Dispose();
        return c;
    }

    /// <summary>The token embedding <c>inp(x)</c> as <c>[N, Dim]</c>; the caller disposes it.</summary>
    public Tensor Embed(IBackend backend, Tensor x)
    {
        ValidateTokens(x);
        return HairDeckNumerics.Project(backend, x, _w.Input);
    }

    /// <summary>Runs every block on <paramref name="h"/> <c>[N, Dim]</c>; the input is left to the caller, the result is new.</summary>
    public Tensor RunBlocks(IBackend backend, Tensor h, Tensor c, Tensor ctx)
    {
        Tensor current = HairDeckNumerics.Clone(h);
        for (int i = 0; i < _w.Blocks.Count; i++)
        {
            Tensor next = Block(backend, _w.Blocks[i], current, c, ctx);
            current.Dispose();
            current = next;
        }
        return current;
    }

    /// <summary>The output head: <c>out(LayerNorm(h) · (1 + scale) + shift)</c> with the shift and scale of <c>f_ada(silu(c))</c>.</summary>
    public Tensor Head(IBackend backend, Tensor h, Tensor c)
    {
        using Tensor act = HairDeckNumerics.Silu(backend, c);
        using Tensor modulation = HairDeckNumerics.Project(backend, act, _w.FinalAda);
        Tensor[] chunks = HairDeckNumerics.Chunks(modulation, 2, _w.Dim);
        using Tensor shift = chunks[0];
        using Tensor scale = chunks[1];
        using Tensor normed = HairDeckNumerics.NormModulate(backend, h, scale, shift, NormEps);
        return HairDeckNumerics.Project(backend, normed, _w.Output);
    }

    private Tensor Block(IBackend backend, HairDeckSetDitCheckpointConverter.Block b, Tensor h, Tensor c, Tensor ctx)
    {
        using Tensor act = HairDeckNumerics.Silu(backend, c);
        using Tensor modulation = HairDeckNumerics.Project(backend, act, b.Ada);
        Tensor[] m = HairDeckNumerics.Chunks(modulation, 9, _w.Dim);
        try
        {
            Tensor selfIn = HairDeckNumerics.NormModulate(backend, h, m[1], m[0], NormEps);
            Tensor selfOut = Attention(backend, selfIn, selfIn, b.SelfAttention);
            selfIn.Dispose();
            Tensor afterSelf = HairDeckNumerics.GatedAdd(h, selfOut, m[2]);
            selfOut.Dispose();

            Tensor crossIn = HairDeckNumerics.NormModulate(backend, afterSelf, m[4], m[3], NormEps);
            Tensor crossOut = Attention(backend, crossIn, ctx, b.CrossAttention);
            crossIn.Dispose();
            Tensor afterCross = HairDeckNumerics.GatedAdd(afterSelf, crossOut, m[5]);
            afterSelf.Dispose();
            crossOut.Dispose();

            Tensor mlpIn = HairDeckNumerics.NormModulate(backend, afterCross, m[7], m[6], NormEps);
            Tensor mlpHidden = HairDeckNumerics.Project(backend, mlpIn, b.MlpIn);
            mlpIn.Dispose();
            Tensor mlpAct = HairDeckNumerics.GeluTanh(backend, mlpHidden);
            mlpHidden.Dispose();
            Tensor mlpOut = HairDeckNumerics.Project(backend, mlpAct, b.MlpOut);
            mlpAct.Dispose();
            Tensor result = HairDeckNumerics.GatedAdd(afterCross, mlpOut, m[8]);
            afterCross.Dispose();
            mlpOut.Dispose();
            return result;
        }
        finally
        {
            foreach (Tensor part in m) part.Dispose();
        }
    }

    private Tensor Attention(IBackend backend, Tensor query, Tensor keyValueInput, HairDeckSetDitCheckpointConverter.Attention a)
    {
        int heads = _w.Heads, headDim = _w.Dim / heads;
        long n = query.Shape[0], m = keyValueInput.Shape[0];
        using Tensor q = HairDeckNumerics.Project(backend, query, a.Query);
        using Tensor kv = HairDeckNumerics.Project(backend, keyValueInput, a.KeyValue);
        using Tensor qh = HairDeckNumerics.Create(1, heads, n, headDim);
        using Tensor kh = HairDeckNumerics.Create(1, heads, m, headDim);
        using Tensor vh = HairDeckNumerics.Create(1, heads, m, headDim);
        HairDeckNumerics.SplitHeads(q, 0, qh, heads, headDim);
        HairDeckNumerics.SplitHeads(kv, 0, kh, heads, headDim);
        HairDeckNumerics.SplitHeads(kv, _w.Dim, vh, heads, headDim);
        using Tensor qn = HairDeckNumerics.Create(1, heads, n, headDim);
        using Tensor kn = HairDeckNumerics.Create(1, heads, m, headDim);
        backend.RmsNorm(qn, qh, a.QueryNorm, NormEps);
        backend.RmsNorm(kn, kh, a.KeyNorm, NormEps);
        using Tensor oh = HairDeckNumerics.Create(1, heads, n, headDim);
        backend.ScaledDotProductAttention(oh, qn, kn, vh, null, _attentionScale);
        using Tensor merged = HairDeckNumerics.Create(n, _w.Dim);
        HairDeckNumerics.MergeHeads(oh, merged, heads, headDim);
        return HairDeckNumerics.Project(backend, merged, a.Output);
    }

    private void ValidateTokens(Tensor x)
    {
        if (x.Shape.Rank != 2 || x.Shape[1] != _w.TokenDim)
            throw new ArgumentException($"HairDeck tokens must be [N, {_w.TokenDim}], got {x.Shape}.", nameof(x));
        if (x.DType != DType.F32)
            throw new ArgumentException($"HairDeck tokens must be F32, got {x.DType}.", nameof(x));
    }
}
