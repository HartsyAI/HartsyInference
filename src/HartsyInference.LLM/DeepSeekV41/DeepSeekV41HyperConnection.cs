using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The hyper-connection (mHC) wrapper around one sublayer of a V4.1 block: derive mixing coefficients from the residual stream, collapse it for the sublayer, expand the result back.</summary>
/// <remarks>Matches upstream <c>Block.hc_mixes</c>, <c>hc_pre</c> and <c>hc_post</c>. The coefficients a sublayer derives are used by the <i>next</i> sublayer's
/// collapse, so a caller keeps <see cref="Mixes"/> results across calls; the block composes that. Only the projection and its flattened RMS statistic live here,
/// the split, collapse and expand go through the shared <see cref="IBackend"/> <c>Hc*</c> ops.</remarks>
public sealed class DeepSeekV41HyperConnection
{
    private readonly IBackend _backend;
    private readonly float[] _fn;
    private readonly float[] _scale;
    private readonly float[] _bias;
    private readonly float _normEps;
    private readonly float _hcEps;
    private readonly int _iters;

    /// <summary>Residual copies.</summary>
    public int Hc { get; }

    /// <summary>Residual width.</summary>
    public int Dim { get; }

    /// <param name="backend">Provides the <c>Hc*</c> primitives.</param>
    /// <param name="hc">Residual copies.</param>
    /// <param name="dim">Residual width.</param>
    /// <param name="iters">Sinkhorn iterations (config <c>hc_sinkhorn_iters</c>).</param>
    /// <param name="hcEps">Config <c>hc_eps</c>.</param>
    /// <param name="normEps">Config <c>rms_norm_eps</c>, used for the stream statistic.</param>
    /// <param name="fn">Projection of the flattened stream, <c>[(2 + hc) * hc, hc * dim]</c>.</param>
    /// <param name="scale">Pre, post and comb scales, <c>[3]</c>.</param>
    /// <param name="bias">Per-mix bias, <c>[(2 + hc) * hc]</c>.</param>
    public DeepSeekV41HyperConnection(IBackend backend, int hc, int dim, int iters, float hcEps, float normEps, float[] fn, float[] scale, float[] bias)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(fn);
        ArgumentNullException.ThrowIfNull(scale);
        ArgumentNullException.ThrowIfNull(bias);
        ArgumentOutOfRangeException.ThrowIfLessThan(hc, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(dim, 1);
        int mix = (2 + hc) * hc;
        if (fn.Length != (long)mix * hc * dim) throw new ArgumentException("fn must be [(2 + hc) * hc, hc * dim].", nameof(fn));
        if (scale.Length != 3) throw new ArgumentException("scale must hold 3 values.", nameof(scale));
        if (bias.Length != mix) throw new ArgumentException("bias must hold (2 + hc) * hc values.", nameof(bias));
        _backend = backend;
        Hc = hc;
        Dim = dim;
        _iters = iters;
        _hcEps = hcEps;
        _normEps = normEps;
        _fn = fn;
        _scale = scale;
        _bias = bias;
    }

    /// <summary>Derives the pre, post and comb coefficients from the residual stream.</summary>
    /// <param name="x">Stream, <c>[tokens, hc, dim]</c>.</param>
    /// <param name="tokens">Row count.</param>
    /// <param name="pre">Receives <c>[tokens, hc]</c>.</param>
    /// <param name="post">Receives <c>[tokens, hc]</c>.</param>
    /// <param name="comb">Receives <c>[tokens, hc, hc]</c>, doubly stochastic.</param>
    public void Mixes(ReadOnlySpan<float> x, int tokens, Span<float> pre, Span<float> post, Span<float> comb)
    {
        int flat = Hc * Dim, mix = (2 + Hc) * Hc;
        CheckSize(x.Length, tokens, flat, nameof(x));
        CheckSize(pre.Length, tokens, Hc, nameof(pre));
        CheckSize(post.Length, tokens, Hc, nameof(post));
        CheckSize(comb.Length, tokens, Hc * Hc, nameof(comb));

        using Tensor mixes = new(new TensorShape(tokens, mix), DType.F32);
        Span<float> m = mixes.AsSpan<float>();
        for (int t = 0; t < tokens; t++)
        {
            ReadOnlySpan<float> row = x.Slice(t * flat, flat);
            float sq = 0f;
            for (int i = 0; i < flat; i++) sq += row[i] * row[i];
            float rsqrt = 1f / MathF.Sqrt(sq / flat + _normEps);
            for (int o = 0; o < mix; o++)
            {
                ReadOnlySpan<float> w = _fn.AsSpan(o * flat, flat);
                float sum = 0f;
                for (int i = 0; i < flat; i++) sum += row[i] * w[i];
                m[t * mix + o] = sum * rsqrt;
            }
        }

        using Tensor preT = new(new TensorShape(tokens, Hc), DType.F32);
        using Tensor postT = new(new TensorShape(tokens, Hc), DType.F32);
        using Tensor combT = new(new TensorShape(tokens, Hc, Hc), DType.F32);
        using Tensor scaleT = Vector(_scale);
        using Tensor biasT = Vector(_bias);
        _backend.HcSplitSinkhorn(preT, postT, combT, mixes, scaleT, biasT, Hc, _iters, _hcEps);
        preT.AsReadOnlySpan<float>().CopyTo(pre);
        postT.AsReadOnlySpan<float>().CopyTo(post);
        combT.AsReadOnlySpan<float>().CopyTo(comb);
    }

    /// <summary>Collapses the copies into one sublayer input: <c>y[t] = sum_i pre[t,i] * x[t,i]</c>.</summary>
    /// <param name="x">Stream, <c>[tokens, hc, dim]</c>.</param>
    /// <param name="pre">Coefficients, <c>[tokens, hc]</c>.</param>
    /// <param name="y">Receives <c>[tokens, dim]</c>.</param>
    public void Collapse(ReadOnlySpan<float> x, ReadOnlySpan<float> pre, int tokens, Span<float> y)
    {
        CheckSize(x.Length, tokens, Hc * Dim, nameof(x));
        CheckSize(pre.Length, tokens, Hc, nameof(pre));
        CheckSize(y.Length, tokens, Dim, nameof(y));
        using Tensor xT = Wrap(x, tokens, Hc, Dim);
        using Tensor preT = Wrap(pre, tokens, Hc);
        using Tensor yT = new(new TensorShape(tokens, Dim), DType.F32);
        _backend.HcPreMix(yT, xT, preT);
        yT.AsReadOnlySpan<float>().CopyTo(y);
    }

    /// <summary>Expands a sublayer output back to the copies and mixes the old stream in: <c>out[t,i] = post[t,i] * sub[t] + sum_j comb[t,j,i] * residual[t,j]</c>.</summary>
    /// <param name="sub">Sublayer output, <c>[tokens, dim]</c>.</param>
    /// <param name="residual">The stream before the sublayer, <c>[tokens, hc, dim]</c>.</param>
    /// <param name="post">Coefficients, <c>[tokens, hc]</c>.</param>
    /// <param name="comb">Coefficients, <c>[tokens, hc, hc]</c>.</param>
    /// <param name="output">Receives <c>[tokens, hc, dim]</c>; must not alias <paramref name="residual"/>.</param>
    public void Expand(ReadOnlySpan<float> sub, ReadOnlySpan<float> residual, ReadOnlySpan<float> post, ReadOnlySpan<float> comb, int tokens,
        Span<float> output)
    {
        CheckSize(sub.Length, tokens, Dim, nameof(sub));
        CheckSize(residual.Length, tokens, Hc * Dim, nameof(residual));
        CheckSize(post.Length, tokens, Hc, nameof(post));
        CheckSize(comb.Length, tokens, Hc * Hc, nameof(comb));
        CheckSize(output.Length, tokens, Hc * Dim, nameof(output));
        using Tensor subT = Wrap(sub, tokens, Dim);
        using Tensor resT = Wrap(residual, tokens, Hc, Dim);
        using Tensor postT = Wrap(post, tokens, Hc);
        using Tensor combT = Wrap(comb, tokens, Hc, Hc);
        using Tensor outT = new(new TensorShape(tokens, Hc, Dim), DType.F32);
        _backend.HcPostMix(outT, subT, resT, postT, combT);
        outT.AsReadOnlySpan<float>().CopyTo(output);
    }

    private static void CheckSize(int length, int tokens, int perToken, string name)
    {
        if (length != (long)tokens * perToken) throw new ArgumentException($"{name} must hold tokens x {perToken} values.", name);
    }

    private static Tensor Vector(float[] values)
    {
        Tensor t = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor Wrap(ReadOnlySpan<float> values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
