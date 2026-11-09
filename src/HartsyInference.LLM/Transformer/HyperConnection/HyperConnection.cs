using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Manifold-constrained hyper-connection (mHC) around one sublayer: derives mixing coefficients from a residual stream of
/// <c>hc</c> copies, collapses the stream for the sublayer, and expands the sublayer output back.</summary>
/// <remarks>The coefficients a sublayer derives are used by the next sublayer's collapse, so a caller keeps <see cref="Mixes"/> results across calls.
/// The projection and its RMS statistic run on the host; the split, collapse and expand go through the shared <see cref="IBackend"/> <c>Hc*</c> ops.
/// Model differences live in <see cref="HyperConnectionConfig"/>.</remarks>
public sealed class HyperConnection
{
    private readonly IBackend _backend;
    private readonly HyperConnectionConfig _config;
    private readonly float[] _fn;
    private readonly float[] _scale;
    private readonly float[] _bias;

    /// <summary>Residual copies.</summary>
    public int Hc => _config.Hc;

    /// <summary>Residual width.</summary>
    public int Dim => _config.Dim;

    /// <param name="backend">Provides the <c>Hc*</c> primitives.</param>
    /// <param name="config">Formula parameters.</param>
    /// <param name="fn">Projection of the flattened stream, <c>[(2 + hc) * hc, hc * dim]</c>.</param>
    /// <param name="scale">Pre, post and comb scales, <c>[3]</c>.</param>
    /// <param name="bias">Per-mix bias, <c>[(2 + hc) * hc]</c>.</param>
    public HyperConnection(IBackend backend, HyperConnectionConfig config, float[] fn, float[] scale, float[] bias)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(fn);
        ArgumentNullException.ThrowIfNull(scale);
        ArgumentNullException.ThrowIfNull(bias);
        ArgumentOutOfRangeException.ThrowIfLessThan(config.Hc, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(config.Dim, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(config.SinkhornIters, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(config.SinkhornEps);
        ArgumentOutOfRangeException.ThrowIfNegative(config.PreEps);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(config.NormEps);
        if (!float.IsFinite(config.PostMultiplier)) throw new ArgumentOutOfRangeException(nameof(config), "PostMultiplier must be finite.");
        int hc = config.Hc, mix = (2 + hc) * hc;
        if (fn.Length != (long)mix * hc * config.Dim) throw new ArgumentException("fn must be [(2 + hc) * hc, hc * dim].", nameof(fn));
        if (scale.Length != 3) throw new ArgumentException("scale must hold 3 values.", nameof(scale));
        if (bias.Length != mix) throw new ArgumentException("bias must hold (2 + hc) * hc values.", nameof(bias));
        _backend = backend;
        _config = config;
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
        int hc = Hc, flat = hc * Dim, mix = (2 + hc) * hc;
        CheckSize(x.Length, tokens, flat, nameof(x));
        CheckSize(pre.Length, tokens, hc, nameof(pre));
        CheckSize(post.Length, tokens, hc, nameof(post));
        CheckSize(comb.Length, tokens, hc * hc, nameof(comb));

        float[] projected = HostMath.Linear(x, _fn, tokens, flat, mix);
        for (int t = 0; t < tokens; t++)
        {
            ReadOnlySpan<float> row = x.Slice(t * flat, flat);
            float sq = 0f;
            for (int i = 0; i < flat; i++) sq += row[i] * row[i];
            float rms = RmsScale(sq / flat, _config.NormEps, _config.MixRms);
            for (int o = 0; o < mix; o++) projected[t * mix + o] *= rms;
        }
        using Tensor mixes = HostMath.Tensor(projected, tokens, mix);

        using Tensor preT = new(new TensorShape(tokens, hc), DType.F32);
        using Tensor postT = new(new TensorShape(tokens, hc), DType.F32);
        using Tensor combT = new(new TensorShape(tokens, hc, hc), DType.F32);
        using Tensor scaleT = HostMath.Tensor(_scale, 3);
        using Tensor biasT = HostMath.Tensor(_bias, _bias.Length);
        _backend.HcSplitSinkhorn(preT, postT, combT, mixes, scaleT, biasT, hc, _config.SinkhornIters, _config.SinkhornEps,
            _config.PreEps, _config.PostMultiplier);
        preT.AsReadOnlySpan<float>().CopyTo(pre);
        postT.AsReadOnlySpan<float>().CopyTo(post);
        combT.AsReadOnlySpan<float>().CopyTo(comb);
    }

    /// <summary>Collapses the copies into one sublayer input: <c>y[t] = sum_i pre[t,i] * x[t,i]</c>.</summary>
    /// <param name="x">Stream, <c>[tokens, hc, dim]</c>.</param>
    /// <param name="pre">Coefficients, <c>[tokens, hc]</c>.</param>
    /// <param name="tokens">Row count.</param>
    /// <param name="y">Receives <c>[tokens, dim]</c>.</param>
    public void Collapse(ReadOnlySpan<float> x, ReadOnlySpan<float> pre, int tokens, Span<float> y)
    {
        CheckSize(x.Length, tokens, Hc * Dim, nameof(x));
        CheckSize(pre.Length, tokens, Hc, nameof(pre));
        CheckSize(y.Length, tokens, Dim, nameof(y));
        using Tensor xT = HostMath.Tensor(x, tokens, Hc, Dim);
        using Tensor preT = HostMath.Tensor(pre, tokens, Hc);
        using Tensor yT = new(new TensorShape(tokens, Dim), DType.F32);
        _backend.HcPreMix(yT, xT, preT);
        yT.AsReadOnlySpan<float>().CopyTo(y);
    }

    /// <summary>Expands a sublayer output back to the copies and mixes the old stream in: <c>out[t,i] = post[t,i] * sub[t] + sum_j comb[t,j,i] * residual[t,j]</c>.</summary>
    /// <param name="sub">Sublayer output, <c>[tokens, dim]</c>.</param>
    /// <param name="residual">The stream before the sublayer, <c>[tokens, hc, dim]</c>.</param>
    /// <param name="post">Coefficients, <c>[tokens, hc]</c>.</param>
    /// <param name="comb">Coefficients, <c>[tokens, hc, hc]</c>.</param>
    /// <param name="tokens">Row count.</param>
    /// <param name="output">Receives <c>[tokens, hc, dim]</c>; the inputs are copied first, so it may alias <paramref name="residual"/>.</param>
    public void Expand(ReadOnlySpan<float> sub, ReadOnlySpan<float> residual, ReadOnlySpan<float> post, ReadOnlySpan<float> comb, int tokens,
        Span<float> output)
    {
        int hc = Hc;
        CheckSize(sub.Length, tokens, Dim, nameof(sub));
        CheckSize(residual.Length, tokens, hc * Dim, nameof(residual));
        CheckSize(post.Length, tokens, hc, nameof(post));
        CheckSize(comb.Length, tokens, hc * hc, nameof(comb));
        CheckSize(output.Length, tokens, hc * Dim, nameof(output));
        using Tensor subT = HostMath.Tensor(sub, tokens, Dim);
        using Tensor resT = HostMath.Tensor(residual, tokens, hc, Dim);
        using Tensor postT = HostMath.Tensor(post, tokens, hc);
        using Tensor combT = HostMath.Tensor(comb, tokens, hc, hc);
        using Tensor outT = new(new TensorShape(tokens, hc, Dim), DType.F32);
        _backend.HcPostMix(outT, subT, resT, postT, combT);
        outT.AsReadOnlySpan<float>().CopyTo(output);
    }

    /// <summary>Final read-out of the stream: a pre-mix from this head's own projection, then the final RMS norm.</summary>
    /// <remarks>Used by models whose head has its own gates (ERNIE-5.5). The head has no Sinkhorn step.</remarks>
    /// <param name="x">Stream, <c>[tokens, hc, dim]</c>.</param>
    /// <param name="tokens">Row count.</param>
    /// <param name="headFn">Head projection, <c>[hc, hc * dim]</c>.</param>
    /// <param name="headScale">Gate scale, <c>[1]</c>.</param>
    /// <param name="headBase">Per-copy gate bias, <c>[hc]</c>.</param>
    /// <param name="normWeight">Final RMS norm weight, <c>[dim]</c>.</param>
    /// <param name="output">Receives <c>[tokens, dim]</c>.</param>
    public void Head(ReadOnlySpan<float> x, int tokens, ReadOnlySpan<float> headFn, ReadOnlySpan<float> headScale, ReadOnlySpan<float> headBase,
        ReadOnlySpan<float> normWeight, Span<float> output)
    {
        int hc = Hc, flat = hc * Dim;
        CheckSize(x.Length, tokens, flat, nameof(x));
        CheckSize(output.Length, tokens, Dim, nameof(output));
        if (headFn.Length != (long)hc * flat) throw new ArgumentException("headFn must be [hc, hc * dim].", nameof(headFn));
        if (headScale.Length != 1) throw new ArgumentException("headScale must hold 1 value.", nameof(headScale));
        if (headBase.Length != hc) throw new ArgumentException("headBase must hold hc values.", nameof(headBase));
        if (normWeight.Length != Dim) throw new ArgumentException("normWeight must hold dim values.", nameof(normWeight));

        float[] projected = HostMath.Linear(x, headFn, tokens, flat, hc);
        float[] pre = new float[tokens * hc];
        for (int t = 0; t < tokens; t++)
        {
            ReadOnlySpan<float> row = x.Slice(t * flat, flat);
            float sq = 0f;
            for (int i = 0; i < flat; i++) sq += row[i] * row[i];
            float rms = RmsScale(sq / flat, _config.NormEps, _config.HeadRms);
            for (int i = 0; i < hc; i++)
                pre[t * hc + i] = Sigmoid(projected[t * hc + i] * rms * headScale[0] + headBase[i]) + _config.PreEps;
        }
        using Tensor preT = HostMath.Tensor(pre, tokens, hc);
        using Tensor xT = HostMath.Tensor(x, tokens, hc, Dim);
        using Tensor yT = new(new TensorShape(tokens, Dim), DType.F32);
        _backend.HcPreMix(yT, xT, preT);
        yT.AsReadOnlySpan<float>().CopyTo(output);
        HostMath.RmsNormRows(output, normWeight, Dim, _config.NormEps);
    }

    /// <summary>The factor that normalizes a row whose mean square is <paramref name="meanSquare"/>.</summary>
    private static float RmsScale(float meanSquare, float eps, RmsEpsPlacement placement) => placement switch
    {
        RmsEpsPlacement.InsideSqrt => 1f / MathF.Sqrt(meanSquare + eps),
        RmsEpsPlacement.OutsideSqrt => 1f / (MathF.Sqrt(meanSquare) + eps),
        _ => throw new ArgumentOutOfRangeException(nameof(placement), placement, "Unknown RMS epsilon placement."),
    };

    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

    private static void CheckSize(int length, int tokens, int perToken, string name)
    {
        if (length != (long)tokens * perToken) throw new ArgumentException($"{name} must hold tokens x {perToken} values.", name);
    }
}
