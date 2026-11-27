using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>The Hy4 Flash identity hyper-connection (transformers <c>HYV4FlashHyperConnection</c> and <c>HYV4FlashHyperHead</c>): from the <c>hc</c>-copy residual stream a
/// pre gate collapses the copies for the sublayer and a post gate writes its output back on top of the unchanged stream. There is no Sinkhorn step and no comb matrix.</summary>
/// <remarks><c>mixes = (fn @ flat) * rsqrt(mean(flat^2) + normEps)</c>; <c>pre = sigmoid(mixes[:hc] * scale[0] + base[:hc]) + hcEps</c>; <c>post = magnitude * sigmoid(mixes[hc:] * scale[1] + base[hc:]) + hcEps</c>;
/// collapse <c>sum_i pre_i x_i</c>; expand <c>out_i = post_i * sub + x_i</c>. The head reads the same way with a single gate set and no post. Collapse and expand run the shared
/// <see cref="IBackend.HcPreMix"/> and <see cref="IBackend.HcPostMix"/> ops (an identity comb is the exact expand).</remarks>
public sealed class Hy4FlashHyperConnection
{
    private readonly IBackend _backend;
    private readonly int _hc;
    private readonly int _hidden;
    private readonly float _hcEps;
    private readonly float _magnitude;
    private readonly float _normEps;
    private readonly Tensor _fn;
    private readonly float[] _base;
    private readonly float[] _scale;
    private readonly bool _head;

    /// <param name="fn"><c>[2 * hc, hc * hidden]</c> for a sublayer, <c>[hc, hc * hidden]</c> for the head; F32.</param>
    /// <param name="bias"><c>base</c>: <c>[2 * hc]</c> or <c>[hc]</c>.</param>
    /// <param name="scale"><c>scale</c>: <c>[2]</c> for a sublayer (pre, post), <c>[1]</c> for the head.</param>
    /// <param name="head">Whether this is the final read-out.</param>
    public Hy4FlashHyperConnection(IBackend backend, int hc, int hidden, float hcEps, float magnitude, float normEps, Tensor fn, float[] bias, float[] scale, bool head)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(fn);
        ArgumentNullException.ThrowIfNull(bias);
        ArgumentNullException.ThrowIfNull(scale);
        int mixes = head ? hc : 2 * hc;
        if (fn.Shape.Rank != 2 || fn.Shape[0] != mixes || fn.Shape[1] != (long)hc * hidden) throw new ArgumentException($"fn must be [{mixes}, {hc * hidden}]; got {fn.Shape}.", nameof(fn));
        if (bias.Length != mixes || scale.Length != (head ? 1 : 2)) throw new ArgumentException("base and scale do not match the connection kind.");
        _backend = backend;
        _hc = hc;
        _hidden = hidden;
        _hcEps = hcEps;
        _magnitude = magnitude;
        _normEps = normEps;
        _fn = fn;
        _base = bias;
        _scale = scale;
        _head = head;
    }

    /// <summary>The pre and post gates of every token's stream (<c>[tokens, hc, hidden]</c>); <paramref name="post"/> is empty for the head.</summary>
    public void Gates(ReadOnlySpan<float> stream, int tokens, out float[] pre, out float[] post)
    {
        int flat = _hc * _hidden, mixes = _head ? _hc : 2 * _hc;
        if (stream.Length != (long)tokens * flat) throw new ArgumentException("stream must hold tokens x hc x hidden values.", nameof(stream));
        float[] projected = HostOps.Project(_backend, stream, tokens, flat, _fn, mixes);
        pre = new float[tokens * _hc];
        post = _head ? [] : new float[tokens * _hc];
        for (int t = 0; t < tokens; t++)
        {
            double sum = 0;
            for (int i = 0; i < flat; i++) sum += (double)stream[t * flat + i] * stream[t * flat + i];
            float rms = 1f / MathF.Sqrt((float)(sum / flat) + _normEps);
            for (int i = 0; i < _hc; i++)
                pre[t * _hc + i] = Sigmoid(projected[t * mixes + i] * rms * _scale[0] + _base[i]) + _hcEps;
            if (_head) continue;
            for (int i = 0; i < _hc; i++)
                post[t * _hc + i] = _magnitude * Sigmoid(projected[t * mixes + _hc + i] * rms * _scale[1] + _base[_hc + i]) + _hcEps;
        }
    }

    /// <summary>Collapses the copies: <c>y[t] = sum_i pre[t,i] * x[t,i]</c>, <c>[tokens, hidden]</c>.</summary>
    public float[] Collapse(ReadOnlySpan<float> stream, ReadOnlySpan<float> pre, int tokens)
    {
        using Tensor xT = HostMath.Tensor(stream, tokens, _hc, _hidden);
        using Tensor preT = HostMath.Tensor(pre, tokens, _hc);
        using Tensor yT = new(new TensorShape(tokens, _hidden), DType.F32);
        _backend.HcPreMix(yT, xT, preT);
        return yT.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>Writes <c>post[t,i] * sub[t] + stream[t,i]</c> over <paramref name="stream"/>.</summary>
    public void Expand(ReadOnlySpan<float> sub, Span<float> stream, ReadOnlySpan<float> post, int tokens)
    {
        float[] identity = new float[tokens * _hc * _hc];
        for (int t = 0; t < tokens; t++)
            for (int i = 0; i < _hc; i++) identity[(t * _hc + i) * _hc + i] = 1f;
        using Tensor subT = HostMath.Tensor(sub, tokens, _hidden);
        using Tensor residualT = HostMath.Tensor(stream, tokens, _hc, _hidden);
        using Tensor postT = HostMath.Tensor(post, tokens, _hc);
        using Tensor combT = HostMath.Tensor(identity, tokens, _hc, _hc);
        using Tensor outT = new(new TensorShape(tokens, _hc, _hidden), DType.F32);
        _backend.HcPostMix(outT, subT, residualT, postT, combT);
        outT.AsReadOnlySpan<float>().CopyTo(stream);
    }

    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));
}
