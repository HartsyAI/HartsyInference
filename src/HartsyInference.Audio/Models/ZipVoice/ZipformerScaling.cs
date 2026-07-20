using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>Zipformer's custom low-level primitives (<c>zipvoice/models/modules/scaling.py</c>), inference-time
/// (eval-mode) forward only — all training-only regularizers (ActivationBalancer/Balancer, Whiten, dropout,
/// LimitParamValue) are exact no-ops in eval mode and are simply omitted. <c>ScaledLinear</c>/<c>ScaledConv1d</c>
/// are plain <c>nn.Linear</c>/<c>nn.Conv1d</c> at inference — the "scale" only affected random init before
/// training and is already baked into the checkpoint's saved weights, so ordinary <see cref="IBackend.Linear"/>
/// / <see cref="IBackend.Conv1d"/> calls are used throughout this port with no extra runtime scaling.</summary>
// TODO(gpu-residency): SwooshL/R (this file) and BiasNorm.Forward (below) remain host `float*` loops — SwooshL/R
// need a Softplus/Exp/Log primitive and BiasNorm needs a per-row sum-of-squares reduction, neither of which is
// exposed on IBackend (searched: no Log/Exp/Softplus elementwise op, no generic per-row-scalar-broadcast op).
// Adding either would need a new PTX kernel (the established pattern elsewhere, e.g. Mish/Prelu/RepeatKvHeads),
// but this sandbox has no nvcc/CUDA compiler available (only a pip ptxas without the nvcc frontend) — see the
// ZipVoice perf-pass entry in docs/Checklists/PHASE_5_AUDIO.md. Everything else in the Zipformer encoder layer
// (residual adds, broadcast adds, bypass highway, GLU gate, attention value-mixing, NonlinAttention, the
// attention content-score term) was converted to GPU-resident IBackend calls; these two remain the last
// unavoidable host-touch points per layer without a new kernel.
internal static unsafe class ZipformerScaling
{
    /// <summary>SwooshL(x) = softplus(x - 4) - 0.08x - 0.035. Used by <c>FeedforwardModule.out_proj</c>.</summary>
    public static float SwooshL(float x) => Softplus(x - 4.0f) - 0.08f * x - 0.035f;

    /// <summary>SwooshR(x) = softplus(x - 1) - 0.08x - 0.313261687. Used by the top-level/per-stage time-embed
    /// MLPs and <c>ConvolutionModule.out_proj</c>.</summary>
    public static float SwooshR(float x) => Softplus(x - 1.0f) - 0.08f * x - 0.313261687f;

    /// <summary>Numerically-stable softplus: max(x,0) + log1p(exp(-|x|)).</summary>
    private static float Softplus(float x) => MathF.Max(x, 0f) + MathF.Log(1f + MathF.Exp(-MathF.Abs(x)));

    public static void SwooshLInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++) p[i] = SwooshL(p[i]);
    }

    public static void SwooshRInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++) p[i] = SwooshR(p[i]);
    }
}

/// <summary>BiasNorm (<c>zipvoice/models/modules/scaling.py:BiasNorm</c>): the normalization used throughout
/// Zipformer instead of LayerNorm/RMSNorm. Computes a per-position scale from the mean-of-squares of
/// <c>(x - bias)</c> (bias participates ONLY in that statistic, never added to the output), then multiplies
/// <c>x</c> (the raw, un-shifted input) by that scale times <c>exp(log_scale)</c>. No additive output bias.</summary>
internal sealed unsafe class ZipformerBiasNorm
{
    private readonly int _dim;
    private Tensor? _bias;
    private float _scale;

    public ZipformerBiasNorm(int dim) => _dim = dim;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _bias = WhisperOps.EnsureF32(w[$"{prefix}.bias"]);
        float logScale = *(float*)WhisperOps.EnsureF32(w[$"{prefix}.log_scale"]).DataPointer;
        _scale = MathF.Exp(logScale);
    }

    /// <summary>In/out shape <c>[T, dim]</c> (batch=1 squeezed out by callers).</summary>
    public Tensor Forward(Tensor x, int t)
    {
        Tensor output = new(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)output.DataPointer;
        float* bp = (float*)_bias!.DataPointer;
        for (int i = 0; i < t; i++)
        {
            long off = (long)i * _dim;
            double sumSq = 0;
            for (int d = 0; d < _dim; d++)
            {
                float diff = xp[off + d] - bp[d];
                sumSq += (double)diff * diff;
            }
            float invStd = (float)(1.0 / Math.Sqrt(sumSq / _dim));
            float scale = invStd * _scale;
            for (int d = 0; d < _dim; d++) op[off + d] = xp[off + d] * scale;
        }
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_bias is not null) yield return _bias;
    }
}
