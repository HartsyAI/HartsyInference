using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the gated-linear-unit contract (<c>docs/BACKEND_OPS.md</c>): <c>out[r, i] = act(g) · v</c> over
/// the gate and value halves of each input row, with the sign-aware sigmoid so no tail overflows. The <see cref="IBackend"/>
/// default and the oracle every kernel is checked against.</summary>
public static unsafe class GluReference
{
    /// <summary>The gate factor of <paramref name="gate"/> under <paramref name="activation"/>.</summary>
    public static float Gate(float gate, in GluSpec spec)
    {
        switch (spec.Activation)
        {
            case GluActivation.Silu:
                return gate * Sigmoid(gate);
            case GluActivation.GeluTanh:
                return 0.5f * gate * (1f + MathF.Tanh(0.7978845608028654f * (gate + 0.044715f * gate * gate * gate)));
            case GluActivation.GeluErf:
                return 0.5f * gate * (1f + Erf(gate * 0.70710678118654752440f));
            case GluActivation.Situ:
                return spec.SituBeta * MathF.Tanh(gate / spec.SituBeta) * Sigmoid(gate);
            default:
                throw new ArgumentOutOfRangeException(nameof(spec), spec, "Unknown GLU activation.");
        }
    }

    /// <summary>The value factor under <paramref name="spec"/>: the value itself, soft-capped to <c>β<sub>lin</sub> · tanh(u / β<sub>lin</sub>)</c> for SiTU.</summary>
    public static float Value(float value, in GluSpec spec)
        => spec.Activation == GluActivation.Situ ? spec.SituLinearBeta * MathF.Tanh(value / spec.SituLinearBeta) : value;

    public static void Apply(Tensor output, Tensor input, in GluSpec spec)
    {
        (int inner, long rows) = GluContract.Validate(output, input, spec);
        Func<long, float> load = Loader(input);
        Action<long, float> store = Storer(output);
        long gateAt = spec.Gate == GluGateHalf.First ? 0 : inner, valueAt = spec.Gate == GluGateHalf.First ? inner : 0;
        for (long r = 0; r < rows; r++)
        {
            long at = r * 2 * inner;
            for (int i = 0; i < inner; i++)
                store(r * inner + i, Gate(load(at + gateAt + i), spec) * Value(load(at + valueAt + i), spec));
        }
    }

    /// <summary>The sign-aware logistic: neither tail evaluates <c>exp</c> of a large positive argument.</summary>
    public static float Sigmoid(float x) => x >= 0f ? 1f / (1f + MathF.Exp(-x)) : MathF.Exp(x) / (1f + MathF.Exp(x));

    /// <summary>Abramowitz-Stegun 7.1.26 rational approximation of the Gauss error function (max error about 1.5e-7), the form every backend's exact GELU uses.</summary>
    public static float Erf(float x)
    {
        int sign = x < 0 ? -1 : 1;
        x = MathF.Abs(x);
        float t = 1f / (1f + 0.3275911f * x);
        float y = 1f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * MathF.Exp(-x * x);
        return sign * y;
    }

    internal static Func<long, float> Loader(Tensor t)
    {
        void* p = t.DataPointer;
        if (t.DType == DType.F32) return i => ((float*)p)[i];
        if (t.DType == DType.F16) return i => (float)((Half*)p)[i];
        return i => BFloat16Bits.ToFloat(((ushort*)p)[i]);
    }

    internal static Action<long, float> Storer(Tensor t)
    {
        void* p = t.DataPointer;
        if (t.DType == DType.F32) return (i, v) => ((float*)p)[i] = v;
        if (t.DType == DType.F16) return (i, v) => ((Half*)p)[i] = (Half)v;
        return (i, v) => ((ushort*)p)[i] = BFloat16Bits.FromFloat(v);
    }
}
