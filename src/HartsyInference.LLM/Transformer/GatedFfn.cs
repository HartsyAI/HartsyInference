using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>The gated-FFN activation epilogue shared by dense and expert FFNs: <c>act(gate) · up</c>, optionally with the clamped-SwiGLU limit.</summary>
internal static class GatedFfn
{
    /// <summary>Applies the FFN activation to <paramref name="inp"/> into <paramref name="outp"/> (same shape): SiLU, tanh-GELU, ReLU, or ReLU².
    /// Any other activation throws, so a gated-only one (SiTU, clamped SwiGLU) never runs as SiLU.</summary>
    public static void Activate(IBackend backend, ActivationKind activation, Tensor outp, Tensor inp)
    {
        switch (activation)
        {
            case ActivationKind.Silu: backend.Silu(outp, inp); break;
            case ActivationKind.GeluTanh: backend.Gelu(outp, inp); break;
            case ActivationKind.Relu: backend.Clamp(outp, inp, 0f, float.PositiveInfinity); break;
            case ActivationKind.ReluSquared:
                backend.Clamp(outp, inp, 0f, float.PositiveInfinity);
                backend.Mul(outp, outp, outp);   // relu(x)² (elementwise, alias-safe)
                break;
            default:
                throw new NotSupportedException($"FFN activation {activation} has no elementwise form; it runs only through its gated path.");
        }
    }

    /// <summary>The GLU spec of a gated FFN over <c>[gate | up]</c>. Throws for an activation with no GLU form.</summary>
    public static GluSpec GluFor(ActivationKind activation, float situBeta, float situLinearBeta) => activation switch
    {
        ActivationKind.Silu => new GluSpec(GluActivation.Silu, GluGateHalf.First),
        ActivationKind.GeluTanh => new GluSpec(GluActivation.GeluTanh, GluGateHalf.First),
        ActivationKind.Situ => new GluSpec(GluActivation.Situ, GluGateHalf.First, situBeta, situLinearBeta),
        _ => throw new NotSupportedException($"FFN activation {activation} has no GLU form."),
    };

    /// <summary>SiTU-GLU over separate gate and up tensors: packs them as <c>[gate | up]</c> and runs one <see cref="IBackend.Glu"/>. Disposes both.</summary>
    public static Tensor SituGlu(IBackend backend, Tensor gate, Tensor up, float situBeta, float situLinearBeta)
    {
        TensorShape outShape = gate.Shape;
        Span<long> packedDims = stackalloc long[outShape.Rank];
        outShape.CopyDimsTo(packedDims);
        packedDims[^1] *= 2;
        Tensor packed = new(new TensorShape(packedDims), DType.F32);
        // Packing costs a 2x copy on this unfused fallback, but keeps SiTU in the one Glu op every backend checks.
        backend.Concat(packed, [gate, up], outShape.Rank - 1);
        gate.Dispose();
        up.Dispose();
        Tensor comb = new(outShape, DType.F32);
        backend.Glu(comb, packed, GluFor(ActivationKind.Situ, situBeta, situLinearBeta));
        packed.Dispose();
        return comb;
    }

    /// <summary>Returns <c>act(gate) · up</c> and disposes <paramref name="gate"/> and <paramref name="up"/>; a positive <paramref name="clampLimit"/> first clamps gate to (-inf, limit] and up to [-limit, limit].
    /// The clamped alpha-SwiGLU reads <paramref name="alpha"/> and <paramref name="clampLimit"/>; SiTU reads the two betas.</summary>
    public static Tensor SwiGlu(IBackend backend, ActivationKind activation, Tensor gate, Tensor up, float clampLimit, float alpha,
        float situBeta, float situLinearBeta)
    {
        if (activation == ActivationKind.SwigluOai) return SwigluOai(backend, gate, up, alpha, clampLimit);
        if (activation == ActivationKind.Situ) return SituGlu(backend, gate, up, situBeta, situLinearBeta);
        Tensor gateAct = new(gate.Shape, DType.F32);
        if (clampLimit > 0f)
        {
            Tensor gateClamped = new(gate.Shape, DType.F32);
            backend.Clamp(gateClamped, gate, float.NegativeInfinity, clampLimit);
            Activate(backend, activation, gateAct, gateClamped);
            gateClamped.Dispose();
            Tensor upClamped = new(up.Shape, DType.F32);
            backend.Clamp(upClamped, up, -clampLimit, clampLimit);
            up.Dispose();
            up = upClamped;
        }
        else
        {
            Activate(backend, activation, gateAct, gate);
        }
        gate.Dispose();
        Tensor comb = new(gateAct.Shape, DType.F32);
        backend.Mul(comb, gateAct, up);
        gateAct.Dispose();
        up.Dispose();
        return comb;
    }

    /// <summary>MiniMax's clamped alpha-SwiGLU, <c>(clamp(up, ±limit) + 1) · g · sigmoid(alpha · g)</c> with <c>g = min(gate, limit)</c>, from backend ops so every backend runs it. Disposes <paramref name="gate"/> and <paramref name="up"/>.</summary>
    private static Tensor SwigluOai(IBackend backend, Tensor gate, Tensor up, float alpha, float limit)
    {
        if (alpha <= 0f || limit <= 0f) throw new InvalidOperationException("The clamped alpha-SwiGLU needs a positive alpha and limit.");
        Tensor g = new(gate.Shape, DType.F32);
        backend.Clamp(g, gate, float.NegativeInfinity, limit);
        gate.Dispose();
        Tensor scaled = new(g.Shape, DType.F32);
        backend.Scale(scaled, g, alpha);
        Tensor sigmoid = new(g.Shape, DType.F32);
        backend.Sigmoid(sigmoid, scaled);
        scaled.Dispose();
        Tensor glu = new(g.Shape, DType.F32);
        backend.Mul(glu, g, sigmoid);
        g.Dispose();
        sigmoid.Dispose();
        Tensor clamped = new(up.Shape, DType.F32);
        backend.Clamp(clamped, up, -limit, limit);
        up.Dispose();
        Tensor shifted = new(clamped.Shape, DType.F32);
        backend.AddScalar(shifted, clamped, 1f);
        clamped.Dispose();
        Tensor comb = new(glu.Shape, DType.F32);
        backend.Mul(comb, shifted, glu);
        glu.Dispose();
        shifted.Dispose();
        return comb;
    }
}
