using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>The gated-FFN activation epilogue shared by dense and expert FFNs: <c>act(gate) · up</c>, optionally with the clamped-SwiGLU limit.</summary>
internal static class GatedFfn
{
    /// <summary>Applies the FFN activation to <paramref name="inp"/> into <paramref name="outp"/> (same shape): SiLU, tanh-GELU, ReLU, or ReLU².</summary>
    public static void Activate(IBackend backend, ActivationKind activation, Tensor outp, Tensor inp)
    {
        switch (activation)
        {
            case ActivationKind.GeluTanh: backend.Gelu(outp, inp); break;
            case ActivationKind.Relu: backend.Clamp(outp, inp, 0f, float.PositiveInfinity); break;
            case ActivationKind.ReluSquared:
                backend.Clamp(outp, inp, 0f, float.PositiveInfinity);
                backend.Mul(outp, outp, outp);   // relu(x)² (elementwise, alias-safe)
                break;
            default: backend.Silu(outp, inp); break;   // SiLU / SwiGLU
        }
    }

    /// <summary>Returns <c>act(gate) · up</c> and disposes <paramref name="gate"/> and <paramref name="up"/>; a positive <paramref name="clampLimit"/> first clamps gate to (-inf, limit] and up to [-limit, limit].</summary>
    public static Tensor SwiGlu(IBackend backend, ActivationKind activation, Tensor gate, Tensor up, float clampLimit = 0f)
    {
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
}
