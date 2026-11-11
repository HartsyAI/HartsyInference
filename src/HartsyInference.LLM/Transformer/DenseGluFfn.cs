using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>A dense gated FFN, <c>down(act(gate(x)) · up(x))</c>, over one sequence of rows. Any activation <see cref="GatedFfn.SwiGlu"/>
/// takes works, so SiTU, SwiGLU and GeGLU families share this path.</summary>
public static class DenseGluFfn
{
    /// <summary>Runs the FFN on <paramref name="input"/> <c>[1, tokens, hidden]</c> into <paramref name="output"/> (same shape).</summary>
    /// <param name="gateWeight">The gate projection, <c>[intermediate, hidden]</c>.</param>
    /// <param name="upWeight">The up projection, <c>[intermediate, hidden]</c>.</param>
    /// <param name="downWeight">The down projection, <c>[hidden, intermediate]</c>.</param>
    public static void Apply(IBackend backend, Tensor input, Tensor gateWeight, Tensor upWeight, Tensor downWeight, Tensor output,
        int intermediate, ActivationKind activation, float situBeta, float situLinearBeta)
    {
        ArgumentNullException.ThrowIfNull(backend);
        long tokens = input.Shape[1];
        Tensor gate = new(new TensorShape(1, tokens, intermediate), DType.F32);
        Tensor up = new(new TensorShape(1, tokens, intermediate), DType.F32);
        ProjectionOps.Project(backend, gate, input, gateWeight, null, lowVram: false);
        ProjectionOps.Project(backend, up, input, upWeight, null, lowVram: false);
        // SwiGlu disposes gate and up and returns act(gate) · up.
        using Tensor act = GatedFfn.SwiGlu(backend, activation, gate, up, 0f, 0f, situBeta, situLinearBeta);
        ProjectionOps.Project(backend, output, act, downWeight, null, lowVram: false);
    }
}
