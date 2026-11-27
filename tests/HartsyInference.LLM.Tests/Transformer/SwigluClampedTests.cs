using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests.Transformer;

/// <summary>Hy4 Flash routed experts: <c>silu(min(gate, limit)) * clamp(up, -limit, limit)</c>, the plain SwiGLU with clamped inputs (not the alpha form of MiniMax).</summary>
public sealed class SwigluClampedTests
{
    [Fact]
    public void Clamps_GateAboveAndUpOnBothSides_ThenPlainSilu()
    {
        float[] gate = [-30f, -2f, 0.5f, 12f, 40f, -11f];
        float[] up = [25f, -25f, 3f, 11f, -9f, 10f];
        IBackend backend = new CpuBackend();
        Tensor g = Make(gate), u = Make(up);
        using Tensor result = GatedFfn.SwiGlu(backend, ActivationKind.SwigluClamped, g, u, 10f, 0f, 0f, 0f);
        for (int i = 0; i < gate.Length; i++)
        {
            float cg = MathF.Min(gate[i], 10f), cu = Math.Clamp(up[i], -10f, 10f);
            Assert.Equal(cg / (1f + MathF.Exp(-cg)) * cu, result.AsReadOnlySpan<float>()[i], 4);
        }
    }

    [Fact]
    public void WithoutALimit_ItRefusesRatherThanRunningUnclamped()
    {
        IBackend backend = new CpuBackend();
        Assert.Throws<InvalidOperationException>(() => GatedFfn.SwiGlu(backend, ActivationKind.SwigluClamped, Make([1f]), Make([1f]), 0f, 0f, 0f, 0f));
    }

    private static Tensor Make(float[] values)
    {
        Tensor t = new(new TensorShape(1, 1, values.Length), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
