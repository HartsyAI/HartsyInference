using HartsyInference.Audio.Layers;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the GRU layer (OpenVoice's tone-color ReferenceEncoder, reusable elsewhere).
/// Golden values come from <c>torch.nn.GRU(2, 3, batch_first=True)</c> with weights initialized to
/// <c>arange * 0.01 - 0.05</c>; a regression here means PyTorch's gate order (r, z, n) or the reset-gate-on-
/// hidden convention is wrong.</summary>
public sealed unsafe class GruTests
{
    private const int Input = 2, Hidden = 3, T = 2;

    [Fact]
    public void Gru_MatchesTorchNnGru_FullSequenceAndLastHidden()
    {
        using CpuBackend backend = new();

        // weight_ih_l0 [3H, in], weight_hh_l0 [3H, H], biases [3H]; filled arange*0.01 - 0.05 like the reference.
        Tensor wIh = Arange(3 * Hidden, Input);
        Tensor wHh = Arange(3 * Hidden, Hidden);
        Tensor bIh = Arange(3 * Hidden);
        Tensor bHh = Arange(3 * Hidden);

        Gru gru = new(Input, Hidden);
        Dictionary<string, Tensor> w = new()
        {
            ["g.weight_ih_l0"] = wIh,
            ["g.weight_hh_l0"] = wHh,
            ["g.bias_ih_l0"] = bIh,
            ["g.bias_hh_l0"] = bHh,
        };
        gru.LoadWeights(w, "g");

        Tensor x = new(new TensorShape(1, T, Input), DType.F32);
        float* xp = (float*)x.DataPointer;
        xp[0] = 0.1f; xp[1] = 0.2f; xp[2] = 0.3f; xp[3] = -0.1f;

        float[] expectedSeq = [0.019179802f, 0.029380824f, 0.039350316f, 0.026513595f, 0.041613489f, 0.056525216f];
        float[] expectedLast = [0.026513595f, 0.041613489f, 0.056525216f];

        Tensor outSeq = gru.Forward(backend, x, 1, T);
        try
        {
            float* op = (float*)outSeq.DataPointer;
            for (int i = 0; i < expectedSeq.Length; i++) Assert.Equal(expectedSeq[i], op[i], precision: 5);
        }
        finally { outSeq.Dispose(); }

        Tensor last = gru.LastHidden(backend, x, 1, T);
        try
        {
            float* lp = (float*)last.DataPointer;
            for (int i = 0; i < expectedLast.Length; i++) Assert.Equal(expectedLast[i], lp[i], precision: 5);
        }
        finally { last.Dispose(); }

        x.Dispose();
        wIh.Dispose(); wHh.Dispose(); bIh.Dispose(); bHh.Dispose();
    }

    private static Tensor Arange(int dim0, int dim1)
    {
        Tensor t = new(new TensorShape(dim0, dim1), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = i * 0.01f - 0.05f;
        return t;
    }

    private static Tensor Arange(int dim0)
    {
        Tensor t = new(new TensorShape(dim0), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = i * 0.01f - 0.05f;
        return t;
    }
}
