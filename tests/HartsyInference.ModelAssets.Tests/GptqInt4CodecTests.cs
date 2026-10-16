using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf.Codecs;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>A symmetric group-128 GPTQ weight is a Q4_0 weight with each group's scale repeated over its blocks: the repack must
/// decode to exactly the GPTQ values, and anything not symmetric or not in input order must be refused rather than misread.</summary>
public sealed unsafe class GptqInt4CodecTests
{
    private const int In = 256, Out = 16, Group = 128;

    [Theory]
    [InlineData(0x77777777u)]
    [InlineData(0x88888888u)]
    public void ToQ4_0_DecodesToExactlyTheGptqValues(uint zeroWord)
    {
        (Tensor qweight, Tensor scales, Tensor qzeros, int[,] q, Half[,] s) = Synthetic(zeroWord, seed: 3);
        using (qweight) using (scales) using (qzeros)
        {
            using Tensor q4 = GptqInt4Codec.ToQ4_0(qweight, scales, qzeros, gIdx: null, Group);
            Assert.Equal(DType.Q4_0, q4.DType);
            Assert.Equal(new TensorShape(Out, In), q4.Shape);
            float[] decoded = new float[Out * In];
            fixed (float* dst = decoded) new Codec_Q4_0().DequantizeToF32((byte*)q4.DataPointer, dst, decoded.Length);
            using Tensor reference = GptqInt4Codec.Dequantize(qweight, scales, qzeros, gIdx: null, Group);
            ReadOnlySpan<float> expected = reference.AsReadOnlySpan<float>();
            for (int o = 0; o < Out; o++)
            {
                for (int i = 0; i < In; i++)
                {
                    float want = (float)s[i / Group, o] * (q[o, i] - 8);
                    Assert.Equal(want, expected[o * In + i]);
                    Assert.Equal(want, decoded[o * In + i]);
                }
            }
        }
    }

    [Fact]
    public void AsymmetricZeros_AreRefused()
    {
        (Tensor qweight, Tensor scales, Tensor qzeros, _, _) = Synthetic(0x77777777u, seed: 4);
        using (qweight) using (scales) using (qzeros)
        {
            qzeros.AsSpan<uint>()[1] = 0x77777776u;
            Assert.Throws<NotSupportedException>(() => GptqInt4Codec.ToQ4_0(qweight, scales, qzeros, null, Group));
        }
    }

    [Fact]
    public void ActivationOrder_IsRefused_AndTheIdentityOrderAccepted()
    {
        (Tensor qweight, Tensor scales, Tensor qzeros, _, _) = Synthetic(0x77777777u, seed: 5);
        using (qweight) using (scales) using (qzeros)
        {
            using Tensor order = new(new TensorShape(In), DType.I32);
            Span<int> g = order.AsSpan<int>();
            for (int i = 0; i < In; i++) g[i] = i / Group;
            GptqInt4Codec.ToQ4_0(qweight, scales, qzeros, order, Group).Dispose();
            (g[0], g[In - 1]) = (g[In - 1], g[0]);
            Assert.Throws<NotSupportedException>(() => GptqInt4Codec.ToQ4_0(qweight, scales, qzeros, order, Group));
        }
    }

    // qweight packs eight inputs per I32 word along the input axis, lowest nibble first.
    private static (Tensor QWeight, Tensor Scales, Tensor QZeros, int[,] Q, Half[,] S) Synthetic(uint zeroWord, int seed)
    {
        Random random = new(seed);
        int[,] q = new int[Out, In];
        Half[,] s = new Half[In / Group, Out];
        Tensor qweight = new(new TensorShape(In / 8, Out), DType.I32);
        Tensor scales = new(new TensorShape(In / Group, Out), DType.F16);
        Tensor qzeros = new(new TensorShape(In / Group, Out / 8), DType.I32);
        Span<uint> words = qweight.AsSpan<uint>();
        words.Clear();
        for (int o = 0; o < Out; o++)
        {
            for (int i = 0; i < In; i++)
            {
                q[o, i] = random.Next(16);
                words[(i / 8) * Out + o] |= (uint)q[o, i] << (4 * (i % 8));
            }
        }
        Span<Half> sv = scales.AsSpan<Half>();
        for (int g = 0; g < In / Group; g++)
        {
            for (int o = 0; o < Out; o++) sv[g * Out + o] = s[g, o] = (Half)(0.001f + random.NextSingle() * 0.05f);
        }
        qzeros.AsSpan<uint>().Fill(zeroWord);
        return (qweight, scales, qzeros, q, s);
    }
}
