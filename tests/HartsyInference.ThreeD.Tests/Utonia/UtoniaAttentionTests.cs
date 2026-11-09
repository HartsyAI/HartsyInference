using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Utonia;

public sealed unsafe class UtoniaAttentionTests
{
    private const int Channels = 12;
    private const int Heads = 2;

    [Theory]
    [InlineData(8, 8)]
    [InlineData(8, 4)]
    public void Forward_AttendsOnlyWithinEachPatchOfTheSerializedOrder(int n, int patch)
    {
        CpuBackend backend = new();
        Random rng = new(17);
        float[] feat = Rand(rng, n * Channels);
        float[] wQkv = Rand(rng, 3 * Channels * Channels), bQkv = Rand(rng, 3 * Channels);
        float[] wProj = Rand(rng, Channels * Channels), bProj = Rand(rng, Channels);
        UtoniaLinear qkv = new(backend, UtoniaTestData.Make(new long[] { 3 * Channels, Channels }, wQkv), UtoniaTestData.Make(new long[] { 3 * Channels }, bQkv));
        UtoniaLinear proj = new(backend, UtoniaTestData.Make(new long[] { Channels, Channels }, wProj), UtoniaTestData.Make(new long[] { Channels }, bProj));
        int[] grid = Enumerable.Range(0, n * 3).Select(_ => rng.Next(16)).ToArray();
        float[] coord = grid.Select(v => v * 0.01f).ToArray();
        UtoniaPoint point = new(UtoniaTestData.Make(new long[] { n, Channels }, feat), coord, grid, new int[n]);
        point.Serialize(new[] { UtoniaSerialOrder.ZOrder });
        UtoniaSerializedAttention attention = new(backend, qkv, proj, Channels, Heads, patch, 10f, 0);

        float[] actual = UtoniaTestData.Read(attention.Forward(point.Feat, point));

        float[] expected = Reference(feat, n, patch, wQkv, bQkv, wProj, bProj, coord, point.Serialized!.Inverse[0]);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 4);
        }
    }

    [Fact]
    public void Forward_RowChunksMatchTheUnchunkedRunExactly()
    {
        CpuBackend backend = new();
        Random rng = new(29);
        int[] batch = Enumerable.Repeat(0, 12).Concat(Enumerable.Repeat(1, 5)).ToArray();   // mixed segment lengths: 8, 8, 5
        int n = batch.Length;
        float[] feat = Rand(rng, n * Channels);
        UtoniaLinear qkv = new(backend, UtoniaTestData.Make(new long[] { 3 * Channels, Channels }, Rand(rng, 3 * Channels * Channels)),
            UtoniaTestData.Make(new long[] { 3 * Channels }, Rand(rng, 3 * Channels)));
        UtoniaLinear proj = new(backend, UtoniaTestData.Make(new long[] { Channels, Channels }, Rand(rng, Channels * Channels)),
            UtoniaTestData.Make(new long[] { Channels }, Rand(rng, Channels)));
        int[] grid = Enumerable.Range(0, n * 3).Select(_ => rng.Next(16)).ToArray();
        UtoniaPoint point = new(UtoniaTestData.Make(new long[] { n, Channels }, feat), grid.Select(v => v * 0.01f).ToArray(), grid, batch);
        point.Serialize(new[] { UtoniaSerialOrder.ZOrder });

        float[] expected = UtoniaTestData.Read(new UtoniaSerializedAttention(backend, qkv, proj, Channels, Heads, 8, 10f, 0)
            .Forward(point.Feat, point));
        float[] actual = UtoniaTestData.Read(new UtoniaSerializedAttention(backend, qkv, proj, Channels, Heads, 8, 10f, 0, chunkBytes: 1)
            .Forward(point.Feat, point));

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
        }
    }

    [Fact]
    public void Rope_AtOriginIsIdentity()
    {
        float[] qkv = Rand(new Random(3), 3 * Channels);
        float[] before = (float[])qkv.Clone();
        fixed (float* p = qkv)
        {
            UtoniaRope.Apply(p, new float[] { 0f, 0f, 0f }, Channels, Heads, 10f);
        }

        for (int i = 0; i < qkv.Length; i++)
        {
            Assert.Equal(before[i], qkv[i], 6);
        }
    }

    [Fact]
    public void Rope_IsARotationOfEachChannelPair()
    {
        float[] qkv = Rand(new Random(5), 3 * Channels);
        float[] before = (float[])qkv.Clone();
        fixed (float* p = qkv)
        {
            UtoniaRope.Apply(p, new float[] { 0.3f, -0.2f, 1.1f }, Channels, Heads, 10f);
        }

        int headDim = Channels / Heads, chunk = headDim / 3, half = chunk / 2;
        for (int part = 0; part < 2; part++)
        {
            for (int h = 0; h < Heads; h++)
            {
                for (int a = 0; a < 3; a++)
                {
                    for (int i = 0; i < half; i++)
                    {
                        int at = part * Channels + h * headDim + a * chunk + i;
                        float normBefore = MathF.Sqrt(before[at] * before[at] + before[at + half] * before[at + half]);
                        float normAfter = MathF.Sqrt(qkv[at] * qkv[at] + qkv[at + half] * qkv[at + half]);
                        Assert.Equal(normBefore, normAfter, 5);
                    }
                }
            }
        }
    }

    /// <summary>Dense reference: RoPE on q and k, softmax over the keys in the same serialized patch, then the output projection.</summary>
    private static float[] Reference(float[] x, int n, int patch, float[] wQkv, float[] bQkv, float[] wProj, float[] bProj,
        float[] coord, int[] serialPos)
    {
        int c = Channels, d = c / Heads;
        float[] qkv = new float[n * 3 * c];
        for (int i = 0; i < n; i++)
        {
            for (int o = 0; o < 3 * c; o++)
            {
                float s = bQkv[o];
                for (int k = 0; k < c; k++)
                {
                    s += x[i * c + k] * wQkv[o * c + k];
                }

                qkv[i * 3 * c + o] = s;
            }
        }

        fixed (float* p = qkv)
        {
            UtoniaRope.Apply(p, coord, c, Heads, 10f);
        }

        float scale = (float)(1.0 / Math.Sqrt(d));
        float[] attn = new float[n * c];
        for (int i = 0; i < n; i++)
        {
            int patchOfI = serialPos[i] / patch;
            for (int h = 0; h < Heads; h++)
            {
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < n; j++)
                {
                    if (serialPos[j] / patch == patchOfI)
                    {
                        maxScore = Math.Max(maxScore, Dot(qkv, i, j, h, d, c) * scale);
                    }
                }

                float total = 0f;
                float[] weights = new float[n];
                for (int j = 0; j < n; j++)
                {
                    if (serialPos[j] / patch == patchOfI)
                    {
                        weights[j] = MathF.Exp(Dot(qkv, i, j, h, d, c) * scale - maxScore);
                        total += weights[j];
                    }
                }

                for (int t = 0; t < d; t++)
                {
                    float acc = 0f;
                    for (int j = 0; j < n; j++)
                    {
                        if (serialPos[j] / patch == patchOfI)
                        {
                            acc += weights[j] / total * qkv[j * 3 * c + 2 * c + h * d + t];
                        }
                    }

                    attn[i * c + h * d + t] = acc;
                }
            }
        }

        float[] y = new float[n * c];
        for (int i = 0; i < n; i++)
        {
            for (int o = 0; o < c; o++)
            {
                float s = bProj[o];
                for (int k = 0; k < c; k++)
                {
                    s += attn[i * c + k] * wProj[o * c + k];
                }

                y[i * c + o] = s;
            }
        }

        return y;
    }

    private static float Dot(float[] qkv, int i, int j, int h, int d, int c)
    {
        float s = 0f;
        for (int t = 0; t < d; t++)
        {
            s += qkv[i * 3 * c + h * d + t] * qkv[j * 3 * c + c + h * d + t];
        }

        return s;
    }

    private static float[] Rand(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)(rng.NextDouble() * 0.6 - 0.3);
        }

        return values;
    }
}
