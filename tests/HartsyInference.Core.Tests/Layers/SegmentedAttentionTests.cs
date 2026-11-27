using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

public sealed class SegmentedAttentionTests
{
    private const int Heads = 3, HeadDim = 8, Tokens = 18;
    private static readonly IBackend Backend = new CpuBackend();

    public static TheoryData<string> Plans => new() { "base", "referenced", "skip", "dense" };

    private static AttentionSegment[] Plan(string name) => name switch
    {
        "dense" => ConditionedAttentionPlan.ForConditionedFrames(2, 9, 0),
        "base" => ConditionedAttentionPlan.ForConditionedFrames(2, 9, 2),
        "referenced" => ConditionedAttentionPlan.ForReferencedFrames(2, 9, 3, 1, 4, 0),
        _ => ConditionedAttentionPlan.ForReferencedFrames(2, 9, 3, 1, 4, 1),
    };

    /// <summary>The segments are one attention with a −∞ mask: each query row sees exactly its segment's key window.</summary>
    [Theory]
    [MemberData(nameof(Plans))]
    public void MatchesMaskedSoftmaxAttention(string name)
    {
        AttentionSegment[] plan = Plan(name);
        Random rng = new(7);
        float[] q = Fill(rng), k = Fill(rng), v = Fill(rng);
        using Tensor qt = FromArray(q), kt = FromArray(k), vt = FromArray(v);
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, ComputeDType = DType.F32 });

        using Tensor result = SegmentedAttention.Run(Backend, layer, qt, kt, vt, plan);

        Assert.Equal(new TensorShape(Tokens, Heads * HeadDim), result.Shape);
        float[] expected = MaskedAttention(q, k, v, plan);
        ReadOnlySpan<float> got = result.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - got[i]) <= 2e-5f * (1f + MathF.Abs(expected[i])), $"{name}[{i}]: {got[i]} vs {expected[i]}");
    }

    [Fact]
    public void ConditioningRowsIgnoreTheNoisyOperands()
    {
        AttentionSegment[] plan = Plan("base");
        Random rng = new(11);
        float[] q = Fill(rng), k = Fill(rng), v = Fill(rng);
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, ComputeDType = DType.F32 });
        float[] condBefore = Run(layer, q, k, v, plan, 4 * Heads * HeadDim);

        float[] k2 = (float[])k.Clone(), v2 = (float[])v.Clone(), q2 = (float[])q.Clone();
        for (int i = 4 * Heads * HeadDim; i < k2.Length; i++)
        {
            k2[i] += 1.5f;
            v2[i] -= 0.75f;
            q2[i] *= -2f;
        }
        float[] condAfter = Run(layer, q2, k2, v2, plan, 4 * Heads * HeadDim);

        Assert.Equal(condBefore, condAfter);
    }

    [Fact]
    public void RunLeavesTheOperandsToTheCaller()
    {
        AttentionSegment[] plan = Plan("skip");
        Random rng = new(3);
        float[] q = Fill(rng), k = Fill(rng), v = Fill(rng);
        using Tensor qt = FromArray(q), kt = FromArray(k), vt = FromArray(v);
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, ComputeDType = DType.F32 });

        using Tensor result = SegmentedAttention.Run(Backend, layer, qt, kt, vt, plan);

        Assert.Equal(q, qt.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(k, kt.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(v, vt.AsReadOnlySpan<float>().ToArray());
    }

    private static float[] Run(AttentionLayer layer, float[] q, float[] k, float[] v, AttentionSegment[] plan, int takeFirst)
    {
        using Tensor qt = FromArray(q), kt = FromArray(k), vt = FromArray(v);
        using Tensor result = SegmentedAttention.Run(Backend, layer, qt, kt, vt, plan);
        return result.AsReadOnlySpan<float>()[..takeFirst].ToArray();
    }

    private static float[] MaskedAttention(float[] q, float[] k, float[] v, AttentionSegment[] plan)
    {
        int dim = Heads * HeadDim;
        float[] output = new float[Tokens * dim];
        float scale = 1f / MathF.Sqrt(HeadDim);
        foreach (AttentionSegment s in plan)
            for (int row = s.QueryStart; row < s.QueryStart + s.QueryCount; row++)
                for (int h = 0; h < Heads; h++)
                {
                    double[] scores = new double[s.KeyCount];
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < s.KeyCount; j++)
                    {
                        double dot = 0;
                        for (int d = 0; d < HeadDim; d++)
                            dot += q[row * dim + h * HeadDim + d] * (double)k[(s.KeyStart + j) * dim + h * HeadDim + d];
                        scores[j] = dot * scale;
                        max = Math.Max(max, scores[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < s.KeyCount; j++)
                    {
                        scores[j] = Math.Exp(scores[j] - max);
                        sum += scores[j];
                    }
                    for (int d = 0; d < HeadDim; d++)
                    {
                        double acc = 0;
                        for (int j = 0; j < s.KeyCount; j++)
                            acc += scores[j] / sum * v[(s.KeyStart + j) * dim + h * HeadDim + d];
                        output[row * dim + h * HeadDim + d] = (float)acc;
                    }
                }
        return output;
    }

    private static float[] Fill(Random rng)
    {
        float[] values = new float[Tokens * Heads * HeadDim];
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    private static Tensor FromArray(float[] values)
    {
        Tensor t = new(new TensorShape(Tokens, Heads, HeadDim), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
