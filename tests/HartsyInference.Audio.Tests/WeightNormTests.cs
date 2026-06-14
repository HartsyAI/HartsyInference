using HartsyInference.Audio.Layers;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Verifies <see cref="WeightNorm.Compose"/> produces the same effective
/// weight tensor as PyTorch's <c>torch.nn.utils.weight_norm</c> hook. The reparam
/// stores magnitude in <c>weight_g</c> and direction in <c>weight_v</c>; the live
/// weight is <c>W = v * (g / ||v||)</c> where the norm is taken along every axis
/// except dim 0 (the output-channel axis).</summary>
public sealed unsafe class WeightNormTests
{
    [Fact]
    public void ComposeMatchesClosedForm_Conv1dShape()
    {
        // weight_v [out=3, in=2, K=4]; weight_g [3, 1, 1]
        int outC = 3, inC = 2, k = 4;
        Tensor v = new(new TensorShape(outC, inC, k), DType.F32);
        Tensor g = new(new TensorShape(outC, 1, 1), DType.F32);
        try
        {
            float* vp = (float*)v.DataPointer;
            float* gp = (float*)g.DataPointer;
            Random rng = new(7);
            for (long i = 0; i < v.ElementCount; i++) vp[i] = (float)(rng.NextDouble() * 2 - 1);
            gp[0] = 0.5f; gp[1] = 2.0f; gp[2] = 1.0f;

            using Tensor composed = WeightNorm.Compose(g, v);
            Assert.Equal(v.Shape, composed.Shape);
            float* wp = (float*)composed.DataPointer;

            int perOut = inC * k;
            for (int oc = 0; oc < outC; oc++)
            {
                double sumSq = 0d;
                int baseIdx = oc * perOut;
                for (int j = 0; j < perOut; j++) sumSq += vp[baseIdx + j] * vp[baseIdx + j];
                float norm = (float)Math.Sqrt(sumSq);
                float scale = gp[oc] / norm;
                for (int j = 0; j < perOut; j++)
                    Assert.Equal(vp[baseIdx + j] * scale, wp[baseIdx + j], precision: 5);
            }
        }
        finally
        {
            v.Dispose(); g.Dispose();
        }
    }

    [Fact]
    public void Compose_AcceptsFlatWeightGVector()
    {
        // PyTorch by convention saves weight_g with same rank as weight_v (dims-after-0
        // size 1), but our helper also accepts a flat [out] tensor — verify.
        int outC = 2, inC = 1, k = 3;
        Tensor v = new(new TensorShape(outC, inC, k), DType.F32);
        Tensor gFlat = new(new TensorShape(outC), DType.F32);
        try
        {
            float* vp = (float*)v.DataPointer;
            float* gp = (float*)gFlat.DataPointer;
            for (int i = 0; i < outC * inC * k; i++) vp[i] = i + 1;     // 1..6
            gp[0] = 1f; gp[1] = 4f;

            using Tensor composed = WeightNorm.Compose(gFlat, v);
            float* wp = (float*)composed.DataPointer;

            // OC 0: ||v|| = sqrt(1+4+9) = sqrt(14); scale = 1/sqrt(14)
            float n0 = MathF.Sqrt(14f);
            for (int j = 0; j < k; j++) Assert.Equal((j + 1f) / n0, wp[j], precision: 5);
            // OC 1: v = [4, 5, 6], ||v|| = sqrt(16+25+36) = sqrt(77); scale = 4/sqrt(77)
            float n1 = MathF.Sqrt(77f);
            for (int j = 0; j < k; j++) Assert.Equal((j + 4f) * 4f / n1, wp[k + j], precision: 4);
        }
        finally
        {
            v.Dispose(); gFlat.Dispose();
        }
    }

    [Fact]
    public void Compose_ThroughDictionaryLookup()
    {
        // Compose(IReadOnlyDictionary, string prefix) is the loader-facing helper.
        int outC = 2, inC = 1, k = 2;
        Tensor v = new(new TensorShape(outC, inC, k), DType.F32);
        Tensor g = new(new TensorShape(outC, 1, 1), DType.F32);
        try
        {
            float* vp = (float*)v.DataPointer;
            float* gp = (float*)g.DataPointer;
            vp[0] = 3f; vp[1] = 4f;     // OC0: ||v|| = 5
            vp[2] = 6f; vp[3] = 8f;     // OC1: ||v|| = 10
            gp[0] = 5f; gp[1] = 1f;

            Dictionary<string, Tensor> w = new()
            {
                ["conv.weight_g"] = g,
                ["conv.weight_v"] = v,
            };
            using Tensor composed = WeightNorm.Compose(w, "conv");
            float* wp = (float*)composed.DataPointer;
            // OC 0: scale = 5/5 = 1 → output = v
            Assert.Equal(3f, wp[0], precision: 5);
            Assert.Equal(4f, wp[1], precision: 5);
            // OC 1: scale = 1/10 → output = v / 10
            Assert.Equal(0.6f, wp[2], precision: 5);
            Assert.Equal(0.8f, wp[3], precision: 5);
        }
        finally
        {
            v.Dispose(); g.Dispose();
        }
    }
}
