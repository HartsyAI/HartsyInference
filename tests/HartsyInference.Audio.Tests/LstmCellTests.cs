using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical correctness tests for the LSTM cell (used by Kokoro prosody
/// predictor, GPT-SoVITS, and EnCodec's bottleneck). The hand-computed reference
/// uses PyTorch's gate order (i, f, g, o) and the standard sigmoid/tanh formulas —
/// a regression here breaks every TTS that runs an LSTM.</summary>
public sealed unsafe class LstmCellTests
{
    [Fact]
    public void LstmCell_RecallsCellStateThroughForgetGate()
    {
        // Set forget-gate bias high, all other gates near zero → cell carries c_prev forward.
        using CpuBackend backend = new();
        int batch = 1, inputDim = 1, hiddenDim = 1;

        Tensor wIh = ZeroTensor(4, inputDim);
        Tensor wHh = ZeroTensor(4, hiddenDim);
        Tensor bIh = ZeroTensor(4);
        Tensor bHh = ZeroTensor(4);
        Tensor x = ZeroTensor(batch, inputDim);
        Tensor hPrev = ZeroTensor(batch, hiddenDim);
        Tensor cPrev = ZeroTensor(batch, hiddenDim);

        try
        {
            // Forget-gate bias = 10 → sigmoid(10) ≈ 1.0; cell state survives.
            float* bp = (float*)bIh.DataPointer;
            bp[1] = 10f;     // f gate, channel 0
            float* cp0 = (float*)cPrev.DataPointer;
            cp0[0] = 0.7f;

            LstmCell cell = new(inputDim, hiddenDim);
            cell.BindWeights(wIh, wHh, bIh, bHh);

            (Tensor hNew, Tensor cNew) = cell.Step(backend, x, hPrev, cPrev, batch);
            try
            {
                float* cp = (float*)cNew.DataPointer;
                // c_new = sigmoid(10) * 0.7 + sigmoid(0) * tanh(0) ≈ 1 * 0.7 + 0 = 0.7.
                Assert.Equal(0.7f, cp[0], precision: 4);
            }
            finally
            {
                hNew.Dispose();
                cNew.Dispose();
            }
        }
        finally
        {
            wIh.Dispose(); wHh.Dispose(); bIh.Dispose(); bHh.Dispose();
            x.Dispose(); hPrev.Dispose(); cPrev.Dispose();
        }
    }

    [Fact]
    public unsafe void WeightNormFusion_ProducesUnitWeightsWhenGEqualsNorm()
    {
        // Now reachable via InternalsVisibleTo: when weight_g[oc] equals ||weight_v[oc]||,
        // the fused weight equals weight_v.
        int outCh = 3, inCh = 2, kernel = 4;
        Tensor v = new(new TensorShape(outCh, inCh, kernel), DType.F32);
        Tensor g = new(new TensorShape(outCh), DType.F32);
        try
        {
            float* vp = (float*)v.DataPointer;
            float* gp = (float*)g.DataPointer;
            Random rng = new(99);
            for (int oc = 0; oc < outCh; oc++)
            {
                double sumSq = 0d;
                for (int ic = 0; ic < inCh; ic++)
                {
                    for (int k = 0; k < kernel; k++)
                    {
                        float vv = (float)(rng.NextDouble() - 0.5);
                        vp[(oc * inCh + ic) * kernel + k] = vv;
                        sumSq += (double)vv * vv;
                    }
                }
                gp[oc] = (float)Math.Sqrt(sumSq);
            }

            Tensor fused = WeightNormFusion.Fuse(g, v);
            try
            {
                float* fp = (float*)fused.DataPointer;
                for (long i = 0; i < v.ElementCount; i++)
                    Assert.Equal(vp[i], fp[i], precision: 4);
            }
            finally
            {
                fused.Dispose();
            }
        }
        finally
        {
            v.Dispose();
            g.Dispose();
        }
    }

    private static Tensor ZeroTensor(int dim0)
    {
        Tensor t = new(new TensorShape(dim0), DType.F32);
        Span<float> s = t.AsSpan<float>();
        s.Clear();
        return t;
    }

    private static Tensor ZeroTensor(int dim0, int dim1)
    {
        Tensor t = new(new TensorShape(dim0, dim1), DType.F32);
        Span<float> s = t.AsSpan<float>();
        s.Clear();
        return t;
    }
}
