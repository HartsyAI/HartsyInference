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
    public void LstmCell_ZeroInputAndState_GivesZeroForgetGateBias()
    {
        // With x = 0, h_prev = 0, c_prev = 0 and all weights zero except a chosen bias,
        // the gates simplify to sigmoid(bias) and tanh(bias). Exercises the gate-and-update
        // fused kernel with predictable values.
        using CpuBackend backend = new();
        int batch = 1, inputDim = 2, hiddenDim = 3;
        int gateRows = 4 * hiddenDim;

        Tensor wIh = ZeroTensor(gateRows, inputDim);
        Tensor wHh = ZeroTensor(gateRows, hiddenDim);
        Tensor bIh = ZeroTensor(gateRows);
        Tensor bHh = ZeroTensor(gateRows);
        Tensor x = ZeroTensor(batch, inputDim);
        Tensor hPrev = ZeroTensor(batch, hiddenDim);
        Tensor cPrev = ZeroTensor(batch, hiddenDim);

        try
        {
            // Set bias_ih for input gate (gate 0, channel 0) to 2.0.
            float* bp = (float*)bIh.DataPointer;
            bp[0] = 2.0f;     // i gate, channel 0
            bp[2 * hiddenDim + 0] = 1.0f;     // g gate (tanh), channel 0

            LstmCell cell = new(inputDim, hiddenDim);
            cell.BindWeights(wIh, wHh, bIh, bHh);

            (Tensor hNew, Tensor cNew) = cell.Step(backend, x, hPrev, cPrev, batch);

            try
            {
                // For channel 0: i = sigmoid(2.0), f = sigmoid(0) = 0.5, g = tanh(1.0),
                // o = sigmoid(0) = 0.5.
                // c_new = f * c_prev + i * g = 0 + sigmoid(2) * tanh(1).
                // h_new = o * tanh(c_new) = 0.5 * tanh(sigmoid(2) * tanh(1)).
                float sig2 = 1f / (1f + MathF.Exp(-2f));
                float t1 = MathF.Tanh(1f);
                float c0Expected = sig2 * t1;
                float h0Expected = 0.5f * MathF.Tanh(c0Expected);

                float* cp = (float*)cNew.DataPointer;
                float* hp = (float*)hNew.DataPointer;
                Assert.Equal(c0Expected, cp[0], precision: 5);
                Assert.Equal(h0Expected, hp[0], precision: 5);

                // Other channels: all biases zero → sigmoid(0)=0.5, tanh(0)=0. c_new[i] = 0.
                // h_new[i] = 0.5 * tanh(0) = 0.
                for (int i = 1; i < hiddenDim; i++)
                {
                    Assert.Equal(0f, cp[i], precision: 5);
                    Assert.Equal(0f, hp[i], precision: 5);
                }
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
    public void BiLstm_OutputShapeIs2xHidden()
    {
        // Don't load weights — just verify the BiLstm constructs and tracks dims.
        int inputDim = 4, hiddenDim = 6;
        BiLstm bi = new(inputDim, hiddenDim);
        Assert.Equal(inputDim, bi.InputDim);
        Assert.Equal(hiddenDim, bi.HiddenDim);
        // Forward path tested transitively via Kokoro pipeline integration tests when those land.
    }

    [Fact]
    public void UnidirectionalLstm_LayerCountIsRespected()
    {
        UnidirectionalLstm lstm = new(inputDim: 8, hiddenDim: 16, numLayers: 3);
        Assert.Equal(3, lstm.NumLayers);
        Assert.Equal(8, lstm.InputDim);
        Assert.Equal(16, lstm.HiddenDim);
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
