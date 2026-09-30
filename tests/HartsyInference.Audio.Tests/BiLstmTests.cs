using HartsyInference.Audio.Layers;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary><see cref="BiLstm"/> runs one batched input projection and a host recurrence; this pins it to the
/// step-by-step <see cref="LstmCell.Step"/> definition on random weights, both directions, so a regression in
/// the stacked weights, the reverse walk, the output half offsets or the SIMD dot would show as a mismatch.
/// Kokoro and StyleTTS 2 (text encoder, duration encoder, shared prosody LSTM) go through this path.</summary>
public sealed unsafe class BiLstmTests
{
    [Theory]
    [InlineData(1, 7, 5, 4)]
    [InlineData(2, 9, 6, 8)]
    [InlineData(1, 13, 12, 19)]
    public void Forward_MatchesPerStepCellReference(int batch, int t, int inputDim, int hidden)
    {
        Random rng = new(1234 + t);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        foreach (string dir in new[] { "", "_reverse" })
        {
            weights[$"lstm.weight_ih_l0{dir}"] = RandomTensor(rng, 4 * hidden, inputDim);
            weights[$"lstm.weight_hh_l0{dir}"] = RandomTensor(rng, 4 * hidden, hidden);
            weights[$"lstm.bias_ih_l0{dir}"] = RandomTensor(rng, 4 * hidden);
            weights[$"lstm.bias_hh_l0{dir}"] = RandomTensor(rng, 4 * hidden);
        }
        Tensor x = RandomTensor(rng, batch, t, inputDim);
        using CpuBackend backend = new();
        try
        {
            BiLstm lstm = new(inputDim, hidden);
            lstm.LoadWeights(weights, "lstm");
            using Tensor actual = lstm.Forward(backend, x, batch, t);
            Assert.Equal(new TensorShape(batch, t, 2 * hidden).ToString(), actual.Shape.ToString());

            float* ap = (float*)actual.DataPointer;
            foreach ((string dir, bool reverse) in new[] { ("", false), ("_reverse", true) })
            {
                LstmCell cell = new(inputDim, hidden);
                cell.BindWeights(weights[$"lstm.weight_ih_l0{dir}"], weights[$"lstm.weight_hh_l0{dir}"],
                    weights[$"lstm.bias_ih_l0{dir}"], weights[$"lstm.bias_hh_l0{dir}"]);
                Tensor h = RnnOps.ZeroAllocate(batch, hidden);
                Tensor c = RnnOps.ZeroAllocate(batch, hidden);
                Tensor stepIn = new(new TensorShape(batch, inputDim), DType.F32);
                float* xp = (float*)x.DataPointer;
                for (int i = 0; i < t; i++)
                {
                    int step = reverse ? t - 1 - i : i;
                    RnnOps.LoadTimestep(xp, stepIn, batch, t, inputDim, step);
                    (Tensor hNew, Tensor cNew) = cell.Step(backend, stepIn, h, c, batch);
                    h.Dispose(); c.Dispose();
                    h = hNew; c = cNew;
                    float* hp = (float*)h.DataPointer;
                    int offset = reverse ? hidden : 0;
                    for (int b = 0; b < batch; b++)
                    {
                        for (int k = 0; k < hidden; k++)
                        {
                            float expected = hp[b * hidden + k];
                            float got = ap[(b * t + step) * 2 * hidden + offset + k];
                            Assert.True(MathF.Abs(expected - got) <= 1e-5f + 1e-5f * MathF.Abs(expected),
                                $"dir '{dir}' b={b} step={step} k={k}: expected {expected}, got {got}");
                        }
                    }
                }
                h.Dispose(); c.Dispose(); stepIn.Dispose();
            }
        }
        finally
        {
            x.Dispose();
            foreach (Tensor w in weights.Values) w.Dispose();
        }
    }

    private static Tensor RandomTensor(Random rng, params long[] dims)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        Span<float> span = tensor.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        return tensor;
    }
}
