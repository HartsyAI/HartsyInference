using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests.LongCat;

/// <summary>The all-hidden-states overload of the Whisper encoder, on a tiny random-weight encoder.</summary>
public sealed class WhisperAllHiddenStatesTests
{
    private static Tensor Mel(WhisperConfig cfg, int frames)
    {
        Tensor mel = new(new TensorShape(1, cfg.NumMelBins, frames), DType.F32);
        Span<float> span = mel.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = MathF.Sin(i * 0.37f) * 0.8f;
        return mel;
    }

    [Fact]
    public void ReturnsLayersPlusOneStates_AndTheLastIsTheNormedForwardOutput()
    {
        WhisperConfig cfg = TinyWhisperEncoder.Config(layers: 3);
        (WhisperEncoder encoder, List<Tensor> weights) = TinyWhisperEncoder.Build(cfg);
        IBackend backend = new CpuBackend();
        using Tensor mel = Mel(cfg, 40);

        Tensor[] states = encoder.ForwardAllHiddenStates(backend, mel);
        using Tensor final = encoder.Forward(backend, mel);

        Assert.Equal(cfg.EncoderLayers + 1, states.Length);
        Assert.All(states, state => Assert.Equal(new TensorShape(1, 20, cfg.HiddenSize), state.Shape));
        Assert.Equal(final.AsReadOnlySpan<float>().ToArray(), states[^1].AsReadOnlySpan<float>().ToArray());
        for (int i = 0; i + 1 < states.Length; i++)
            Assert.NotEqual(states[i].AsReadOnlySpan<float>().ToArray(), states[i + 1].AsReadOnlySpan<float>().ToArray());
        foreach (Tensor state in states) state.Dispose();
        encoder.Dispose();
        foreach (Tensor w in weights) w.Dispose();
    }

    [Fact]
    public void TheSecondToLastStateIsNotNormed_AsHfHiddenStatesAreNot()
    {
        // The last layer's raw output never appears; the final LayerNorm of the last layer's input-to-output step does.
        // With the norm's gamma at 1 and beta at 0, a normed row has zero mean and unit variance; a raw residual stream does not.
        WhisperConfig cfg = TinyWhisperEncoder.Config(layers: 2);
        (WhisperEncoder encoder, List<Tensor> weights) = TinyWhisperEncoder.Build(cfg);
        IBackend backend = new CpuBackend();
        using Tensor mel = Mel(cfg, 16);

        Tensor[] states = encoder.ForwardAllHiddenStates(backend, mel);

        ReadOnlySpan<float> last = states[^1].AsReadOnlySpan<float>();
        ReadOnlySpan<float> before = states[^2].AsReadOnlySpan<float>();
        (double meanLast, double varLast) = RowStats(last[..cfg.HiddenSize]);
        (double meanBefore, double varBefore) = RowStats(before[..cfg.HiddenSize]);
        Assert.InRange(meanLast, -1e-5, 1e-5);
        Assert.InRange(varLast, 0.99, 1.0001);
        Assert.False(Math.Abs(meanBefore) < 1e-5 && Math.Abs(varBefore - 1) < 1e-3, "the second-to-last state looks normed");
        foreach (Tensor state in states) state.Dispose();
        encoder.Dispose();
        foreach (Tensor w in weights) w.Dispose();
    }

    private static (double Mean, double Variance) RowStats(ReadOnlySpan<float> row)
    {
        double mean = 0;
        foreach (float v in row) mean += v;
        mean /= row.Length;
        double variance = 0;
        foreach (float v in row) variance += (v - mean) * (v - mean);
        return (mean, variance / row.Length);
    }
}
