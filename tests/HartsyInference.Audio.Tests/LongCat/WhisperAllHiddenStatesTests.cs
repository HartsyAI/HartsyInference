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

    private static void Release(Tensor[] states, WhisperEncoder encoder, List<Tensor> weights)
    {
        foreach (Tensor state in states) state.Dispose();
        encoder.Dispose();
        foreach (Tensor w in weights) w.Dispose();
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
        Release(states, encoder, weights);
    }

    [Fact]
    public void ALayerOutputAppearsUnnormedInTheNextSlot_AndNormedAtTheEnd()
    {
        // HF puts layer i's output in slot i + 1 raw, except the last, which is replaced by its normed form. So
        // LayerNorm(slot 1 of a 2-layer encoder) is the last slot of a 1-layer encoder with the same first layer.
        WhisperConfig two = TinyWhisperEncoder.Config(layers: 2);
        WhisperConfig one = TinyWhisperEncoder.Config(layers: 1);
        (WhisperEncoder twoLayers, List<Tensor> twoWeights) = TinyWhisperEncoder.Build(two);
        (WhisperEncoder oneLayer, List<Tensor> oneWeights) = TinyWhisperEncoder.Build(one);
        IBackend backend = new CpuBackend();
        using Tensor mel = Mel(two, 16);

        Tensor[] twoStates = twoLayers.ForwardAllHiddenStates(backend, mel);
        Tensor[] oneStates = oneLayer.ForwardAllHiddenStates(backend, mel);

        using Tensor gamma = new(new TensorShape(two.HiddenSize), DType.F32);
        gamma.AsSpan<float>().Fill(1f);
        using Tensor beta = new(new TensorShape(two.HiddenSize), DType.F32);
        beta.AsSpan<float>().Clear();
        using Tensor normed = new(twoStates[1].Shape, DType.F32);
        backend.LayerNorm(normed, twoStates[1], gamma, beta, two.LayerNormEps);

        Assert.Equal(oneStates[0].AsReadOnlySpan<float>().ToArray(), twoStates[0].AsReadOnlySpan<float>().ToArray());
        Assert.Equal(oneStates[1].AsReadOnlySpan<float>().ToArray(), normed.AsReadOnlySpan<float>().ToArray());
        Assert.NotEqual(normed.AsReadOnlySpan<float>().ToArray(), twoStates[1].AsReadOnlySpan<float>().ToArray());
        Release(twoStates, twoLayers, twoWeights);
        Release(oneStates, oneLayer, oneWeights);
    }
}
