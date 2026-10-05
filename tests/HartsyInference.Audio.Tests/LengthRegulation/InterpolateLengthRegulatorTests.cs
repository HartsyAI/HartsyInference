using HartsyInference.Audio.Models.LengthRegulation;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests.LengthRegulation;

/// <summary>Tiny random weights through <see cref="InterpolateLengthRegulator"/>: output shape matches the
/// explicit target length (both upsampling and downsampling), and values are finite. Says nothing about
/// parity with the real IndexTTS-2 <c>s2mel.pth</c>'s own <c>length_regulator</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class InterpolateLengthRegulatorTests : IDisposable
{
    private const int Channels = 6;
    private const int InChannels = 10;
    private const int NumStages = 2;

    private readonly List<Tensor> _owned = [];
    private readonly CpuBackend _backend = new();

    private Tensor Rand(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * 0.1);
        _owned.Add(t);
        return t;
    }

    private Tensor Ones(params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = 1f;
        _owned.Add(t);
        return t;
    }

    private InterpolateLengthRegulator BuildRegulator(Random rng)
    {
        Dictionary<string, Tensor> w = [];
        w["lr.content_in_proj.weight"] = Rand(rng, Channels, InChannels);
        w["lr.content_in_proj.bias"] = Rand(rng, Channels);
        for (int i = 0; i < NumStages; i++)
        {
            int convIdx = i * 3, normIdx = i * 3 + 1;
            w[$"lr.model.{convIdx}.weight"] = Rand(rng, Channels, Channels, 3);
            w[$"lr.model.{convIdx}.bias"] = Rand(rng, Channels);
            w[$"lr.model.{normIdx}.weight"] = Ones(Channels);
            w[$"lr.model.{normIdx}.bias"] = Rand(rng, Channels);
        }
        w[$"lr.model.{NumStages * 3}.weight"] = Rand(rng, Channels, Channels, 1);
        w[$"lr.model.{NumStages * 3}.bias"] = Rand(rng, Channels);

        InterpolateLengthRegulator reg = new(Channels, InChannels, NumStages);
        reg.LoadWeights(w, "lr");
        return reg;
    }

    [Theory]
    [InlineData(8, 14)]   // upsample
    [InlineData(14, 8)]   // downsample
    [InlineData(10, 10)]  // identity length
    public void Forward_ProducesExactTargetLength_AndFiniteValues(int sourceT, int targetT)
    {
        Random rng = new(1);
        InterpolateLengthRegulator reg = BuildRegulator(rng);
        Tensor content = Rand(rng, 1, sourceT, InChannels);

        using Tensor output = reg.Forward(_backend, content, sourceT, targetT);

        Assert.Equal(new TensorShape(1, targetT, Channels), output.Shape);
        foreach (float v in output.AsSpan<float>()) Assert.True(float.IsFinite(v));
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _backend.Dispose();
    }
}

/// <summary>Loads the real IndexTTS-2.5 <c>s2mel.pth</c>'s own <c>net.length_regulator.*</c> weights (too large
/// to bundle — point <c>INDEXTTS2_S2MEL_PTH_PATH</c> at a local copy, same file
/// <see cref="HartsyInference.Audio.Tests.IndexTts2.IndexTts2DitRealWeightTests"/> uses). A clean
/// <c>LoadWeights</c> confirms every key name this class expects matches the real checkpoint.</summary>
[Trait("Category", "Integration")]
public sealed class InterpolateLengthRegulatorRealWeightTests
{
    [Fact]
    public void Forward_SucceedsAgainstRealS2MelPth()
    {
        string? path = Environment.GetEnvironmentVariable("INDEXTTS2_S2MEL_PTH_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;   // resource-gated: skip without the real checkpoint file

        using PytorchPickleLoader loader = new();
        loader.Load(path, recursiveFlatten: true);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        try
        {
            // Real config.yaml: length_regulator.channels=512, in_channels=1024, sampling_ratios=[1,1,1,1] (4 stages).
            InterpolateLengthRegulator reg = new(channels: 512, inChannels: 1024, numStages: 4);
            reg.LoadWeights(weights, "net.length_regulator");

            const int sourceT = 15, targetT = 26;
            Random rng = new(2);
            Tensor content = new(new TensorShape(1, sourceT, 1024), DType.F32);
            foreach (ref float v in content.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);

            using CpuBackend backend = new();
            using Tensor output = reg.Forward(backend, content, sourceT, targetT);
            content.Dispose();

            Assert.Equal(new TensorShape(1, targetT, 512), output.Shape);
            foreach (float v in output.AsSpan<float>()) Assert.True(float.IsFinite(v));
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}
