using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Video.Tests;

/// <summary>Part A3 gate: SeedVR2 VAE encoder mean/logvar and decoder RGB vs the real-weight Python
/// reference (<c>dump_seedvr2_vae_reference.py</c>, basic_forward path, no slicing). relL2 &lt; 1e-3 per the
/// full-model fp32 ladder. Env-gated: <c>SEEDVR2_VAE</c> (converted safetensors) + <c>SEEDVR2_VAE_REF</c>
/// (reference dump); skips cleanly when unset. CPU backend, whole-clip forward.</summary>
public sealed class SeedVr2VaeParityTests
{
    private readonly ITestOutputHelper _output;

    public SeedVr2VaeParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EncoderAndDecoder_MatchRealWeightReference()
    {
        string? vaePath = Environment.GetEnvironmentVariable("SEEDVR2_VAE");
        string? refPath = Environment.GetEnvironmentVariable("SEEDVR2_VAE_REF");
        if (vaePath is null || refPath is null || !File.Exists(vaePath) || !File.Exists(refPath))
        {
            _output.WriteLine("SKIPPED: set SEEDVR2_VAE and SEEDVR2_VAE_REF.");
            return;
        }

        using SafeTensorsLoader weightsLoader = new();
        weightsLoader.Load(vaePath);
        Dictionary<string, Tensor> weights = weightsLoader.GetAllTensors();
        using SafeTensorsLoader refLoader = new();
        refLoader.Load(refPath);

        IBackend backend = new CpuBackend();
        SeedVr2VaeConfig config = SeedVr2VaeConfig.Default;

        SeedVr2VaeEncoder encoder = new(config);
        encoder.LoadWeights(weights);
        (Tensor mean, Tensor logvar) = encoder.Encode(backend, refLoader.GetTensor("enc.input"));
        backend.Sync();
        double meanRel = RelL2(mean, refLoader.GetTensor("enc.mean"));
        double logvarRel = RelL2(logvar, refLoader.GetTensor("enc.logvar"));
        _output.WriteLine($"encoder: mean relL2 {meanRel:e2}, logvar relL2 {logvarRel:e2}");

        SeedVr2VaeDecoder decoder = new(config);
        decoder.LoadWeights(weights);
        Tensor decoded = decoder.Decode(backend, refLoader.GetTensor("dec.input"));
        backend.Sync();
        double decRel = RelL2(decoded, refLoader.GetTensor("dec.output"));
        _output.WriteLine($"decoder: output relL2 {decRel:e2}");

        Assert.True(meanRel < 1e-3, $"encoder mean relL2 {meanRel:e2} exceeds 1e-3");
        Assert.True(logvarRel < 1e-3, $"encoder logvar relL2 {logvarRel:e2} exceeds 1e-3");
        Assert.True(decRel < 1e-3, $"decoder relL2 {decRel:e2} exceeds 1e-3");
    }

    private static double RelL2(Tensor actual, Tensor expected)
    {
        Assert.Equal(expected.Shape.ElementCount, actual.Shape.ElementCount);
        ReadOnlySpan<float> a = actual.AsSpan<float>();
        ReadOnlySpan<float> e = expected.AsSpan<float>();
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - e[i];
            num += d * d;
            den += (double)e[i] * e[i];
        }
        return Math.Sqrt(num / (den + 1e-12));
    }
}
