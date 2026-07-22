using HartsyInference.Audio.Models.Vits;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Onnx;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Verifies a real Piper voice ONNX loads into the VITS synthesizer: the weight-name resolver recovers the
/// anonymized weight-norm conv weights, and every VITS submodule binds without a missing key. Gated on
/// <c>PIPER_ONNX</c> pointing at a Piper <c>.onnx</c> file.</summary>
public sealed class PiperLoadTests
{
    private readonly ITestOutputHelper _out;
    public PiperLoadTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void LoadsRealPiperVoice()
    {
        string? onnx = Environment.GetEnvironmentVariable("PIPER_ONNX");
        if (string.IsNullOrEmpty(onnx) || !File.Exists(onnx)) return; // gated

        using OnnxWeightLoader loader = new();
        loader.Load(onnx);
        Dictionary<string, Tensor> weights = loader.GetResolvedTensors();
        _out.WriteLine($"resolved {weights.Count} tensors");

        int anonLeft = weights.Keys.Count(k => k.StartsWith("onnx::", StringComparison.Ordinal));
        _out.WriteLine($"anonymous tensors remaining: {anonLeft}");

        // Every flow coupling's WaveNet convs must now be named (the recovered weight-norm weights).
        for (int i = 0; i < 4; i++)
        {
            int couplingIdx = 2 * i;
            Assert.True(weights.ContainsKey($"flow.flows.{couplingIdx}.enc.in_layers.0.weight"),
                $"missing flow.flows.{couplingIdx}.enc.in_layers.0.weight");
        }

        VitsConfig cfg = VitsConfig.PiperMedium;
        using VitsSynthesizer synth = new(cfg);
        synth.LoadWeights(weights); // throws if any key is missing

        _out.WriteLine("VitsSynthesizer.LoadWeights succeeded — all VITS keys bound.");

        // Tiny smoke inference (deterministic: zero noise) over a short id sequence.
        using CpuBackend backend = new();
        int[] tokens = VitsLengthRegulator.Intersperse([10, 20, 30, 40], blank: 0, bos: 1, eos: 2);
        float[] audio = synth.Infer(backend, tokens, lengthScale: 1f, noiseScale: 0f, seed: 0);
        _out.WriteLine($"produced {audio.Length} samples");
        Assert.NotEmpty(audio);
        foreach (float a in audio) Assert.True(float.IsFinite(a));
    }
}
