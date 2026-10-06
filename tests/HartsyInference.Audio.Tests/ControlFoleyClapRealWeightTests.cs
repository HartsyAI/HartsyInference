using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight CLAP check; <c>HARTSY_CONTROLFOLEY_CLAP_DIR</c> must hold <c>clap_audio.safetensors</c>
/// (<c>tools/controlfoley/clap_reference.py convert</c>) and <c>clap_real_reference.safetensors</c>
/// (<c>clap_reference.py real</c>, on <c>examples/ac_v2a/acv2a_reference.wav</c>); the test does nothing when it is unset.</summary>
[Trait("Category", "Integration")]
public sealed unsafe class ControlFoleyClapRealWeightTests(ITestOutputHelper output)
{
    private static string? Dir => Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_CLAP_DIR");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    [Fact]
    public void AudioEmbedding_MatchesPythonOnRealWeights()
    {
        if (Dir is not { Length: > 0 } dir)
        {
            return;
        }

        using SafeTensorsLoader weightsLoader = new();
        weightsLoader.Load(Path.Combine(dir, "clap_audio.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in weightsLoader.Descriptors.Keys)
        {
            weights[name] = weightsLoader.GetTensor(name);
        }

        using SafeTensorsLoader refLoader = new();
        refLoader.Load(Path.Combine(dir, "clap_real_reference.safetensors"));
        float[] audio = Read(refLoader.GetTensor("in.audio0"));

        using IBackend backend = new CpuBackend();
        ControlFoleyClap clap = new(ControlFoleyClapConfig.HtsatBase);
        clap.LoadWeights(weights);

        float melDiff = ControlFoleyClapParityTests.MaxAbs(Read(refLoader.GetTensor("ref.logmel0")), clap.LogMel(audio));
        float[] embedding = clap.Embed(backend, audio);
        float[] want = Read(refLoader.GetTensor("ref.embedding0"));
        float diff = ControlFoleyClapParityTests.MaxAbs(want, embedding);
        double dot = 0.0;
        for (int i = 0; i < want.Length; i++)
        {
            dot += (double)want[i] * embedding[i];
        }

        output.WriteLine($"log-mel max |d| = {melDiff:E3} dB; embedding [{embedding.Length}] max |d| = {diff:E3}, cosine = {dot:F8}");
        Assert.True(melDiff <= 5e-3f, $"log-mel max |d| = {melDiff}");
        Assert.True(diff <= 1e-4f, $"embedding max |d| = {diff}");
    }
}
