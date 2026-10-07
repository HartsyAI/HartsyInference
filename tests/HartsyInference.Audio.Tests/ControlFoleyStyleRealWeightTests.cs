using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight timbre-encoder check; <c>HARTSY_CONTROLFOLEY_STYLE_DIR</c> must hold <c>style_encoder.safetensors</c>
/// (<c>tools/controlfoley/style_reference.py convert</c>) and <c>style_real_reference.safetensors</c>
/// (<c>style_reference.py real</c>, on <c>examples/ac_v2a/acv2a_reference.wav</c>); the test does nothing when it is unset.</summary>
[Trait("Category", "Integration")]
public sealed unsafe class ControlFoleyStyleRealWeightTests(ITestOutputHelper output)
{
    private static string? Dir => Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_STYLE_DIR");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    [Fact]
    public void StyleTokens_MatchPythonOnRealWeights()
    {
        if (Dir is not { Length: > 0 } dir)
        {
            return;
        }

        using SafeTensorsLoader weightsLoader = new();
        weightsLoader.Load(Path.Combine(dir, "style_encoder.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in weightsLoader.Descriptors.Keys)
        {
            weights[name] = weightsLoader.GetTensor(name);
        }

        using SafeTensorsLoader refLoader = new();
        refLoader.Load(Path.Combine(dir, "style_real_reference.safetensors"));
        float[] wave = Read(refLoader.GetTensor("in.wave0"));

        using IBackend backend = new CpuBackend();
        using ControlFoleyStyleEncoder encoder = new(ControlFoleyStyleConfig.MusicGenStyle);
        encoder.LoadWeights(weights);

        float mert = ControlFoleyClapParityTests.MaxAbs(Read(refLoader.GetTensor("ref.mert0")), encoder.EncodeMert(backend, wave, out int _));
        float pre = ControlFoleyClapParityTests.MaxAbs(Read(refLoader.GetTensor("ref.pre0")), encoder.EncodePreQuantizer(backend, wave, out int _));
        float[] tokens = encoder.Encode(backend, wave);
        float tokenDiff = ControlFoleyClapParityTests.MaxAbs(Read(refLoader.GetTensor("ref.tokens0")), tokens);
        float timbreDiff = ControlFoleyClapParityTests.MaxAbs(Read(refLoader.GetTensor("ref.timbre0")), encoder.EncodeTimbre(backend, wave));
        output.WriteLine($"MERT max |d| = {mert:E3}; pre-RVQ max |d| = {pre:E3}; tokens [{tokens.Length / 1536},1536] max |d| = {tokenDiff:E3}; " +
            $"timbre max |d| = {timbreDiff:E3}");
        Assert.True(mert <= 1e-3f, $"MERT max |d| = {mert}");
        Assert.True(pre <= 1e-3f, $"pre-RVQ max |d| = {pre}");
        Assert.True(tokenDiff <= 1e-3f, $"tokens max |d| = {tokenDiff}");
        Assert.True(timbreDiff <= 1e-3f, $"timbre max |d| = {timbreDiff}");
    }
}
