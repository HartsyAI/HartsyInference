using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Decodes 24 frames of fixed codes through the real Fish Audio S2 codec and compares with the official
/// implementation's audio. Needs <c>HARTSY_FISH_S2_CODEC_DIR</c> to name a folder holding <c>codec.safetensors</c> (the
/// released <c>codec.pth</c> converted without its rope/mask buffers) and <c>s2_codec_reference.safetensors</c>
/// (<c>tools/fish_audio/modded_dac_reference.py &lt;repo&gt; &lt;dir&gt; codec.pth</c>); otherwise it does nothing.</summary>
[Trait("Category", "RealWeights")]
public sealed unsafe class ModifiedDacRealWeightTests(ITestOutputHelper output)
{
    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static string Under(string dir, string file) => dir.Length > 0 ? Path.Combine(dir, file) : $"HARTSY_FISH_S2_CODEC_DIR is unset ({file})";

    [Fact]
    public void RealS2Codec_MatchesOfficialImplementation()
    {
        string dir = Environment.GetEnvironmentVariable("HARTSY_FISH_S2_CODEC_DIR") ?? string.Empty;
        if (!RealWeightGate.Require(output.WriteLine, Under(dir, "codec.safetensors"), Under(dir, "s2_codec_reference.safetensors")))
            return;

        using SafeTensorsLoader codec = new();
        codec.Load(Path.Combine(dir, "codec.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in codec.Descriptors.Keys) weights[name] = codec.GetTensor(name);

        using SafeTensorsLoader reference = new();
        reference.Load(Path.Combine(dir, "s2_codec_reference.safetensors"));
        float[] flat = Read(reference.GetTensor("codes"));
        float[] expected = Read(reference.GetTensor("audio"));
        int books = 10, t = flat.Length / books;
        int[,] codes = new int[books, t];
        for (int i = 0; i < books; i++) for (int j = 0; j < t; j++) codes[i, j] = (int)flat[i * t + j];

        using IBackend backend = new CpuBackend();
        using ModifiedDacDecoder decoder = new(ModifiedDacConfig.S2);
        decoder.LoadWeights(weights);
        float[] audio = decoder.Decode(backend, codes, t);

        Assert.Equal(expected.Length, audio.Length);
        float worst = 0f, rms = 0f;
        for (int i = 0; i < expected.Length; i++) { worst = MathF.Max(worst, MathF.Abs(expected[i] - audio[i])); rms += expected[i] * expected[i]; }
        output.WriteLine($"max |Δ| = {worst:E2}, reference rms = {MathF.Sqrt(rms / expected.Length):E2}");
        Assert.True(worst <= 2e-3f, $"max |Δ| = {worst}");
    }

    [Fact]
    public void RealS2Codec_EncodesLikeTheOfficialImplementation()
    {
        string dir = Environment.GetEnvironmentVariable("HARTSY_FISH_S2_CODEC_DIR") ?? string.Empty;
        if (!RealWeightGate.Require(output.WriteLine, Under(dir, "s2_codec_reference.safetensors"))) return;

        using SafeTensorsLoader reference = new();
        reference.Load(Path.Combine(dir, "s2_codec_reference.safetensors"));
        // A reference dump without enc_codes is stale, not a pass: regenerate it with tools/fish_audio/modded_dac_reference.py.
        Assert.True(reference.Descriptors.ContainsKey("enc_codes"),
            "s2_codec_reference.safetensors has no enc_codes; regenerate it with tools/fish_audio/modded_dac_reference.py.");
        using SafeTensorsLoader codec = new();
        codec.Load(Path.Combine(dir, "codec.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in codec.Descriptors.Keys) weights[name] = codec.GetTensor(name);

        float[] audio = Read(reference.GetTensor("enc_audio"));
        float[] expected = Read(reference.GetTensor("enc_codes"));
        using IBackend backend = new CpuBackend();
        using ModifiedDacEncoder encoder = new(ModifiedDacConfig.S2);
        encoder.LoadWeights(weights);
        int[,] codes = encoder.Encode(backend, audio);

        int books = codes.GetLength(0), t = codes.GetLength(1);
        Assert.Equal(expected.Length, books * t);
        int same = 0;
        for (int i = 0; i < books; i++) for (int j = 0; j < t; j++) if ((int)expected[i * t + j] == codes[i, j]) same++;
        output.WriteLine($"{same}/{books * t} codes identical to the official encoder");
        Assert.True(same >= 0.98 * books * t, $"{same}/{books * t} codes identical");
    }
}
