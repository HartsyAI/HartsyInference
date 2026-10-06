using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Decodes fixed codes through a tiny random ModifiedDAC and compares with the official fish-speech
/// <c>DAC.quantizer.decode</c> + <c>DAC.decoder</c> (<c>tools/fish_audio/modded_dac_reference.py</c>, float32 rope).</summary>
public sealed unsafe class ModifiedDacDecoderParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "FishAudioS2");

    private static ModifiedDacConfig Tiny() => new()
    {
        LatentDim = 64, CodebookDim = 8, NumResidualCodebooks = 3, SemanticCodebookSize = 32, ResidualCodebookSize = 16,
        DecoderDim = 32, DecoderRates = [2, 2, 2, 2], TransformerLayers = 2, TransformerHeads = 4, TransformerHeadDim = 16,
        TransformerIntermediate = 96, TransformerWindow = 8,
    };

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    [Fact]
    public void Decode_MatchesOfficialImplementation()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "modded_dac_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys) weights[name] = loader.GetTensor(name);

        float[] flat = Read(weights["ref.codes"]);
        int books = 4, t = flat.Length / books;
        int[,] codes = new int[books, t];
        for (int i = 0; i < books; i++) for (int j = 0; j < t; j++) codes[i, j] = (int)flat[i * t + j];

        using IBackend backend = new CpuBackend();
        using ModifiedDacDecoder decoder = new(Tiny());
        decoder.LoadWeights(weights);
        Dictionary<string, float[]> taps = new();
        float[] audio = decoder.Decode(backend, codes, t, (name, v) => taps[name] = v);
        foreach (KeyValuePair<string, float[]> tap in taps)
        {
            float[] want = Read(weights[$"tap.{tap.Key}"]);
            float d = 0f;
            for (int i = 0; i < want.Length; i++) d = MathF.Max(d, MathF.Abs(want[i] - tap.Value[i]));
            float mag = want.Max(MathF.Abs);
            Console.WriteLine($"TAP {tap.Key}: max|Δ|={d:E2} max|ref|={mag:E2}");
            Assert.True(d <= 1e-3f * MathF.Max(1f, mag), $"stage {tap.Key}: max |Δ| = {d} (ref magnitude {mag})");
        }

        float[] expected = Read(weights["ref.audio"]);
        Assert.Equal(expected.Length, audio.Length);
        Assert.Equal(t * Tiny().SamplesPerFrame, audio.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - audio[i]));
        Assert.True(worst <= 1e-4f, $"max |Δ| = {worst}");
    }
}
