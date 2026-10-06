using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tiny random VAE decoder + BigVGAN-v2 built from the official ControlFoley classes
/// (<c>tools/controlfoley/audio_decoder_reference.py tiny</c>); every stage must match.</summary>
public sealed unsafe class ControlFoleyAudioDecoderParityTests
{
    private static readonly string Path_ = Path.Combine(AppContext.BaseDirectory, "Fixtures", "ControlFoley", "audio_decoder_tiny.safetensors");

    private static readonly ControlFoleyVaeConfig TinyVae = new() { DataDim = 128, EmbedDim = 4, HiddenDim = 8 };

    private static readonly ControlFoleyBigVganConfig TinyVoc = new()
    {
        NumMels = 128,
        UpsampleInitialChannel = 128,
        UpsampleRates = [8, 4, 2, 2, 2, 2],
        UpsampleKernelSizes = [16, 8, 4, 4, 4, 4],
        ResblockKernelSizes = [3, 7, 11],
        ResblockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    };

    private static float[] Read(Tensor t) => t.AsSpan<float>().ToArray();

    private static void Check(string stage, float[] got, float[] want, float relTol)
    {
        Assert.Equal(want.Length, got.Length);
        float worst = 0f;
        for (int i = 0; i < want.Length; i++) worst = MathF.Max(worst, MathF.Abs(want[i] - got[i]));
        float mag = want.Max(MathF.Abs);
        Console.WriteLine($"{stage}: max|diff|={worst:E2} max|ref|={mag:E2}");
        Assert.True(worst <= relTol * MathF.Max(1f, mag), $"{stage}: max|diff|={worst} (ref magnitude {mag})");
    }

    [Fact]
    public void VaeDecoderAndVocoder_MatchOfficialStageByStage()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path_);
        Dictionary<string, Tensor> w = loader.GetAllTensors();
        using IBackend backend = new CpuBackend();
        using ControlFoleyAudioDecoder decoder = new(TinyVae, TinyVoc);
        decoder.LoadWeights(w, w, "vae.", "voc.");

        float[] latent = Read(w["ref.latent"]);
        Dictionary<string, float[]> taps = [];
        using Tensor mel = decoder.DecodeToMel(backend, latent, (name, t) => taps["vae." + name] = Read(t));
        foreach (KeyValuePair<string, float[]> tap in taps)
        {
            if (!w.TryGetValue($"tap.{tap.Key}", out Tensor? want)) continue;
            Check(tap.Key, tap.Value, Read(want), 1e-4f);
        }
        Check("vae.mel", Read(mel), Read(w["tap.vae.mel"]), 1e-4f);

        taps.Clear();
        float[] wave = decoder.Vocode(backend, mel, (name, t) => taps["voc." + name] = Read(t));
        foreach (KeyValuePair<string, float[]> tap in taps) Check(tap.Key, tap.Value, Read(w[$"tap.{tap.Key}"]), 1e-4f);
        Check("voc.wave", wave, Read(w["tap.voc.wave"]), 1e-4f);
        Assert.Equal(latent.Length / TinyVae.EmbedDim * 2 * 512, wave.Length);
    }

    [Fact]
    public void MelConverter_MatchesOfficial()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path_);
        Dictionary<string, Tensor> w = loader.GetAllTensors();
        ControlFoleyMelConverter converter = ControlFoleyMelConverter.Create44k();
        float[] want = Read(w["tap.mel.output"]);
        float[] basis = Read(w["tap.mel.basis"]);
        float[,] ours = converter.MelBasis;
        float basisWorst = 0f;
        for (int m = 0; m < ours.GetLength(0); m++)
            for (int k = 0; k < ours.GetLength(1); k++) basisWorst = MathF.Max(basisWorst, MathF.Abs(ours[m, k] - basis[m * ours.GetLength(1) + k]));
        Console.WriteLine($"mel basis: max|diff|={basisWorst:E2}");
        Assert.True(basisWorst <= 1e-6f);
        float[] got = converter.Compute(Read(w["tap.mel.input"]));
        Check("mel", got, want, 1e-4f);
    }
}
