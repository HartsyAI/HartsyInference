using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Decodes a fixed latent and round-trips a clip (wav to mel to vocode) with the real ControlFoley VAE and
/// BigVGAN weights. Needs <c>HARTSY_CONTROLFOLEY_DECODER_DIR</c> to name a folder with <c>audio_decoder.safetensors</c> and
/// <c>audio_decoder_reference.safetensors</c> (<c>tools/controlfoley/audio_decoder_reference.py real</c>); otherwise it does
/// nothing. The waveforms are also written there as <c>csharp_latent.wav</c> and <c>csharp_roundtrip.wav</c>.</summary>
[Trait("Category", "RealWeights")]
public sealed class ControlFoleyAudioDecoderRealWeightTests(ITestOutputHelper output)
{
    private static float[] Read(Tensor t) => t.AsSpan<float>().ToArray();

    private float Report(string stage, float[] got, float[] want)
    {
        Assert.Equal(want.Length, got.Length);
        float worst = 0f;
        double err = 0.0, ref2 = 0.0;
        for (int i = 0; i < want.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(want[i] - got[i]));
            err += (double)(want[i] - got[i]) * (want[i] - got[i]);
            ref2 += (double)want[i] * want[i];
        }
        output.WriteLine($"{stage}: max|diff|={worst:E3} rel-rms={Math.Sqrt(err / Math.Max(ref2, 1e-30)):E3} ref-rms={Math.Sqrt(ref2 / want.Length):E3}");
        return worst;
    }

    [Fact]
    public void RealWeights_DecodeAndRoundTripMatchOfficial()
    {
        string? dir = Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_DECODER_DIR");
        if (string.IsNullOrEmpty(dir)) return;

        using SafeTensorsLoader reference = new();
        reference.Load(Path.Combine(dir, "audio_decoder_reference.safetensors"));
        Dictionary<string, Tensor> r = reference.GetAllTensors();
        using IBackend backend = new CpuBackend();
        using ControlFoleyAudioDecoder decoder = new();
        decoder.LoadFromCombinedFile(Path.Combine(dir, "audio_decoder.safetensors"));

        using Tensor mel = decoder.DecodeToMel(backend, Read(r["latent"]));
        Report("vae mel", Read(mel), Read(r["mel"]));
        float[] wave = decoder.Vocode(backend, mel);
        float worst = Report("latent -> wave", wave, Read(r["wave"]));
        WavFile.WriteMono16(Path.Combine(dir, "csharp_latent.wav"), wave, decoder.SampleRate);
        Assert.True(worst <= 5e-3f, $"latent -> wave max|diff| = {worst}");

        ControlFoleyMelConverter converter = ControlFoleyMelConverter.Create44k();
        float[] clip = Read(r["rt_input"]);
        float[] rtMel = converter.Compute(clip);
        Report("clip mel", rtMel, Read(r["rt_mel"]));
        float[] rtWave = decoder.Vocode(backend, rtMel, converter.FrameCount(clip.Length));
        float rtWorst = Report("round trip wave", rtWave, Read(r["rt_wave"]));
        WavFile.WriteMono16(Path.Combine(dir, "csharp_roundtrip.wav"), rtWave, decoder.SampleRate);
        Assert.True(rtWorst <= 5e-3f, $"round trip max|diff| = {rtWorst}");
    }
}
