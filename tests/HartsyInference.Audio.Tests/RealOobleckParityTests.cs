using Xunit;
using HartsyInference.Audio.Models.Codecs.Oobleck;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight numeric parity for the ACE-Step 1.5 Oobleck VAE decode against the diffusers
/// AutoencoderOobleck oracle (scratchpad/acevae/ref.py). Decodes the SAME latent both sides and compares.
/// Gated on env ACE15_VAE (weights) + ACE15_VAE_LATENT/ACE15_VAE_WAV (oracle dumps); skips if unset.</summary>
public sealed unsafe class RealOobleckParityTests
{
    [Fact]
    public void Decode_MatchesDiffusersOracle()
    {
        string? vae = Environment.GetEnvironmentVariable("ACE15_VAE");
        string? latPath = Environment.GetEnvironmentVariable("ACE15_VAE_LATENT");
        string? wavPath = Environment.GetEnvironmentVariable("ACE15_VAE_WAV");
        if (vae is null || latPath is null || wavPath is null) return;   // not in a real-weights env

        CpuBackend backend = new();
        SafeTensorsLoader loader = new();
        loader.Load(vae);
        OobleckVae model = new(OobleckConfig.AceStep15);
        model.LoadWeights(loader.GetAllTensors());

        float[] latRaw = ReadF32(latPath);          // [1,64,100]
        int T = latRaw.Length / 64;
        Tensor latent = new(new TensorShape(1, 64, T), DType.F32);
        new ReadOnlySpan<float>(latRaw).CopyTo(new Span<float>((float*)latent.DataPointer, latRaw.Length));

        Tensor wav = model.Decode(backend, latent);
        int n = (int)wav.Shape.ElementCount;
        float[] cs = new float[n];
        new ReadOnlySpan<float>((float*)wav.DataPointer, n).CopyTo(cs);
        float[] tor = ReadF32(wavPath);

        int m = Math.Min(cs.Length, tor.Length);
        double csPeak = 0, torPeak = 0, dot = 0, na = 0, nb = 0, maxDiff = 0;
        for (int i = 0; i < m; i++)
        {
            csPeak = Math.Max(csPeak, Math.Abs(cs[i]));
            torPeak = Math.Max(torPeak, Math.Abs(tor[i]));
            dot += cs[i] * tor[i]; na += cs[i] * (double)cs[i]; nb += tor[i] * (double)tor[i];
            maxDiff = Math.Max(maxDiff, Math.Abs(cs[i] - tor[i]));
        }
        double corr = dot / (Math.Sqrt(na * nb) + 1e-12);
        Console.WriteLine($"[OobleckParity] n={m} corr={corr:F6} maxDiff={maxDiff:E3} csPeak={csPeak:F4} torchPeak={torPeak:F4} csShape={wav.Shape}");
        latent.Dispose(); wav.Dispose();
        Assert.True(corr > 0.999, $"C# Oobleck decode diverges from diffusers: corr={corr:F6}, csPeak={csPeak:F4} vs torchPeak={torPeak:F4}");
    }

    private static float[] ReadF32(string path)
    {
        byte[] b = File.ReadAllBytes(path);
        float[] f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, f.Length * 4);
        return f;
    }
}
