using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Discriminates whether LTX-2.3's ~80 dB-quiet soundtrack originates in the audio VAE or upstream in the
/// audio latent, by decoding latents drawn from the checkpoint's own <c>per_channel_statistics</c> (the training
/// distribution) and comparing the log-mel level against a real-audio reference of about −1.3 mean.
/// <code>LTX2_AUDIO_VAE=/path/LTX23_audio_vae_bf16.safetensors dotnet test --filter LtxAudioVaeLevelDiagnostic</code></summary>
public unsafe class LtxAudioVaeLevelDiagnosticTests
{
    private readonly ITestOutputHelper _output;

    public LtxAudioVaeLevelDiagnosticTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void AudioVae_DecodesTrainingDistributionLatent_ToHealthyLogMelLevel()
    {
        string? path = Environment.GetEnvironmentVariable("LTX2_AUDIO_VAE");
        if (path is null || !File.Exists(path))
        {
            return;
        }

        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        Dictionary<string, Tensor> all = new Dictionary<string, Tensor>(loader.GetAllTensors());
        LtxVideo2CheckpointConverter.ConvertedWeights conv = LtxVideo2CheckpointConverter.Convert(all);
        Assert.True(conv.AudioVae.Count > 0, "no audio-VAE keys routed");

        float[] mean = ReadStat(conv.AudioVae, "per_channel_statistics.mean-of-means", "latents_mean");
        float[] std = ReadStat(conv.AudioVae, "per_channel_statistics.std-of-means", "latents_std");
        _output.WriteLine($"stats: n={mean.Length} meanAvg={Avg(mean):F4} stdAvg={Avg(std):F4}");

        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(LtxAudioVaeLevelDiagnosticTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        LtxAudioVaeDecoder vae = new LtxAudioVaeDecoder();
        vae.LoadWeights(VaePrecisionHelper.CastVaeWeights(conv.AudioVae, DType.F32));
        using IBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

        // The observed generation's latent stats, for contrast with the training distribution.
        Report(backend, vae, mean, std, "training distribution (stats)", useStats: true, mu: 0f, sigma: 0f);
        Report(backend, vae, mean, std, "observed generation latent", useStats: false, mu: -1.483f, sigma: 2.893f);
        Report(backend, vae, mean, std, "unit normal", useStats: false, mu: 0f, sigma: 1f);

        loader.Dispose();
    }

    private void Report(IBackend backend, LtxAudioVaeDecoder vae, float[] mean, float[] std, string label,
        bool useStats, float mu, float sigma)
    {
        const int channels = 8, melLat = 16, frames = 26;
        Tensor latent = new Tensor(new TensorShape(1, channels, frames, melLat), DType.F32);
        float* p = (float*)latent.DataPointer;
        Random rng = new Random(42);
        for (int c = 0; c < channels; c++)
        {
            for (int t = 0; t < frames; t++)
            {
                for (int m = 0; m < melLat; m++)
                {
                    int packed = c * melLat + m;
                    double g = Gauss(rng);
                    double v = useStats ? mean[packed] + std[packed] * g : mu + sigma * g;
                    p[((long)c * frames + t) * melLat + m] = (float)v;
                }
            }
        }

        Tensor mel = vae.Decode(backend, latent);
        Tensor f32 = mel.DType == DType.F32 ? mel : mel.CastTo(DType.F32);
        float* q = (float*)f32.DataPointer;
        long n = f32.ElementCount;
        float mn = float.MaxValue, mx = float.MinValue;
        double sum = 0;
        for (long i = 0; i < n; i++)
        {
            float v = q[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        _output.WriteLine($"[{label}] log-mel: min={mn:F3} max={mx:F3} mean={sum / n:F3} n={n}  (healthy real audio ~ mean -1.3 / max +4.5)");
        if (!ReferenceEquals(f32, mel)) f32.Dispose();
        mel.Dispose();
        latent.Dispose();
    }

    private static double Gauss(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static float Avg(float[] a)
    {
        double s = 0;
        foreach (float v in a) s += v;
        return (float)(s / a.Length);
    }

    private static float[] ReadStat(Dictionary<string, Tensor> vae, params string[] keys)
    {
        Tensor? t = null;
        foreach (string k in keys)
        {
            if (vae.TryGetValue(k, out Tensor? found)) { t = found; break; }
        }
        Assert.NotNull(t);
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float[] result = new float[f.Shape.ElementCount];
        float* p = (float*)f.DataPointer;
        for (int i = 0; i < result.Length; i++) result[i] = p[i];
        return result;
    }
}
