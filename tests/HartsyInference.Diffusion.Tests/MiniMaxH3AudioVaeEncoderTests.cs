using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight gates for the MiniMax-H3 audio VAE encoder, checked by inverting the decoder that already has
/// reference parity. The stereo check is the load-bearing one: a swapped L/R convention is inaudible on the mono-ish
/// audio the model generates, but becomes an audible defect the moment a real stereo reference is encoded.</summary>
[Trait("Category", "GpuIntegration")]
public unsafe class MiniMaxH3AudioVaeEncoderTests
{
    private const int SampleRate = 32000;

    private readonly ITestOutputHelper _output;

    public MiniMaxH3AudioVaeEncoderTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
        {
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        }
        return dir;
    }

    /// <summary>Energy at one frequency, by direct correlation against that bin's sine and cosine.</summary>
    private static double BandEnergy(ReadOnlySpan<float> x, double hz)
    {
        double re = 0, im = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double a = 2 * Math.PI * hz * i / SampleRate;
            re += x[i] * Math.Cos(a);
            im += x[i] * Math.Sin(a);
        }
        return Math.Sqrt(re * re + im * im) / x.Length;
    }

    [Fact]
    public void EncodePreservesStereoChannelIdentity()
    {
        if (!File.Exists(TestPaths.MiniMaxH3.AudioVae))
        {
            _output.WriteLine($"skipped: {TestPaths.MiniMaxH3.AudioVae} not present");
            return;
        }
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(TestPaths.MiniMaxH3.AudioVae);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(loader.GetAllTensors());
        Assert.True(MiniMaxH3AudioVaeEncoder.Matches(weights), "the audio VAE carries no encoder half");

        using MiniMaxH3AudioVaeEncoder encoder = new MiniMaxH3AudioVaeEncoder();
        encoder.LoadWeights(weights);
        using MiniMaxH3AudioVaeDecoder decoder = new MiniMaxH3AudioVaeDecoder();
        decoder.LoadWeights(weights);
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        // Distinct per-channel tones: left 440 Hz, right 1320 Hz. Silence would prove nothing — encoded latent rms
        // sits near 0.58 whatever the content — so the discriminator has to be spectral, not level-based.
        const double leftHz = 440, rightHz = 1320;
        const int samples = SampleRate * 2;
        using Tensor wave = new Tensor(new TensorShape(1, 2, samples), DType.F32);
        float* wp = (float*)wave.DataPointer;
        for (int i = 0; i < samples; i++)
        {
            wp[i] = 0.5f * MathF.Sin((float)(2 * Math.PI * leftHz * i / SampleRate));
            wp[samples + i] = 0.5f * MathF.Sin((float)(2 * Math.PI * rightHz * i / SampleRate));
        }

        using Tensor latent = encoder.Encode(backend, wave);
        Assert.Equal(4, latent.Shape.Rank);
        Assert.Equal(1, (int)latent.Shape[0]);
        Assert.Equal(encoder.Config.LatentChannels, (int)latent.Shape[1]);
        Assert.Equal(2, (int)latent.Shape[2]);
        Assert.Equal(samples / encoder.SamplesPerLatentFrame, (int)latent.Shape[3]);

        float* lp = (float*)latent.DataPointer;
        for (long i = 0; i < latent.ElementCount; i++)
        {
            Assert.True(float.IsFinite(lp[i]), $"latent[{i}] is not finite");
        }

        using Tensor decoded = decoder.Decode(backend, latent);
        int outLen = (int)decoded.Shape[2];
        float* dp = (float*)decoded.DataPointer;
        // Skip the first second: the model ramps content in, so an early window reads as silent even when correct.
        int skip = Math.Min(SampleRate, outLen / 2);
        ReadOnlySpan<float> left = new ReadOnlySpan<float>(dp + skip, outLen - skip);
        ReadOnlySpan<float> right = new ReadOnlySpan<float>(dp + outLen + skip, outLen - skip);

        double l440 = BandEnergy(left, leftHz), l1320 = BandEnergy(left, rightHz);
        double r440 = BandEnergy(right, leftHz), r1320 = BandEnergy(right, rightHz);
        _output.WriteLine($"left  ch: 440Hz {l440:F5}  1320Hz {l1320:F5}");
        _output.WriteLine($"right ch: 440Hz {r440:F5}  1320Hz {r1320:F5}");

        Assert.True(l440 > l1320 * 2, $"left channel is not dominated by its own 440 Hz tone ({l440:F5} vs {l1320:F5})");
        Assert.True(r1320 > r440 * 2, $"right channel is not dominated by its own 1320 Hz tone ({r1320:F5} vs {r440:F5})");
    }

    /// <summary>A tail shorter than one latent frame must round up rather than truncate, or a reference clip silently
    /// loses its last fraction of a second.</summary>
    [Fact]
    public void EncodePadsAPartialFrame()
    {
        if (!File.Exists(TestPaths.MiniMaxH3.AudioVae))
        {
            _output.WriteLine($"skipped: {TestPaths.MiniMaxH3.AudioVae} not present");
            return;
        }
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(TestPaths.MiniMaxH3.AudioVae);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(loader.GetAllTensors());
        using MiniMaxH3AudioVaeEncoder encoder = new MiniMaxH3AudioVaeEncoder();
        encoder.LoadWeights(weights);
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        int hop = encoder.SamplesPerLatentFrame;
        int samples = hop * 40 + hop / 3;
        using Tensor wave = new Tensor(new TensorShape(1, 2, samples), DType.F32);
        float* wp = (float*)wave.DataPointer;
        for (int i = 0; i < samples * 2; i++)
        {
            wp[i] = 0.25f * MathF.Sin(i * 0.01f);
        }
        using Tensor latent = encoder.Encode(backend, wave);
        Assert.Equal(41, (int)latent.Shape[3]);
        _output.WriteLine($"{samples} samples (hop {hop}) -> {latent.Shape[3]} latent frames");
    }
}
