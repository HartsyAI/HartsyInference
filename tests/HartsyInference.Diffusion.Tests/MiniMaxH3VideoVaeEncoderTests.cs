using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight gates for the MiniMax-H3 video VAE encoder. The encoder has no reference activations to score
/// against, so the standing check is that it inverts the decoder that was already validated end-to-end: a frame that
/// survives encode→decode with its structure intact could not have come from a wrong padding mode, stride split, or
/// group-norm axis. GPU-only — the first stage alone runs four 128-channel 3x3x3 convolutions at full resolution,
/// which is minutes of CPU per frame.</summary>
[Trait("Category", "GpuIntegration")]
public unsafe class MiniMaxH3VideoVaeEncoderTests
{
    private readonly ITestOutputHelper _output;

    public MiniMaxH3VideoVaeEncoderTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
        {
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        }
        return dir;
    }

    /// <summary>A deterministic in-distribution frame: smooth gradients plus a few hard edges, which is what
    /// distinguishes a working encoder from one that merely produces finite numbers.</summary>
    private static byte[] TestFrame(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = (y * width + x) * 3;
                rgb[i] = (byte)(255f * x / width);
                rgb[i + 1] = (byte)(255f * y / height);
                bool block = x * 4 / width % 2 == y * 4 / height % 2;
                rgb[i + 2] = (byte)(block ? 220 : 40);
            }
        }
        return rgb;
    }

    private static double Correlation(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double ma = 0, mb = 0;
        for (int i = 0; i < a.Length; i++) { ma += a[i]; mb += b[i]; }
        ma /= a.Length; mb /= b.Length;
        double num = 0, da = 0, db = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double x = a[i] - ma, y = b[i] - mb;
            num += x * y; da += x * x; db += y * y;
        }
        return da > 0 && db > 0 ? num / Math.Sqrt(da * db) : double.NaN;
    }

    private static (MiniMaxH3VideoVaeEncoder Encoder, MiniMaxH3VideoVaeDecoder Decoder, SafeTensorsLoader Loader) Load()
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(TestPaths.MiniMaxH3.VideoVae);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(loader.GetAllTensors());
        MiniMaxH3VideoVaeConfig config = MiniMaxH3VideoVaeConfig.Detect(weights);
        MiniMaxH3VideoVaeEncoder encoder = new MiniMaxH3VideoVaeEncoder(config);
        encoder.LoadWeights(weights);
        MiniMaxH3VideoVaeDecoder decoder = new MiniMaxH3VideoVaeDecoder(config);
        decoder.LoadWeights(weights);
        return (encoder, decoder, loader);
    }

    [Theory]
    [InlineData(256, 256)]   // one tile: the untiled encode path
    [InlineData(512, 288)]   // the benchmark geometry, wide enough to tile on both axes
    public void EncodeInvertsTheDecoderOnARealFrame(int width, int height)
    {
        if (!File.Exists(TestPaths.MiniMaxH3.VideoVae))
        {
            _output.WriteLine($"skipped: {TestPaths.MiniMaxH3.VideoVae} not present");
            return;
        }
        (MiniMaxH3VideoVaeEncoder encoder, MiniMaxH3VideoVaeDecoder decoder, SafeTensorsLoader loader) = Load();
        using SafeTensorsLoader _ = loader;
        using MiniMaxH3VideoVaeDecoder __ = decoder;
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        byte[] frame = TestFrame(width, height);
        using Tensor latent = encoder.EncodeRgbFrame(backend, frame, width, height);

        Assert.Equal(encoder.Config.LatentChannels, (int)latent.Shape[1]);
        Assert.Equal(1, (int)latent.Shape[2]);
        Assert.Equal(height / encoder.Config.VaeRatio, (int)latent.Shape[3]);
        Assert.Equal(width / encoder.Config.VaeRatio, (int)latent.Shape[4]);

        float* lp = (float*)latent.DataPointer;
        for (long i = 0; i < latent.ElementCount; i++)
        {
            Assert.True(float.IsFinite(lp[i]), $"latent[{i}] is not finite");
        }
        double latentRms = 0;
        for (long i = 0; i < latent.ElementCount; i++) latentRms += (double)lp[i] * lp[i];
        latentRms = Math.Sqrt(latentRms / latent.ElementCount);

        using Tensor rgb = decoder.Decode(backend, latent);
        Assert.Equal(height, (int)rgb.Shape[3]);
        Assert.Equal(width, (int)rgb.Shape[4]);

        float[] original = new float[width * height * 3];
        for (int c = 0; c < 3; c++)
        {
            for (int p = 0; p < width * height; p++)
            {
                original[c * width * height + p] = frame[p * 3 + c] / 127.5f - 1f;
            }
        }
        float[] decoded = new float[width * height * 3];
        new ReadOnlySpan<float>((float*)rgb.DataPointer, decoded.Length).CopyTo(decoded);

        double corr = Correlation(original, decoded);
        _output.WriteLine($"{width}x{height}: latent rms {latentRms:F3}, round-trip correlation {corr:F4}");
        // The VAE is 16x spatial, so fine detail is genuinely lost; structure is not. An encoder with a wrong padding
        // mode, stride split or norm axis lands near zero here, not merely lower.
        Assert.True(corr > 0.9, $"round-trip correlation {corr:F4} — the encoder is not inverting the decoder");
        // Latents live in the DiT's normalized space, so a healthy encode sits near unit scale.
        Assert.InRange(latentRms, 0.2, 5.0);
    }

    /// <summary>A clip long enough to exercise the chunked temporal path, which pads to the encoder's clip length and
    /// then drops the trailing tokens — arithmetic that silently truncates a reference video if wrong.</summary>
    [Fact]
    public void EncodeChunksAClipOntoTheTokenGrid()
    {
        if (!File.Exists(TestPaths.MiniMaxH3.VideoVae))
        {
            _output.WriteLine($"skipped: {TestPaths.MiniMaxH3.VideoVae} not present");
            return;
        }
        (MiniMaxH3VideoVaeEncoder encoder, MiniMaxH3VideoVaeDecoder decoder, SafeTensorsLoader loader) = Load();
        using SafeTensorsLoader _ = loader;
        using MiniMaxH3VideoVaeDecoder __ = decoder;
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        const int width = 128, height = 128, frames = 22;
        List<byte[]> clip = new List<byte[]>(frames);
        for (int f = 0; f < frames; f++)
        {
            clip.Add(TestFrame(width, height));
        }
        using Tensor latent = encoder.EncodeRgbClip(backend, clip, width, height);

        MiniMaxH3VideoVaeConfig c = encoder.Config;
        int padded = frames % c.ClipLength == 0 ? frames : frames + (c.ClipLength - frames % c.ClipLength);
        int chunks = padded / c.ClipLength;
        int expected = chunks * c.TokensChunkSize - c.TokenDrop;
        Assert.Equal(expected, (int)latent.Shape[2]);
        _output.WriteLine($"{frames} frames -> {latent.Shape[2]} latent tokens ({chunks} chunk(s))");

        float* lp = (float*)latent.DataPointer;
        for (long i = 0; i < latent.ElementCount; i++)
        {
            Assert.True(float.IsFinite(lp[i]), $"latent[{i}] is not finite");
        }
    }
}
