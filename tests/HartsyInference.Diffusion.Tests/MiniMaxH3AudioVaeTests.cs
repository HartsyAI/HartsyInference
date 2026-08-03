using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Music;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MiniMax-H3 audio VAE decoder tests. The forward test runs a miniature decoder over the <i>real</i>
/// checkpoint key names — weight-norm <c>weight_g</c>/<c>weight_v</c> pairs, the <c>ups.N.0</c> ModuleList level, the
/// interleaved <c>activations.2j</c>/<c>activations.2j+1</c> SnakeBeta pairs, <c>activation_post</c>, and a
/// bias-free <c>conv_post</c> — with real shapes, so a mis-mapped name or a wrong g/v pairing fails rather than
/// silently loading a stub. Numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class MiniMaxH3AudioVaeTests
{
    private static int _seed = 101;

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void Decoder_TinyConfig_ProducesFiniteStereoWaveform()
    {
        CpuBackend backend = new();
        MiniMaxH3AudioVaeConfig config = new()
        {
            LatentChannels = 4,
            DecoderInputChannels = 8,
            DecoderDim = 8,
            // Keeps the real checkpoint's odd k=9/stride=5 stage, where pad=(k-u)/2 truncates.
            UpsampleRates = [5, 2],
            UpsampleKernels = [9, 4],
            LatentsMean = [0.1f, -0.2f, 0.05f, 0.3f],
            LatentsStd = [1.5f, 2.0f, 1.2f, 1.8f],
        };

        MiniMaxH3AudioVaeDecoder decoder = new(config);
        decoder.LoadWeights(BuildWeights(config));

        int frames = 8, stereo = 2;
        Tensor latent = Rand([1, config.LatentChannels, stereo, frames]);
        Tensor waveform = decoder.Decode(backend, latent);

        Assert.Equal(1, (int)waveform.Shape[0]);
        Assert.Equal(stereo, (int)waveform.Shape[1]);
        Assert.Equal(frames * config.SamplesPerLatentFrame, (int)waveform.Shape[2]);
        float* p = (float*)waveform.DataPointer;
        for (long i = 0; i < waveform.Shape.ElementCount; i++)
        {
            Assert.True(float.IsFinite(p[i]), $"non-finite sample at {i}");
            Assert.InRange(p[i], -1f, 1f);
        }
        waveform.Dispose();
        latent.Dispose();
        decoder.Dispose();
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void LatentNormFold_MatchesExplicitDenormalization()
    {
        CpuBackend backend = new();
        float[] mean = [0.1f, -0.2f, 0.05f, 0.3f];
        float[] std = [1.5f, 2.0f, 1.2f, 1.8f];
        MiniMaxH3AudioVaeConfig folded = new()
        {
            LatentChannels = 4, DecoderInputChannels = 8, DecoderDim = 8,
            UpsampleRates = [2, 2], UpsampleKernels = [4, 4],
            LatentsMean = mean, LatentsStd = std,
        };
        MiniMaxH3AudioVaeConfig identity = folded with { LatentsMean = [0f, 0f, 0f, 0f], LatentsStd = [1f, 1f, 1f, 1f] };

        Dictionary<string, Tensor> w = BuildWeights(folded);
        MiniMaxH3AudioVaeDecoder a = new(folded);
        MiniMaxH3AudioVaeDecoder b = new(identity);
        a.LoadWeights(w);
        b.LoadWeights(w);

        int stereo = 2, frames = 6;
        Tensor latent = Rand([1, 4, stereo, frames]);
        Tensor denorm = new(new TensorShape(1, 4, stereo, frames), DType.F32);
        float* lp = (float*)latent.DataPointer;
        float* dp = (float*)denorm.DataPointer;
        for (int c = 0; c < 4; c++)
            for (int s = 0; s < stereo; s++)
                for (int t = 0; t < frames; t++)
                {
                    long i = ((long)c * stereo + s) * frames + t;
                    dp[i] = lp[i] * std[c] + mean[c];
                }

        Tensor viaFold = a.Decode(backend, latent);
        Tensor viaExplicit = b.Decode(backend, denorm);
        float* fp = (float*)viaFold.DataPointer;
        float* ep = (float*)viaExplicit.DataPointer;
        for (long i = 0; i < viaFold.Shape.ElementCount; i++)
            Assert.True(MathF.Abs(fp[i] - ep[i]) < 1e-5f, $"fold mismatch at {i}: {fp[i]} vs {ep[i]}");

        viaFold.Dispose();
        viaExplicit.Dispose();
        latent.Dispose();
        denorm.Dispose();
        a.Dispose();
        b.Dispose();
    }

    [Fact]
    public void DefaultConfig_Matches32kHzShippedGeometry()
    {
        MiniMaxH3AudioVaeDecoder decoder = new();
        Assert.Equal(32000, decoder.SampleRate);
        Assert.Equal(800, decoder.SamplesPerLatentFrame);
        decoder.Dispose();
    }

    [Fact]
    public void Constructor_LatentStatLengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new MiniMaxH3AudioVaeDecoder(new MiniMaxH3AudioVaeConfig { LatentsMean = [0f, 1f] }));
    }

    /// <summary>Emits every key the decoder reads, under the shipped checkpoint's own naming, at real shapes.</summary>
    private static Dictionary<string, Tensor> BuildWeights(MiniMaxH3AudioVaeConfig c)
    {
        Dictionary<string, Tensor> w = new()
        {
            ["dec_in_proj.weight"] = Rand([c.DecoderInputChannels, c.LatentChannels, 1]),
            ["dec_in_proj.bias"] = Rand([c.DecoderInputChannels]),
            ["decoder.conv_pre.bias"] = Rand([c.DecoderDim]),
        };
        AddWeightNorm(w, "decoder.conv_pre", [c.DecoderDim, c.DecoderInputChannels, 7]);

        int ch = c.DecoderDim;
        int resPerUp = c.ResblockKernels.Length;
        for (int i = 0; i < c.UpsampleRates.Length; i++)
        {
            int outC = ch / 2;
            AddWeightNorm(w, $"decoder.ups.{i}.0", [ch, outC, c.UpsampleKernels[i]]);
            w[$"decoder.ups.{i}.0.bias"] = Rand([outC]);
            for (int j = 0; j < resPerUp; j++)
            {
                int n = i * resPerUp + j;
                int k = c.ResblockKernels[j];
                for (int m = 0; m < c.ResblockDilations[j].Length; m++)
                {
                    AddWeightNorm(w, $"decoder.resblocks.{n}.convs1.{m}", [outC, outC, k]);
                    AddWeightNorm(w, $"decoder.resblocks.{n}.convs2.{m}", [outC, outC, k]);
                    w[$"decoder.resblocks.{n}.convs1.{m}.bias"] = Rand([outC]);
                    w[$"decoder.resblocks.{n}.convs2.{m}.bias"] = Rand([outC]);
                    w[$"decoder.resblocks.{n}.activations.{2 * m}.act.alpha"] = Rand([outC]);
                    w[$"decoder.resblocks.{n}.activations.{2 * m}.act.beta"] = Rand([outC]);
                    w[$"decoder.resblocks.{n}.activations.{2 * m + 1}.act.alpha"] = Rand([outC]);
                    w[$"decoder.resblocks.{n}.activations.{2 * m + 1}.act.beta"] = Rand([outC]);
                }
            }
            ch = outC;
        }
        w["decoder.activation_post.act.alpha"] = Rand([ch]);
        w["decoder.activation_post.act.beta"] = Rand([ch]);
        AddWeightNorm(w, "decoder.conv_post", [1, ch, 7]);
        return w;
    }

    private static void AddWeightNorm(Dictionary<string, Tensor> w, string prefix, long[] dims)
    {
        w[$"{prefix}.weight_v"] = Rand(dims);
        w[$"{prefix}.weight_g"] = Rand([dims[0], 1, 1]);
    }

    private static Tensor Rand(long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }
}
