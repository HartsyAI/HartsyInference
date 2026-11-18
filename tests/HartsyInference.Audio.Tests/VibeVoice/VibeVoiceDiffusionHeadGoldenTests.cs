using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.VibeVoice;

/// <summary>Pins the bits of the VibeVoice diffusion head's output on a tiny random head, so moving its layers onto shared code
/// (1.5B, 7B and Realtime all run it) cannot change an output value unnoticed. The values are pinned with a tight tolerance, so the
/// test holds on hosts whose SIMD width or FMA path rounds differently; the bit hash of the reference host is logged for comparison.</summary>
public sealed unsafe class VibeVoiceDiffusionHeadGoldenTests(ITestOutputHelper output)
{
    private const string Prefix = "diffusion_head";
    private const int Hidden = 32;
    private const int Latent = 8;
    private const int Rows = 3;

    private static Tensor Weight(Random rng, float scale, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            p[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        }
        return t;
    }

    private static Dictionary<string, Tensor> Weights(VibeVoiceDiffusionHeadConfig config)
    {
        Random rng = new(21);
        int ffn = (int)(Hidden * config.HeadFfnRatio);
        float s = 1f / MathF.Sqrt(Hidden);
        Dictionary<string, Tensor> w = new()
        {
            [$"{Prefix}.noisy_images_proj.weight"] = Weight(rng, s, Hidden, Latent),
            [$"{Prefix}.cond_proj.weight"] = Weight(rng, s, Hidden, Hidden),
            [$"{Prefix}.t_embedder.mlp.0.weight"] = Weight(rng, 1f / 16f, Hidden, 256),
            [$"{Prefix}.t_embedder.mlp.2.weight"] = Weight(rng, s, Hidden, Hidden),
            [$"{Prefix}.final_layer.adaLN_modulation.1.weight"] = Weight(rng, s, 2 * Hidden, Hidden),
            [$"{Prefix}.final_layer.linear.weight"] = Weight(rng, s, Latent, Hidden),
        };
        for (int i = 0; i < config.HeadLayers; i++)
        {
            string p = $"{Prefix}.layers.{i}";
            Tensor norm = Weight(rng, 0.2f, Hidden);
            float* n = (float*)norm.DataPointer;
            for (int k = 0; k < Hidden; k++)
            {
                n[k] += 1f;
            }
            w[$"{p}.norm.weight"] = norm;
            w[$"{p}.adaLN_modulation.1.weight"] = Weight(rng, s, 3 * Hidden, Hidden);
            w[$"{p}.ffn.gate_proj.weight"] = Weight(rng, s, ffn, Hidden);
            w[$"{p}.ffn.up_proj.weight"] = Weight(rng, s, ffn, Hidden);
            w[$"{p}.ffn.down_proj.weight"] = Weight(rng, 1f / MathF.Sqrt(ffn), Hidden, ffn);
        }
        return w;
    }

    [Fact]
    public void Forward_WithNoLayers_RunsTheFinalLayerOnly()
    {
        VibeVoiceDiffusionHeadConfig config = new() { HiddenSize = Hidden, LatentSize = Latent, HeadLayers = 0 };
        VibeVoiceDiffusionHead head = new(config, Prefix);
        head.LoadWeights(Weights(config));
        Random rng = new(5);
        using Tensor noisy = Weight(rng, 1f, 1, Rows, Latent);
        using Tensor condition = Weight(rng, 1f, 1, Rows, Hidden);
        using CpuBackend backend = new();
        using Tensor result = head.Forward(backend, noisy, [999f, 500f, 3.5f], condition);
        Assert.Equal(Rows * Latent, (int)result.ElementCount);
        float* p = (float*)result.DataPointer;
        for (long i = 0; i < result.ElementCount; i++)
        {
            Assert.True(float.IsFinite(p[i]));
        }
    }

    [Fact]
    public void Forward_BitsMatchPinnedGolden()
    {
        VibeVoiceDiffusionHeadConfig config = new() { HiddenSize = Hidden, LatentSize = Latent, HeadLayers = 2 };
        VibeVoiceDiffusionHead head = new(config, Prefix);
        head.LoadWeights(Weights(config));
        Random rng = new(5);
        using Tensor noisy = Weight(rng, 1f, 1, Rows, Latent);
        using Tensor condition = Weight(rng, 1f, 1, Rows, Hidden);
        using CpuBackend backend = new();
        using Tensor result = head.Forward(backend, noisy, [999f, 500f, 3.5f], condition);

        float* p = (float*)result.DataPointer;
        ulong hash = 14695981039346656037UL;
        double sum = 0;
        for (long i = 0; i < result.ElementCount; i++)
        {
            hash = (hash ^ (uint)BitConverter.SingleToInt32Bits(p[i])) * 1099511628211UL;
            sum += p[i];
        }
        output.WriteLine($"head: n={result.ElementCount} hash={hash} sum={sum:R} first={p[0]:R}");
        Assert.Equal(Rows * Latent, (int)result.ElementCount);
        // Captured on the unmodified head (origin/main 58f248e00), CPU F32. On that host the bits were identical before and after the
        // move onto the shared stack (hash 9371697628378098361); elsewhere only the rounding of the same values may differ.
        float[] want =
        [
            0.18370247f, 0.06727442f, 0.63279974f, -0.13976434f, 0.020019636f, 0.48729897f, -0.3741818f, 0.19355868f,
            -0.36610532f, -0.46375814f, -0.84159815f, -0.7675812f, -0.47988892f, -1.3139249f, 0.6675261f, 0.39533973f,
            -0.03509122f, -0.90873206f, -0.6806934f, -0.318073f, -0.28035873f, -0.8532614f, 0.7894477f, 0.263615f,
        ];
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(MathF.Abs(p[i] - want[i]) <= 1e-6f * MathF.Max(1f, MathF.Abs(want[i])), $"value {i}: {p[i]:R} vs {want[i]:R}");
        }
    }
}
