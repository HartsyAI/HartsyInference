using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The avatar KV-cache path (the reference's <c>return_kv</c> pass and <c>kv_cache_dict</c> denoising, as the pipeline's use_kv_cache
/// runs them) against the pinned model on the tiny config (tests/python-reference/longcat_avatar/dump_longcat_avatar_dit.py). The cache's keys of
/// block 0 and the denoising velocity over the last two frames must match. CPU F32. Skipped until HARTSY_LONGCAT_AVATAR_DIT_DIR names the dump.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAvatarKvParityTests
{
    private const string EnvName = "HARTSY_LONGCAT_AVATAR_DIT_DIR";
    private const double Tolerance = 1e-5;

    [LongCatAvatarKvFact]
    public void TheCachePassAndTheCachedDenoisingMatchTheReference()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        JsonElement root = document.RootElement;
        LongCatConfig config = new()
        {
            InChannels = 4, OutChannels = 4, HiddenSize = 64, Depth = 2, NumHeads = 4, CaptionChannels = 32, MlpRatio = 4,
            AdalnTembedDim = 32, FrequencyEmbeddingSize = 32, PatchSize = (1, 2, 2), TextTokensZeroPad = true,
            AudioOutputDim = 16, AudioWindow = 5, AudioBlock = 3, AudioChannel = 6, AudioIntermediateDim = 16, AudioContextTokens = 4,
            AudioVaeScale = 4,
        };

        Dictionary<string, Tensor> weights = [];
        try
        {
            foreach (JsonProperty tensor in root.GetProperty("tensors").EnumerateObject())
                if (tensor.Name.StartsWith("dit.", StringComparison.Ordinal))
                    weights[tensor.Name["dit.".Length..]] = LongCatReferenceDump.Load(dir, document, tensor.Name);

            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using CpuBackend backend = new();

            int frames = root.GetProperty("frames").GetInt32(), height = root.GetProperty("h").GetInt32(), width = root.GetProperty("w").GetInt32();
            int textLen = root.GetProperty("text_len").GetInt32(), channels = config.InChannels, hw = height * width;
            float[] latentFlat = LongCatReferenceDump.Read(dir, document, "inputs.latent");
            using Tensor cond = Frames(latentFlat, channels, frames, hw, 0, 2, height, width);
            using Tensor noisy = Frames(latentFlat, channels, frames, hw, 2, 2, height, width);
            using Tensor encoder = Shaped(LongCatReferenceDump.Read(dir, document, "inputs.encoder"), textLen, config.CaptionChannels);
            using Tensor audio = Shaped(LongCatReferenceDump.Read(dir, document, "inputs.audio"), 1 + config.AudioVaeScale * (frames - 1),
                config.AudioWindow * config.AudioBlock * config.AudioChannel);

            using Tensor audioTokens = transformer.ProjectAudio(backend, audio, frames);
            int ctx = config.AudioContextTokens;
            using Tensor noisyAudio = audioTokens.SliceRows(2 * ctx, 2 * ctx);

            LongCatConditioning conditioning = LongCatConditioning.Avatar(2, 10);
            using LongCatKvCache cache = transformer.BuildKvCache(backend, cond, conditioning);

            float[] referenceK = LongCatReferenceDump.Read(dir, document, "kv.block0.k");
            double kRelRms = RelRms(cache.Blocks[0].K.AsReadOnlySpan<float>(), referenceK);
            Assert.True(kRelRms <= Tolerance, $"block 0 cached keys relRMS {kRelRms:E3} exceeds {Tolerance:E0}.");
            double block1K = RelRms(cache.Blocks[1].K.AsReadOnlySpan<float>(), LongCatReferenceDump.Read(dir, document, "kv.block1.k"));
            Assert.True(block1K <= Tolerance, $"block 1 cached keys relRMS {block1K:E3} exceeds {Tolerance:E0}.");

            using Tensor velocity = transformer.ForwardCached(backend, noisy, encoder, [500f], conditioning, cache, audioTokens: noisyAudio);
            double vRelRms = RelRms(velocity.AsReadOnlySpan<float>(), LongCatReferenceDump.Read(dir, document, "velocity.avc_cache"));
            Assert.True(vRelRms <= Tolerance, $"cached denoising velocity relRMS {vRelRms:E3} exceeds {Tolerance:E0}.");
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
                tensor.Dispose();
        }
    }

    /// <summary>Frames <paramref name="from"/>..<paramref name="from"/>+<paramref name="count"/> of a flat <c>[1, C, T, H, W]</c> latent, as a
    /// new <c>[1, C, count, H, W]</c> tensor.</summary>
    private static Tensor Frames(float[] flat, int channels, int frames, int hw, int from, int count, int height, int width)
    {
        Tensor tensor = new(new TensorShape([1, channels, count, height, width]), DType.F32);
        Span<float> destination = tensor.AsSpan<float>();
        for (int c = 0; c < channels; c++)
            for (int t = 0; t < count; t++)
                flat.AsSpan(((c * frames) + from + t) * hw, hw).CopyTo(destination.Slice(((c * count) + t) * hw, hw));
        return tensor;
    }

    private static Tensor Shaped(float[] values, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static double RelRms(ReadOnlySpan<float> actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double diff = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            diff += d * d;
            norm += (double)expected[i] * expected[i];
        }
        return norm == 0 ? Math.Sqrt(diff) : Math.Sqrt(diff / norm);
    }

    private sealed class LongCatAvatarKvFactAttribute : FactAttribute
    {
        public LongCatAvatarKvFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the LongCat avatar KV-cache comparison did not run.";
        }
    }
}
