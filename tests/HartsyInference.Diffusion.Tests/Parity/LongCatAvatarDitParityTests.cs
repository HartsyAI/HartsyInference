using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The avatar DiT (audio projection, audio step in every block, text, self-attention, final layer) against the pinned reference model
/// (tests/python-reference/longcat_avatar/dump_longcat_avatar_dit.py @ 58eb336) on a tiny config, CPU F32, for three calls: at2v (no conditioning
/// frames), ai2v (one conditioning latent) and avc (two conditioning latents with one reference latent at index 10). The audio projection
/// output is compared too. Skipped, and so not evidence, until HARTSY_LONGCAT_AVATAR_DIT_DIR names the dump directory.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAvatarDitParityTests
{
    private const string EnvName = "HARTSY_LONGCAT_AVATAR_DIT_DIR";
    private const double Tolerance = 1e-5;

    [LongCatAvatarDitFact]
    public void EveryCallMatchesTheReference()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        JsonElement root = document.RootElement;
        LongCatConfig config = TinyAvatarConfig();

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
            int textLen = root.GetProperty("text_len").GetInt32();
            using Tensor latent = Shaped(LongCatReferenceDump.Read(dir, document, "inputs.latent"), 1, config.InChannels, frames, height, width);
            using Tensor encoder = Shaped(LongCatReferenceDump.Read(dir, document, "inputs.encoder"), textLen, config.CaptionChannels);
            using Tensor audio = Shaped(LongCatReferenceDump.Read(dir, document, "inputs.audio"), 1 + config.AudioVaeScale * (frames - 1),
                config.AudioWindow * config.AudioBlock * config.AudioChannel);
            float[] timestep = [500f];

            using Tensor audioTokens = transformer.ProjectAudio(backend, audio, frames);
            CompareAll(dir, document, "audio_proj", audioTokens);

            List<string> failures = [];
            Compare(failures, "at2v", transformer, backend, latent, encoder, timestep, new LongCatConditioning(), audioTokens, dir, document);
            Compare(failures, "ai2v", transformer, backend, latent, encoder, timestep, new LongCatConditioning(NumCondLatents: 1), audioTokens, dir,
                document);
            Compare(failures, "avc", transformer, backend, latent, encoder, timestep, LongCatConditioning.Avatar(2, 10), audioTokens, dir, document);
            Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
                tensor.Dispose();
        }
    }

    private static void Compare(List<string> failures, string name, LongCatTransformer transformer, CpuBackend backend, Tensor latent,
        Tensor encoder, float[] timestep, LongCatConditioning conditioning, Tensor audioTokens, string dir, JsonDocument document)
    {
        using Tensor velocity = transformer.Forward(backend, latent, encoder, timestep, in conditioning, audioTokens: audioTokens);
        double relRms = RelRms(velocity.AsReadOnlySpan<float>(), LongCatReferenceDump.Read(dir, document, $"velocity.{name}"));
        if (relRms > Tolerance)
            failures.Add($"{name}: velocity relRMS {relRms:E3} exceeds {Tolerance:E0}.");
    }

    private static void CompareAll(string dir, JsonDocument document, string name, Tensor actual)
    {
        double relRms = RelRms(actual.AsReadOnlySpan<float>(), LongCatReferenceDump.Read(dir, document, name));
        Assert.True(relRms <= Tolerance, $"{name} relRMS {relRms:E3} exceeds {Tolerance:E0}.");
    }

    /// <summary>The tiny avatar config the dump uses: hidden 64, depth 2, 4 heads, caption 32, audio window 5, 3 blocks of 6 channels, output 16.</summary>
    internal static LongCatConfig TinyAvatarConfig() => new()
    {
        InChannels = 4, OutChannels = 4, HiddenSize = 64, Depth = 2, NumHeads = 4, CaptionChannels = 32, MlpRatio = 4,
        AdalnTembedDim = 32, FrequencyEmbeddingSize = 32, PatchSize = (1, 2, 2), TextTokensZeroPad = true,
        AudioOutputDim = 16, AudioWindow = 5, AudioBlock = 3, AudioChannel = 6, AudioIntermediateDim = 16, AudioContextTokens = 4,
        AudioVaeScale = 4,
    };

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

    private sealed class LongCatAvatarDitFactAttribute : FactAttribute
    {
        public LongCatAvatarDitFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the LongCat avatar DiT reference comparison did not run.";
        }
    }
}
