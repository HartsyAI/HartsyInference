using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>A base checkpoint loaded into a transformer that held an avatar checkpoint: the audio projection and every block's audio step are
/// released, the audio weights leave the streaming enumeration, and a forward without audio tokens runs. Uses the avatar DiT dump
/// (HARTSY_LONGCAT_AVATAR_DIT_DIR); skipped until it is set.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAvatarReloadTests
{
    private const string EnvName = "HARTSY_LONGCAT_AVATAR_DIT_DIR";

    [LongCatAvatarReloadFact]
    public void ABaseLoadAfterAnAvatarLoadDropsTheAudioPath()
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

        Dictionary<string, Tensor> avatar = [];
        Dictionary<string, Tensor> bare = [];
        try
        {
            foreach (JsonProperty tensor in root.GetProperty("tensors").EnumerateObject())
            {
                if (!tensor.Name.StartsWith("dit.", StringComparison.Ordinal))
                    continue;
                string name = tensor.Name["dit.".Length..];
                Tensor loaded = LongCatReferenceDump.Load(dir, document, tensor.Name);
                avatar[name] = loaded;
                if (!IsAudio(name))
                    bare[name] = loaded;
            }

            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(avatar);
            int withAudio = transformer.EnumerateWeights().Count();

            transformer.LoadWeights(bare);
            int bareCount = transformer.EnumerateWeights().Count();
            Assert.True(bareCount < withAudio, "the audio weights must leave the streaming enumeration after a base load");

            using CpuBackend backend = new();
            int frames = root.GetProperty("frames").GetInt32(), height = root.GetProperty("h").GetInt32(), width = root.GetProperty("w").GetInt32();
            using Tensor latent = new(new TensorShape([1, config.InChannels, frames, height, width]), DType.F32);
            LongCatReferenceDump.Read(dir, document, "inputs.latent").CopyTo(latent.AsSpan<float>());
            using Tensor encoder = new(new TensorShape(root.GetProperty("text_len").GetInt32(), config.CaptionChannels), DType.F32);
            LongCatReferenceDump.Read(dir, document, "inputs.encoder").CopyTo(encoder.AsSpan<float>());
            using Tensor velocity = transformer.Forward(backend, latent, encoder, [500f], new LongCatConditioning());
            Assert.Equal(frames * height * width * config.OutChannels, (int)velocity.Shape.ElementCount);
            // With no audio projection loaded, passing audio tokens is refused by name, before any shape check.
            Assert.Throws<ArgumentException>(() => transformer.Forward(backend, latent, encoder, [500f], new LongCatConditioning(),
                audioTokens: velocity));
        }
        finally
        {
            foreach (Tensor tensor in avatar.Values)
                tensor.Dispose();
        }
    }

    private static bool IsAudio(string name) =>
        name.StartsWith("audio_proj.", StringComparison.Ordinal) || name.Contains("audio_adaLN_modulation", StringComparison.Ordinal)
        || name.Contains("audio_cross_attn", StringComparison.Ordinal) || name.Contains("pre_video_crs_attn_norm", StringComparison.Ordinal);

    private sealed class LongCatAvatarReloadFactAttribute : FactAttribute
    {
        public LongCatAvatarReloadFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the avatar reload check did not run.";
        }
    }
}
