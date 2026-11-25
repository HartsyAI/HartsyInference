using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The avatar denoising loop (<see cref="LongCatAvatarPipeline"/>) against a transcription of the pinned generate_at2v and generate_avc loops
/// (tests/python-reference/longcat_avatar/dump_longcat_avatar_pipeline.py) on the tiny avatar DiT: four steps, linear sigmas with the 1.6 shift 7,
/// text and audio guidance 4. The at2v loop uses no conditioning; the avc loop uses two cached conditioning frames. Skipped until
/// HARTSY_LONGCAT_AVATAR_DIT_DIR names a dump that has the pipeline tensors.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAvatarPipelineParityTests
{
    private const string EnvName = "HARTSY_LONGCAT_AVATAR_DIT_DIR";
    private const double Tolerance = 1e-5;

    [LongCatAvatarPipelineFact]
    public void TheAt2vAndAvcLoopsMatchTheReference()
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
        JsonElement pipeline = root.GetProperty("pipeline");
        Assert.Equal(4, pipeline.GetProperty("steps").GetInt32());
        Assert.Equal(7f, (float)pipeline.GetProperty("shift").GetDouble());
        Assert.Equal(2, pipeline.GetProperty("cond_frames").GetInt32());

        Dictionary<string, Tensor> weights = [];
        try
        {
            foreach (JsonProperty tensor in root.GetProperty("tensors").EnumerateObject())
                if (tensor.Name.StartsWith("dit.", StringComparison.Ordinal))
                    weights[tensor.Name["dit.".Length..]] = LongCatReferenceDump.Load(dir, document, tensor.Name);

            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using CpuBackend backend = new();
            LongCatAvatarPipeline pipelineRunner = new(backend, transformer);

            int frames = root.GetProperty("frames").GetInt32(), height = root.GetProperty("h").GetInt32(), width = root.GetProperty("w").GetInt32();
            int textLen = root.GetProperty("text_len").GetInt32(), channels = config.InChannels, hw = height * width, ctx = config.AudioContextTokens;
            float[] latent = LongCatReferenceDump.Read(dir, document, "inputs.latent");
            float[] encoder = LongCatReferenceDump.Read(dir, document, "inputs.encoder");
            float[] negativeEmbeds = LongCatReferenceDump.Read(dir, document, "pipeline.negative");
            float[] rawAudio = LongCatReferenceDump.Read(dir, document, "inputs.audio");
            float[] zeroRawAudio = new float[rawAudio.Length];
            LongCatPrompt positive = new(encoder, textLen, textLen), negative = new(negativeEmbeds, textLen, textLen);
            using Tensor audio = Shaped(rawAudio, 1 + config.AudioVaeScale * (frames - 1), config.AudioWindow * config.AudioBlock * config.AudioChannel);
            using Tensor audioZero = Shaped(zeroRawAudio, 1 + config.AudioVaeScale * (frames - 1), config.AudioWindow * config.AudioBlock * config.AudioChannel);
            using Tensor tokens = transformer.ProjectAudio(backend, audio, frames);
            using Tensor unconditionalTokens = transformer.ProjectAudio(backend, audioZero, frames);
            float[] audioTokens = tokens.AsReadOnlySpan<float>().ToArray();
            float[] unconditional = unconditionalTokens.AsReadOnlySpan<float>().ToArray();

            LongCatAvatarRun at2v = new(channels, frames, height, width, latent, positive, negative, audioTokens, unconditional, 4,
                LongCatAvatarSchedule.Kind.Linear, 7f, 4f, 4f, LongCatAvatarMode.At2v);
            LongCatAvatarResult got = pipelineRunner.Denoise(at2v);
            double at2vError = RelRms(got.Latents, LongCatReferenceDump.Read(dir, document, "pipeline.at2v.latents"));
            Assert.True(at2vError <= Tolerance, $"at2v loop relRMS {at2vError:E3} exceeds {Tolerance:E0}.");

            int cond = pipeline.GetProperty("cond_frames").GetInt32();
            float[] condLatents = Frames(latent, channels, frames, hw, 0, cond);
            LongCatAvatarRun avc = new(channels, frames, height, width, latent, positive, negative, audioTokens, unconditional, 4,
                LongCatAvatarSchedule.Kind.Linear, 7f, 4f, 4f, LongCatAvatarMode.Avc, NumCondLatents: cond, CondLatents: condLatents,
                RefImgIndex: 10, UseKvCache: true);
            LongCatAvatarResult cached = pipelineRunner.Denoise(avc);
            double avcError = RelRms(cached.Latents, LongCatReferenceDump.Read(dir, document, "pipeline.avc.latents"));
            Assert.True(avcError <= Tolerance, $"avc loop relRMS {avcError:E3} exceeds {Tolerance:E0}.");
            // The same continuation without the cache: full forwards with the conditioning frames' timesteps at zero.
            LongCatAvatarRun avcNoCache = avc with { UseKvCache = false };
            double noCacheError = RelRms(pipelineRunner.Denoise(avcNoCache).Latents, LongCatReferenceDump.Read(dir, document, "pipeline.avc_nocache.latents"));
            Assert.True(noCacheError <= Tolerance, $"avc without the cache relRMS {noCacheError:E3} exceeds {Tolerance:E0}.");

            // Image to video: one conditioning frame (the image), no reference, no cache.
            float[] firstFrame = Frames(latent, channels, frames, hw, 0, 1);
            LongCatAvatarRun ai2v = new(channels, frames, height, width, latent, positive, negative, audioTokens, unconditional, 4,
                LongCatAvatarSchedule.Kind.Linear, 7f, 4f, 4f, LongCatAvatarMode.Ai2v, NumCondLatents: 1, CondLatents: firstFrame);
            double ai2vError = RelRms(pipelineRunner.Denoise(ai2v).Latents, LongCatReferenceDump.Read(dir, document, "pipeline.ai2v.latents"));
            Assert.True(ai2vError <= Tolerance, $"ai2v loop relRMS {ai2vError:E3} exceeds {Tolerance:E0}.");
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
                tensor.Dispose();
        }
    }

    /// <summary>A two-segment chain on the tiny DiT: segment 0 is the first latents, segment 1 adds the noisy frames after the context, and every
    /// value is finite. The reference comparison of the chain is not in this test (the chain's reference loop is not dumped yet).</summary>
    [LongCatAvatarPipelineFact]
    public void TheChainRunsTwoSegmentsAndKeepsEveryFrameFinite()
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
            LongCatAvatarPipeline runner = new(backend, transformer);

            int channels = config.InChannels, height = 4, width = 4, hw = height * width, segmentFrames = 4, ctx = config.AudioContextTokens;
            int templateFrames = segmentFrames + 1, textLen = 6;
            float[] encoder = LongCatReferenceDump.Read(dir, document, "inputs.encoder");
            LongCatPrompt prompt = new(encoder, textLen, textLen);
            float[] noise = Enumerable.Range(0, channels * templateFrames * hw).Select(i => 0.01f * (i % 17) - 0.05f).ToArray();
            float[] firstLatents = Enumerable.Range(0, channels * segmentFrames * hw).Select(i => 0.02f * (i % 13) - 0.1f).ToArray();
            float[] audioTokens = Enumerable.Range(0, templateFrames * ctx * config.AudioOutputDim).Select(i => 0.005f * (i % 11) - 0.02f).ToArray();
            float[] unconditional = new float[audioTokens.Length];

            LongCatAvatarRun template = new(channels, templateFrames, height, width, noise, prompt, prompt, audioTokens, unconditional, 2,
                LongCatAvatarSchedule.Kind.Linear, 7f, 1f, 1f, LongCatAvatarMode.Avc, NumCondLatents: 0, CondLatents: null, RefImgIndex: 10);
            IReadOnlyList<float[]> segments = runner.Chain(template, 2, firstLatents, [audioTokens], [noise]);

            Assert.Equal(2, segments.Count);
            Assert.Equal(firstLatents, segments[0]);
            Assert.Equal(channels * (segmentFrames - 2) * hw, segments[1].Length);
            Assert.All(segments[1], static v => Assert.True(float.IsFinite(v)));
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
                tensor.Dispose();
        }
    }

    /// <summary>The first <paramref name="count"/> frames of a flat <c>[1, C, F, H, W]</c> latent, as a flat <c>[C, count, hw]</c> array.</summary>
    private static float[] Frames(float[] flat, int channels, int frames, int hw, int from, int count)
    {
        float[] result = new float[channels * count * hw];
        int k = 0;
        for (int c = 0; c < channels; c++)
            for (int f = from; f < from + count; f++)
                for (int e = 0; e < hw; e++, k++)
                    result[k] = flat[(c * frames + f) * hw + e];
        return result;
    }

    private static Tensor Shaped(float[] values, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static double RelRms(float[] actual, float[] expected)
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

    private sealed class LongCatAvatarPipelineFactAttribute : FactAttribute
    {
        public LongCatAvatarPipelineFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the LongCat avatar pipeline comparison did not run.";
        }
    }
}
