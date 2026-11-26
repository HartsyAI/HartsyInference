using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The long-form avatar chain on the tiny config, end to end through the DiT and the decoder: segment 0 is an ordinary generation, each later
/// segment runs the chain's Avc conditioning, and the joined latent decodes to the requested frame count. The plan's frame arithmetic is checked on
/// its own.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAvatarChainTests : IDisposable
{
    private const int Height = 4, Width = 4, TextLength = 6, SegmentLatents = 7, Context = 4;
    private static readonly int Hw = Height * Width;
    private readonly LongCatConfig _config = AvatarConfig();
    private readonly Dictionary<string, Tensor> _weights;
    private readonly CpuBackend _backend = new();

    public LongCatAvatarChainTests()
    {
        _weights = AvatarWeights(_config, 131);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_weights);
        _backend.Dispose();
    }

    [Fact]
    public void ThePlanCountsSegmentsAndFramesForAClip()
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(SegmentLatents, 45);

        // 45 pixel frames need 12 latents; segment 0 gives 7 and each later segment adds 3: three segments, 13 latents, 49 pixel frames.
        Assert.Equal(3, plan.Segments);
        Assert.Equal(13, plan.TotalLatents);
        Assert.Equal(49, plan.TotalPixelFrames);
        Assert.Equal(8, plan.RunLatents);
        // The reference (latent 0), the four context latents before the first new one (3…6), then the new latents (7…9).
        Assert.Equal([0, 3, 4, 5, 6, 7, 8, 9], plan.RunLatentIds(1));
        Assert.Equal(29, plan.RunPixelIds(1).Length);
    }

    /// <summary>A later segment's new frames take the reference layout's audio windows: the 80 windows after its 13 context windows, starting at
    /// <c>SegmentStartFrame</c>. Only the context's 16-vs-13 part of the layout is still unverified (see <c>RunPixelIds</c>).</summary>
    [Fact]
    public void LaterSegmentsNewFramesTakeTheReferenceWindows()
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(24, 241);
        LongCatAudioLayout layout = new() { Blocks = 5, Channels = 16, VideoFps = 25, AudioStride = 1 };
        Assert.True(plan.Segments >= 3);
        for (int s = 1; s < plan.Segments; s++)
        {
            int[] pixels = plan.RunPixelIds(s);
            int start = LongCatAudioFeatures.SegmentStartFrame(s, layout);
            int[] newFrames = pixels[^80..];
            for (int i = 0; i < 80; i++)
                Assert.Equal(start + 13 + i, newFrames[i]);
        }
    }

    [Fact]
    public void AChainRunsEachSegmentAndDecodesToTheRequestedFrames()
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(SegmentLatents, 45);
        Random rng = new(7);
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        LongCatAvatarPipeline pipeline = new(_backend, transformer);
        LongCatPrompt positive = new(Values(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);
        LongCatPrompt negative = new(Values(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);
        int ctx = _config.AudioContextTokens, width = _config.AudioOutputDim, channels = _config.InChannels;

        // Segment 0: an ordinary generation of SegmentLatents latents, with its own audio.
        LongCatAvatarRun first = new(channels, SegmentLatents, Height, Width, Values(rng, channels * SegmentLatents * Hw), positive, negative,
            Values(rng, SegmentLatents * ctx * width), Values(rng, SegmentLatents * ctx * width), Steps: 2, LongCatAvatarSchedule.Kind.Linear, 3f,
            2f, 1.5f, LongCatAvatarMode.At2v);
        float[] firstLatents = pipeline.Denoise(first).Latents;

        // Later segments: the run's latents (reference, context, new) and their audio and noise.
        int runLatents = plan.RunLatents;
        List<float[]> audio = [];
        List<float[]> noises = [];
        for (int s = 1; s < plan.Segments; s++)
        {
            audio.Add(Values(rng, runLatents * ctx * width));
            noises.Add(Values(rng, channels * runLatents * Hw));
        }
        LongCatAvatarRun template = new(channels, runLatents, Height, Width, noises[0], positive, negative, audio[0],
            Values(rng, runLatents * ctx * width), Steps: 2, LongCatAvatarSchedule.Kind.Linear, 3f, 2f, 1.5f, LongCatAvatarMode.Avc,
            Context + 1);

        IReadOnlyList<float[]> segments = pipeline.Chain(template, plan.ContextLatents, firstLatents, audio, noises);
        float[] joined = plan.Join(segments, channels, Hw);

        // Segment 0 is the first latents as they were generated; the joined latent has the plan's total.
        Assert.Equal(channels * plan.TotalLatents * Hw, joined.Length);
        for (int c = 0; c < channels; c++)
            for (int i = 0; i < SegmentLatents * Hw; i++)
                Assert.Equal(firstLatents[c * SegmentLatents * Hw + i], joined[c * plan.TotalLatents * Hw + i]);

        // The decoder turns the latent into 1 + 4·(latents − 1) pixel frames; the clip is cut to the requested 45.
        using Tensor clip = new(new TensorShape([1, channels, plan.TotalLatents, Height, Width]), DType.F32);
        joined.AsSpan().CopyTo(clip.AsSpan<float>());
        using Tensor pixels = new StandInDecoder().Decode(_backend, clip);
        Assert.Equal(plan.TotalPixelFrames, (int)pixels.Shape[2]);
        Assert.True(plan.TotalPixelFrames >= 45);
    }

    private sealed class StandInDecoder : IWanVaeDecoder
    {
        public Tensor Decode(IBackend backend, Tensor latent) =>
            new(new TensorShape([1, 3, 1 + 4 * (latent.Shape[2] - 1), latent.Shape[3] * 8, latent.Shape[4] * 8]), DType.F32);

        public IEnumerable<Tensor> EnumerateWeights() => [];
    }

    /// <summary>The tiny avatar config: the base tiny shape plus the audio projection and the per-block audio step.</summary>
    private static LongCatConfig AvatarConfig() => new()
    {
        HiddenSize = 128, Depth = 2, NumHeads = 4, InChannels = 4, OutChannels = 4, CaptionChannels = 48, AdalnTembedDim = 64,
        FrequencyEmbeddingSize = 64, PatchSize = (1, 2, 2), TextTokensZeroPad = true, AudioOutputDim = 16, AudioWindow = 5, AudioBlock = 3,
        AudioChannel = 6, AudioIntermediateDim = 16, AudioContextTokens = 4, AudioVaeScale = 4,
    };

    /// <summary>The tiny config's base weights plus the audio projection and each block's audio step, with seeded values.</summary>
    private static Dictionary<string, Tensor> AvatarWeights(LongCatConfig config, int seed)
    {
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, seed);
        Random rng = new(seed + 1);
        int h = config.HiddenSize, ctx = config.AudioContextTokens, outDim = config.AudioOutputDim, inter = config.AudioIntermediateDim;
        int seqLen = config.AudioWindow, seqLenVf = config.AudioWindow + config.AudioVaeScale - 1, blocks = config.AudioBlock, channels = config.AudioChannel;
        void Add(string key, params int[] dims) => weights[key] = RandomTensor(rng, dims);
        Add("audio_proj.proj1.weight", inter, seqLen * blocks * channels);
        Add("audio_proj.proj1.bias", inter);
        Add("audio_proj.proj1_vf.weight", inter, seqLenVf * blocks * channels);
        Add("audio_proj.proj1_vf.bias", inter);
        Add("audio_proj.proj2.weight", inter, inter);
        Add("audio_proj.proj2.bias", inter);
        Add("audio_proj.proj3.weight", ctx * outDim, inter);
        Add("audio_proj.proj3.bias", ctx * outDim);
        Add("audio_proj.norm.weight", outDim);
        Add("audio_proj.norm.bias", outDim);
        for (int b = 0; b < config.Depth; b++)
        {
            string p = $"blocks.{b}.";
            Add(p + "audio_adaLN_modulation.1.weight", 3 * h, config.AdalnTembedDim);
            Add(p + "audio_adaLN_modulation.1.bias", 3 * h);
            Add(p + "pre_video_crs_attn_norm.weight", h);
            Add(p + "pre_video_crs_attn_norm.bias", h);
            Add(p + "audio_cross_attn.q_linear.weight", h, h);
            Add(p + "audio_cross_attn.q_linear.bias", h);
            Add(p + "audio_cross_attn.kv_linear.weight", 2 * h, outDim);
            Add(p + "audio_cross_attn.kv_linear.bias", 2 * h);
            Add(p + "audio_cross_attn.q_norm.weight", config.HeadDim);
            Add(p + "audio_cross_attn.k_norm.weight", config.HeadDim);
            Add(p + "audio_cross_attn.proj.weight", h, h);
            Add(p + "audio_cross_attn.proj.bias", h);
        }
        return weights;
    }

    private static Tensor RandomTensor(Random rng, params int[] dims)
    {
        Tensor tensor = new(new TensorShape(dims.Select(d => (long)d).ToArray()), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return tensor;
    }

    private static float[] Values(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }
}
