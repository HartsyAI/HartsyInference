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
    private readonly LongCatConfig _config = LongCatTestData.TinyAvatar;
    private readonly Dictionary<string, Tensor> _weights;
    private readonly CpuBackend _backend = new();

    public LongCatAvatarChainTests()
    {
        _weights = LongCatTestData.AvatarWeights(_config, 131);
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
        // A later run's audio is 4·(7 − 1) + 1 windows.
        Assert.Equal(25, plan.RunPixelIds(1).Length);
    }

    /// <summary>A later segment's audio is the reference's 93 windows from <c>SegmentStartFrame</c> (the demo's layout, 58eb336): the 13 context
    /// windows, then the 80 new frames' windows. Its reference slot needs no window of its own.</summary>
    [Fact]
    public void LaterSegmentsReadTheReferenceWindows()
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(24, 241);
        LongCatAudioLayout layout = new() { Blocks = 5, Channels = 16, VideoFps = 25, AudioStride = 1 };
        Assert.True(plan.Segments >= 3);
        for (int s = 1; s < plan.Segments; s++)
        {
            int start = LongCatAudioFeatures.SegmentStartFrame(s, layout);
            int[] windows = plan.RunPixelIds(s);
            Assert.Equal(93, windows.Length);
            for (int i = 0; i < windows.Length; i++)
                Assert.Equal(start + i, windows[i]);
            // The new frames are the run's latents 4…23: their first window is the 14th (index 13).
            Assert.Equal(start + 13, windows[13]);
        }
    }

    /// <summary>The reference slot takes a copy of the first frame's tokens, placed in front (the reference's <c>audio_start_ref</c>).</summary>
    [Fact]
    public void TheReferenceSlotTakesACopyOfTheFirstFrameTokens()
    {
        float[] frames = [1, 2, 3, 4, 5, 6];
        Assert.Equal([1, 2, 1, 2, 3, 4, 5, 6], LongCatAvatarChainPlan.ReferenceSlotTokens(frames, 2));
    }

    /// <summary>On the 1.5 path the context comes from the re-encoding callback: it is asked once for segment 0 and once per later segment, each
    /// time with that segment's latents (the run's frames without the reference), and its result is the next run's conditioning.</summary>
    [Fact]
    public void TheReencodedContextIsAskedForEachSegment()
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(SegmentLatents, 45);
        Random rng = new(9);
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        LongCatAvatarPipeline pipeline = new(_backend, transformer);
        LongCatPrompt positive = new(LongCatTestData.RandomValues(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);
        LongCatPrompt negative = new(LongCatTestData.RandomValues(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);
        int ctx = _config.AudioContextTokens, width = _config.AudioOutputDim, channels = _config.InChannels;
        LongCatAvatarRun first = new(channels, SegmentLatents, Height, Width, LongCatTestData.RandomValues(rng, channels * SegmentLatents * Hw), positive, negative,
            LongCatTestData.RandomValues(rng, SegmentLatents * ctx * width), LongCatTestData.RandomValues(rng, SegmentLatents * ctx * width), Steps: 2, LongCatAvatarSchedule.Kind.Linear, 3f,
            2f, 1.5f, LongCatAvatarMode.At2v);
        float[] firstLatents = pipeline.Denoise(first).Latents;
        int runLatents = plan.RunLatents;
        List<float[]> audio = [];
        List<float[]> noises = [];
        for (int s = 1; s < plan.Segments; s++)
        {
            audio.Add(LongCatTestData.RandomValues(rng, runLatents * ctx * width));
            noises.Add(LongCatTestData.RandomValues(rng, channels * runLatents * Hw));
        }
        LongCatAvatarRun template = new(channels, runLatents, Height, Width, noises[0], positive, negative, audio[0],
            LongCatTestData.RandomValues(rng, runLatents * ctx * width), Steps: 2, LongCatAvatarSchedule.Kind.Linear, 3f, 2f, 1.5f, LongCatAvatarMode.Avc, Context + 1);

        List<float[]> asked = [];
        float[] zeroContext = new float[channels * Context * Hw];
        pipeline.Chain(template, plan.ContextLatents, firstLatents, audio, noises, (segment, last) =>
        {
            asked.Add(segment);
            return zeroContext;
        });

        Assert.Equal(plan.Segments, asked.Count);
        Assert.Equal(firstLatents, asked[0]);
        foreach (float[] segment in asked)
            Assert.Equal(channels * SegmentLatents * Hw, segment.Length);
    }

    [Fact]
    public void AChainRunsEachSegmentAndDecodesToTheRequestedFrames()
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(SegmentLatents, 45);
        Random rng = new(7);
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        LongCatAvatarPipeline pipeline = new(_backend, transformer);
        LongCatPrompt positive = new(LongCatTestData.RandomValues(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);
        LongCatPrompt negative = new(LongCatTestData.RandomValues(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);
        int ctx = _config.AudioContextTokens, width = _config.AudioOutputDim, channels = _config.InChannels;

        // Segment 0: an ordinary generation of SegmentLatents latents, with its own audio.
        LongCatAvatarRun first = new(channels, SegmentLatents, Height, Width, LongCatTestData.RandomValues(rng, channels * SegmentLatents * Hw), positive, negative,
            LongCatTestData.RandomValues(rng, SegmentLatents * ctx * width), LongCatTestData.RandomValues(rng, SegmentLatents * ctx * width), Steps: 2, LongCatAvatarSchedule.Kind.Linear, 3f,
            2f, 1.5f, LongCatAvatarMode.At2v);
        float[] firstLatents = pipeline.Denoise(first).Latents;

        // Later segments: the run's latents (reference, context, new) and their audio and noise.
        int runLatents = plan.RunLatents;
        List<float[]> audio = [];
        List<float[]> noises = [];
        for (int s = 1; s < plan.Segments; s++)
        {
            audio.Add(LongCatTestData.RandomValues(rng, runLatents * ctx * width));
            noises.Add(LongCatTestData.RandomValues(rng, channels * runLatents * Hw));
        }
        LongCatAvatarRun template = new(channels, runLatents, Height, Width, noises[0], positive, negative, audio[0],
            LongCatTestData.RandomValues(rng, runLatents * ctx * width), Steps: 2, LongCatAvatarSchedule.Kind.Linear, 3f, 2f, 1.5f, LongCatAvatarMode.Avc,
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
}
