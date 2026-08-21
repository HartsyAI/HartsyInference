using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>The 36-channel packing of Wan-Animate-2's two token streams. Every failure here is silent: the channel
/// order, the mask polarity and the driving latent's duplication all produce a well-shaped tensor when wrong, and
/// the model then generates a plausible clip that ignores the conditioning. Upstream's polarity is <b>1 = known</b>
/// (ComfyUI stores the inverse and flips it in <c>concat_cond</c>, so its node is not a polarity reference).</summary>
public sealed unsafe class WanAnimate2ConditioningTests
{
    private const int LatentChannels = 16;
    private const int InChannels = 36;

    /// <summary>Fills a <c>[1, z, T, h, w]</c> latent so every element is uniquely identifiable by (c, t, y, x).</summary>
    private static Tensor RampLatent(int z, int t, int h, int w, float offset)
    {
        Tensor x = new Tensor(new TensorShape([1L, z, t, h, w]), DType.F32);
        float* p = (float*)x.DataPointer;
        long n = x.Shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = offset + i;
        return x;
    }

    [Fact]
    public void GetI2vMask_MatchesUpstreamAlgorithm_ForEveryCallSiteTheReferenceUses()
    {
        // Reference-image slot: get_i2v_mask(1, h, w, 1) is all ones.
        using Tensor refMask = WanAnimate2Conditioning.GetI2vMask(1, 2, 3, maskLen: 1);
        Assert.Equal(new TensorShape([4L, 1, 2, 3]), refMask.Shape);
        AssertAll(refMask, 1f);

        // Video slot on chunk 0: get_i2v_mask(lat_t - 1, h, w, 0) is all zeros.
        using Tensor chunk0 = WanAnimate2Conditioning.GetI2vMask(21, 2, 3, maskLen: 0);
        AssertAll(chunk0, 0f);

        // Video slot on a continuation chunk: maskLen 1 marks latent frame 0 known across all 4 channels, nothing else.
        using Tensor chunkN = WanAnimate2Conditioning.GetI2vMask(21, 2, 3, maskLen: 1);
        float* p = (float*)chunkN.DataPointer;
        long frame = 2 * 3, perChannel = 21 * frame;
        for (int c = 0; c < 4; c++)
            for (int t = 0; t < 21; t++)
            {
                float expected = t == 0 ? 1f : 0f;
                for (long i = 0; i < frame; i++)
                    Assert.Equal(expected, p[c * perChannel + t * frame + i]);
            }

        // Driving stream: get_i2v_mask(21, h, w, CLIP_LEN=81) covers every pixel frame, so it is all ones.
        using Tensor drivingMask = WanAnimate2Conditioning.GetI2vMask(21, 2, 3, maskLen: 81);
        AssertAll(drivingMask, 1f);
    }

    [Fact]
    public void GetI2vMask_LeadingRepeatShiftsEveryLaterCell_NotAStraightReshape()
    {
        // maskLen 5 covers pixel frames 0..4. Cell (c, t) reads pixel t·4 + c − 3 for t > 0, so latent frame 1 is
        // known on channels 0 and 1 only (pixels 1 and 2... through 4) — a straight view(L,4) without the 3
        // prepended copies of element 0 would mark channels 0..3 of frame 1 instead.
        using Tensor mask = WanAnimate2Conditioning.GetI2vMask(4, 1, 1, maskLen: 5);
        float* p = (float*)mask.DataPointer;
        long perChannel = 4;
        for (int c = 0; c < 4; c++)
        {
            Assert.Equal(1f, p[c * perChannel + 0]);                       // t=0 always reads pixel 0
            Assert.Equal(4 + c - 3 < 5 ? 1f : 0f, p[c * perChannel + 1]);  // t=1 reads pixels 1,2,3,4 -> all known
            Assert.Equal(8 + c - 3 < 5 ? 1f : 0f, p[c * perChannel + 2]);  // t=2 reads pixels 5,6,7,8 -> none known
            Assert.Equal(0f, p[c * perChannel + 3]);
        }
    }

    [Fact]
    public void BuildGenerationChannels_OrdersMaskThenCondLatent_WithReferenceAtFrameZero()
    {
        const int t = 5, h = 2, w = 3;
        using Tensor reference = RampLatent(LatentChannels, 1, h, w, offset: 1000f);
        using Tensor video = RampLatent(LatentChannels, t, h, w, offset: 500_000f);
        using Tensor cond = WanAnimate2Conditioning.BuildGenerationChannels(reference, video, continuationChunk: false);

        Assert.Equal(new TensorShape([1L, 20, t + 1, h, w]), cond.Shape);
        float* p = (float*)cond.DataPointer;
        long frame = (long)h * w, perChannel = (t + 1) * frame;

        // Mask channels are 0..3 and the conditioning latent is 4..19 — y is mask-first, which is what makes the
        // full 36 come out as [latent | mask | cond].
        for (int c = 0; c < 4; c++)
        {
            for (long i = 0; i < frame; i++) Assert.Equal(1f, p[c * perChannel + i]);          // reference slot known
            for (int tt = 1; tt <= t; tt++)
                for (long i = 0; i < frame; i++) Assert.Equal(0f, p[c * perChannel + tt * frame + i]);
        }
        float* refSrc = (float*)reference.DataPointer;
        float* vidSrc = (float*)video.DataPointer;
        for (int c = 0; c < LatentChannels; c++)
        {
            long dstBase = (4 + c) * perChannel;
            for (long i = 0; i < frame; i++) Assert.Equal(refSrc[c * frame + i], p[dstBase + i]);
            for (long i = 0; i < (long)t * frame; i++) Assert.Equal(vidSrc[c * t * frame + i], p[dstBase + frame + i]);
        }
    }

    [Fact]
    public void BuildGenerationChannels_ContinuationChunk_MarksLatentFrameOneKnown()
    {
        const int t = 5, h = 2, w = 2;
        using Tensor reference = RampLatent(LatentChannels, 1, h, w, offset: 1f);
        using Tensor video = RampLatent(LatentChannels, t, h, w, offset: 2f);
        using Tensor cond = WanAnimate2Conditioning.BuildGenerationChannels(reference, video, continuationChunk: true);

        float* p = (float*)cond.DataPointer;
        long frame = (long)h * w, perChannel = (t + 1) * frame;
        for (int c = 0; c < 4; c++)
        {
            for (long i = 0; i < frame; i++) Assert.Equal(1f, p[c * perChannel + i]);              // reference slot
            for (long i = 0; i < frame; i++) Assert.Equal(1f, p[c * perChannel + frame + i]);      // carried frame
            for (int tt = 2; tt <= t; tt++)
                for (long i = 0; i < frame; i++) Assert.Equal(0f, p[c * perChannel + tt * frame + i]);
        }
    }

    [Fact]
    public void BuildDrivingChannels_PlacesTheSameLatentTwice_AroundAnAllOnesMask()
    {
        const int t = 4, h = 2, w = 3;
        using Tensor driving = RampLatent(LatentChannels, t, h, w, offset: 7f);
        using Tensor packed = WanAnimate2Transformer.BuildDrivingChannels(driving, InChannels);

        Assert.Equal(new TensorShape([1L, InChannels, t, h, w]), packed.Shape);
        float* src = (float*)driving.DataPointer;
        float* p = (float*)packed.DataPointer;
        long perChannel = (long)t * h * w;
        for (int c = 0; c < LatentChannels; c++)
            for (long i = 0; i < perChannel; i++)
            {
                float expected = src[c * perChannel + i];
                Assert.Equal(expected, p[c * perChannel + i]);                            // slot 1: channels 0..15
                Assert.Equal(expected, p[(20 + c) * perChannel + i]);                     // slot 2: channels 20..35
            }
        for (int c = LatentChannels; c < LatentChannels + 4; c++)
            for (long i = 0; i < perChannel; i++)
                Assert.Equal(1f, p[c * perChannel + i]);
    }

    [Fact]
    public void DrivingStream_HasExactlyOneFewerLatentFrameThanTheGenerationStream()
    {
        // The generation stream prepends the reference slot to the video encode; the driving stream does not, so the
        // gap is exactly 1 for every legal clip length. WanAnimate2Transformer.EncodeDriving refuses anything else.
        foreach (int pixelFrames in new[] { 5, 9, 21, 81 })
        {
            int genFrames = WanAnimate2Pipeline.GenerationLatentFrames(pixelFrames, temporalCompression: 4);
            int drivingFrames = (pixelFrames - 1) / 4 + 1;
            Assert.Equal(genFrames - 1, drivingFrames);
        }
        Assert.Equal(22, WanAnimate2Pipeline.GenerationLatentFrames(81, 4));   // the reference's worked example
    }

    [Fact]
    public void TrimReferenceFrame_DropsLatentFrameZeroOnly()
    {
        const int z = 3, t = 4, h = 2, w = 2;
        using Tensor latents = RampLatent(z, t, h, w, offset: 0f);
        using Tensor trimmed = WanAnimate2Conditioning.TrimReferenceFrame(latents);

        Assert.Equal(new TensorShape([1L, z, t - 1, h, w]), trimmed.Shape);
        float* src = (float*)latents.DataPointer;
        float* dst = (float*)trimmed.DataPointer;
        long frame = (long)h * w;
        for (int c = 0; c < z; c++)
            for (int tt = 0; tt < t - 1; tt++)
                for (long i = 0; i < frame; i++)
                    Assert.Equal(src[(c * t + tt + 1) * frame + i], dst[(c * (t - 1) + tt) * frame + i]);
    }

    [Fact]
    public void ConcatChannels_PutsNoiseFirst_ThenTheTwentyChannelConditioningBlock()
    {
        const int t = 3, h = 2, w = 2;
        using Tensor noise = RampLatent(LatentChannels, t, h, w, offset: 0f);
        using Tensor cond = RampLatent(20, t, h, w, offset: 100_000f);
        using Tensor input = WanAnimate2Conditioning.ConcatChannels(noise, cond);

        Assert.Equal(new TensorShape([1L, InChannels, t, h, w]), input.Shape);
        float* p = (float*)input.DataPointer;
        long perChannel = (long)t * h * w;
        float* np = (float*)noise.DataPointer;
        float* cp = (float*)cond.DataPointer;
        for (long i = 0; i < LatentChannels * perChannel; i++) Assert.Equal(np[i], p[i]);
        for (long i = 0; i < 20 * perChannel; i++) Assert.Equal(cp[i], p[LatentChannels * perChannel + i]);
    }

    private static void AssertAll(Tensor x, float expected)
    {
        float* p = (float*)x.DataPointer;
        long n = x.Shape.ElementCount;
        for (long i = 0; i < n; i++) Assert.Equal(expected, p[i]);
    }
}
