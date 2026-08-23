using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The chunk-loop gating both Wan-Animate recipes route decoded frames through: continuation chunks
/// are Lab-matched to the reference image, chunk 0 is never touched (single-chunk generations must stay
/// byte-identical), and strength 0 disables the whole pass.</summary>
public sealed class WanAnimateColorCorrectionTests
{
    private const int W = 32, H = 24;

    private static byte[] Frame(byte seed)
    {
        byte[] rgb = new byte[W * H * 3];
        for (int i = 0; i < rgb.Length; i++)
        {
            rgb[i] = (byte)(seed + i * 7 % 160);
        }
        return rgb;
    }

    private static VideoColorMatch.LabStats ReferenceStats()
    {
        byte[] reference = Frame(10);
        return VideoColorMatch.ComputeStats(reference, W, H);
    }

    [Fact]
    public void ChunkZeroIsNeverTouched()
    {
        byte[][] frames = [Frame(90), Frame(120)];
        byte[][] original = [(byte[])frames[0].Clone(), (byte[])frames[1].Clone()];
        VideoRecipeUtils.CorrectContinuationChunk(frames, W, H, chunkIndex: 0, ReferenceStats(), strength: 1f);
        Assert.Equal(original[0], frames[0]);
        Assert.Equal(original[1], frames[1]);
    }

    [Fact]
    public void ContinuationChunksAreCorrectedInPlace()
    {
        VideoColorMatch.LabStats reference = ReferenceStats();
        byte[][] frames = [Frame(90), Frame(120)];
        byte[][] original = [(byte[])frames[0].Clone(), (byte[])frames[1].Clone()];
        VideoRecipeUtils.CorrectContinuationChunk(frames, W, H, chunkIndex: 1, reference, strength: 1f);
        for (int i = 0; i < frames.Length; i++)
        {
            Assert.NotEqual(original[i], frames[i]);
            VideoColorMatch.LabStats after = VideoColorMatch.ComputeStats(frames[i], W, H);
            Assert.True(Math.Abs(after.MeanL - reference.MeanL) < 1.0,
                $"Frame {i} mean L should land on the reference: off by {after.MeanL - reference.MeanL:F3}");
        }
    }

    [Fact]
    public void StrengthZeroDisablesTheCorrectionOnEveryChunk()
    {
        byte[][] frames = [Frame(90)];
        byte[] original = (byte[])frames[0].Clone();
        VideoRecipeUtils.CorrectContinuationChunk(frames, W, H, chunkIndex: 3, ReferenceStats(), strength: 0f);
        Assert.Equal(original, frames[0]);
    }
}
