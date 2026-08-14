using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Locks MiniMax Music 3's window geometry: how frames split into denoising windows, how many latents a
/// window covers, and how the decoded windows are cropped so their kept spans tile the song exactly. These constants
/// are a checkpoint contract — getting them wrong produces audio that is the right length and seams at every window
/// boundary, which no shape or dtype check would catch.</summary>
public sealed unsafe class MiniMaxMusic3GeometryTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(100, 1)]
    [InlineData(200, 1)]
    [InlineData(201, 2)]
    [InlineData(250, 2)]
    [InlineData(300, 2)]
    [InlineData(301, 3)]
    [InlineData(1500, 14)]
    public void ChunkStarts_MatchesTheReferenceRange(int frames, int expectedWindows)
    {
        int[] starts = MiniMaxMusic3FlowPipeline.ChunkStarts(frames);
        Assert.Equal(expectedWindows, starts.Length);
        Assert.Equal(0, starts[0]);
        for (int i = 1; i < starts.Length; i++)
        {
            Assert.Equal(MiniMaxMusic3FlowPipeline.ChunkHop, starts[i] - starts[i - 1]);
        }
        // The reference is list(range(0, frames - hop, hop)) once past one window.
        Assert.True(starts[^1] < Math.Max(frames - MiniMaxMusic3FlowPipeline.ChunkHop, 1));
    }

    [Theory]
    [InlineData(1, 3)]
    [InlineData(40, 137)]
    [InlineData(200, 689)]
    public void LatentLength_TruncatesLikeTheReference(int frames, int expected) =>
        Assert.Equal(expected, MiniMaxMusic3ConditionEncoder.LatentLength(frames));

    /// <summary>Two 200-frame windows over 300 frames decode to 529408 samples in the diffusers reference. Every
    /// crop constant has to be right for that to come out.</summary>
    [Fact]
    public void Stitch_ReproducesTheReferenceLength()
    {
        const int latents = 689;
        const int hop = 512;
        Tensor[] windows = [Window(latents * hop, 0.5f), Window(latents * hop, 0.25f)];
        try
        {
            (float[] left, float[] right) = MiniMaxMusic3FlowPipeline.Stitch(windows, hop);
            Assert.Equal(529408, left.Length);
            Assert.Equal(529408, right.Length);

            // The first window contributes everything up to its right crop; the second contributes the rest.
            int firstKept = (latents - MiniMaxMusic3FlowPipeline.CropRightLatents) * hop;
            Assert.Equal(0.5f, left[0]);
            Assert.Equal(0.5f, left[firstKept - 1]);
            Assert.Equal(0.25f, left[firstKept]);
            Assert.Equal(0.25f, left[^1]);
            // Right is the second half of each window's [1, 2, samples] payload, not a copy of left.
            Assert.Equal(-0.5f, right[0]);
            Assert.Equal(-0.25f, right[^1]);
        }
        finally
        {
            foreach (Tensor window in windows)
            {
                window.Dispose();
            }
        }
    }

    [Fact]
    public void Stitch_ClampsToUnitRange()
    {
        Tensor[] windows = [Window(1024, 3f)];
        try
        {
            (float[] left, float[] right) = MiniMaxMusic3FlowPipeline.Stitch(windows, 512);
            Assert.Equal(1024, left.Length);
            Assert.Equal(1f, left[0]);
            Assert.Equal(-1f, right[0]);
        }
        finally
        {
            windows[0].Dispose();
        }
    }

    /// <summary>A <c>[1, 2, samples]</c> window whose left channel is <paramref name="value"/> and right is its negation.</summary>
    private static Tensor Window(int samples, float value)
    {
        Tensor window = new Tensor(new TensorShape(1, 2, samples), DType.F32);
        Span<float> values = new Span<float>((float*)window.DataPointer, 2 * samples);
        values[..samples].Fill(value);
        values[samples..].Fill(-value);
        return window;
    }
}
