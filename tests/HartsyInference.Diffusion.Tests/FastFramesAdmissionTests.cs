using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The frame-wise cache's admission check: a clip whose cache does not fit the free VRAM is refused before any frame is
/// denoised, and a backend that reports no VRAM is not checked.</summary>
public sealed class FastFramesAdmissionTests
{
    private const long Gib = 1L << 30;

    [Fact]
    public void ASClipThatFitsPassesAndOneThatDoesNotIsRefusedBeforeItRuns()
    {
        // S at 81 frames: 2 · 30 blocks · 21 latent frames · 1560 tokens · 1536 channels · F16 bytes, about 5.6 GiB.
        Assert.Throws<InvalidOperationException>(() =>
            FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, freeBytes: 5 * Gib, totalBytes: 12 * Gib));
        FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, freeBytes: 6 * Gib, totalBytes: 12 * Gib);
    }

    [Fact]
    public void ABackendWithoutVramIsNotChecked()
    {
        FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, freeBytes: 0, totalBytes: 0);
    }

    [Fact]
    public void TheMessageNamesTheCacheAndTheMemory()
    {
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, freeBytes: 1 * Gib, totalBytes: 12 * Gib));
        Assert.Contains("FF-S needs", error.Message);
        Assert.Contains("frame KV cache", error.Message);
        Assert.Contains("1024 MiB free", error.Message);
    }
}
