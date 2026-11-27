using HartsyInference.Core.Tensors;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The frame-wise cache's admission check: a clip whose cache does not fit the free VRAM is refused before any frame is
/// denoised, a backend that reports no VRAM is not checked, and the slot dtype is the one the rollout writes (F32 when the attention
/// branch is F32).</summary>
public sealed class FastFramesAdmissionTests
{
    private const long Gib = 1L << 30;

    [Fact]
    public void AnF16ClipThatFitsPassesAndOneThatDoesNotIsRefusedBeforeItRuns()
    {
        // The F16 arithmetic on S's shape (L's slot dtype): 2 · 30 blocks · 21 latent frames · 1560 tokens · 1536 channels · 2 bytes,
        // about 5.6 GiB, plus one block's joined prefix (about 0.2 GiB).
        Assert.Throws<InvalidOperationException>(() =>
            FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, DType.F16,
                freeBytes: 5 * Gib, totalBytes: 12 * Gib));
        FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, DType.F16,
            freeBytes: 6 * Gib, totalBytes: 12 * Gib);
    }

    /// <summary>S runs F32 slots (its inner width is 1536, so no F16 branches): the same clip is about 11.6 GiB with the joined prefix.</summary>
    [Fact]
    public void AnF32SClipOf81FramesNeedsAboutElevenPointSixGiB()
    {
        Assert.Throws<InvalidOperationException>(() =>
            FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, DType.F32,
                freeBytes: 11 * Gib, totalBytes: 12 * Gib));
        FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, DType.F32,
            freeBytes: 12 * Gib, totalBytes: 12 * Gib);
    }

    [Fact]
    public void AnF32SlotDoublesTheCacheAndIsRefusedWhereTheF16EstimatePasses()
    {
        // The same clip with F32 slots (F16 branches off, or a backend without F16 activations): about 11.6 GiB, so 6 GiB free refuses it.
        Assert.Throws<InvalidOperationException>(() =>
            FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, DType.F32,
                freeBytes: 6 * Gib, totalBytes: 12 * Gib));
        FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, DType.F32,
            freeBytes: 12 * Gib, totalBytes: 12 * Gib);
    }

    [Fact]
    public void ABackendWithoutVramIsNotChecked()
    {
        FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, DType.F32,
            freeBytes: 0, totalBytes: 0);
    }

    [Fact]
    public void TheMessageNamesTheCacheTheDtypeAndTheMemory()
    {
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            FastFramesPipeline.CheckKvFits("FF-S", blocks: 30, latentFrames: 21, tokensPerFrame: 1560, hidden: 1536, DType.F16,
                freeBytes: 1 * Gib, totalBytes: 12 * Gib));
        Assert.Contains("FF-S needs", error.Message);
        Assert.Contains("frame KV cache", error.Message);
        Assert.Contains("1024 MiB free", error.Message);
    }
}
