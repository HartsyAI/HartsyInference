using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The driving cache stores each block's self-attention INPUT and re-projects K/V per forward instead of
/// storing both. Two things must hold and are silent when they don't: the re-projected pair has to be the pair the
/// prepass would have stored (a dropped QK-norm or a swapped projection still produces a plausible video), and the
/// cache has to actually be half the size — the whole point of the change is fitting real geometries.</summary>
public unsafe class WanAnimate2DrivingCacheTests
{
    private const int Frames = 3;
    private const int GridH = 3;
    private const int GridW = 2;

    private static WanVideoConfig Config(int layers) => new WanVideoConfig
    {
        NumHeads = 2,
        HeadDim = 16,
        InChannels = 36,
        OutChannels = 16,
        VaeLatentChannels = 16,
        FfnDim = 64,
        NumLayers = layers,
        VaeSpatialCompression = 8,
        ImageDim = 8,
        AddedKvProjDim = 8,
        IsAnimate2 = true,
    };

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        Random rng = new Random(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    private static float MaxAbsDiff(Tensor a, Tensor b)
    {
        Assert.Equal(a.ElementCount, b.ElementCount);
        float* pa = (float*)a.DataPointer, pb = (float*)b.DataPointer;
        float worst = 0f;
        for (long i = 0; i < a.ElementCount; i++) worst = MathF.Max(worst, MathF.Abs(pa[i] - pb[i]));
        return worst;
    }

    /// <summary>Re-projecting K and V from the cached block input reproduces the K and V the block computed inline
    /// during the prepass. This is the entire safety argument for not storing them: the prepass's pair is captured
    /// from the live <c>Attention</c> path, so it is an independent witness of what the cache used to hold.</summary>
    [Fact]
    public void ReprojectedKv_EqualsTheKvTheBlockComputedInline()
    {
        WanVideoConfig c = Config(layers: 1);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        using CpuBackend backend = new CpuBackend();
        WanVideoBlock block = new WanVideoBlock(c, crossAttnNorm: true);
        block.LoadWeights(weights, "blocks.0");
        WanRope rope = new WanRope(c.HeadDim, c.RopeTheta, c.RopeMaxSeqLen);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(Frames, GridH, GridW);
        int s = Frames * GridH * GridW;

        using Tensor hidden = Random(new TensorShape(s, c.InnerDim), 61);
        using Tensor context = Random(new TensorShape(7, c.InnerDim), 62);
        using Tensor temb = Random(new TensorShape(1, 6, c.InnerDim), 63);

        WanAnimate2KvCapture capture = new WanAnimate2KvCapture { CaptureProjected = true };
        using Tensor _ = block.Forward(backend, hidden, context, temb, rope, cos, sin, s, selfAttnKvCapture: capture);
        Assert.NotNull(capture.Input);
        Assert.NotNull(capture.K);
        Assert.NotNull(capture.V);

        (Tensor k, Tensor v) = block.ProjectDrivingKv(backend, capture.Input!, s, 0);
        Assert.Equal(0f, MaxAbsDiff(capture.K!, k), 7);
        Assert.Equal(0f, MaxAbsDiff(capture.V!, v), 7);
        // A witness that the comparison is not two identically-zero tensors.
        Assert.True(MaxAbsDiff(capture.K!, capture.V!) > 1e-3f, "K and V coincide — the fixture is degenerate.");

        k.Dispose(); v.Dispose();
        capture.Input!.Dispose(); capture.K!.Dispose(); capture.V!.Dispose();
        cos.Dispose(); sin.Dispose();
    }

    /// <summary>The cache is <c>blocks × refSeq × dim</c> elements, not twice that. Asserted as bytes per driving
    /// token so the figure a run has to be sized against is the thing under test.</summary>
    [Theory]
    [InlineData(false, 4)]
    [InlineData(true, 2)]
    public void StoredBytes_AreOneTensorPerBlock_NotTwo(bool bf16, int bytesPerElement)
    {
        WanVideoConfig c = Config(layers: 3);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        using CpuBackend backend = new CpuBackend();
        using WanAnimate2Transformer dit = new WanAnimate2Transformer(c);
        dit.LoadWeights(weights);
        using Tensor driving = Random(new TensorShape([1L, c.VaeLatentChannels, Frames, GridH * 2, GridW * 2]), 71);
        using Tensor encoder = Random(new TensorShape(6, c.TextDim), 72);
        using Tensor clip = Random(new TensorShape(5, c.ImageDim), 73);

        using WanAnimate2DrivingCache cache = dit.EncodeDriving(backend, driving, encoder, clip,
            (Frames + 1, GridH, GridW), bf16Cache: bf16);

        Assert.Equal(bf16 ? DType.BF16 : DType.F32, cache.StorageDType);
        long drivingTokens = (long)cache.Frames * cache.TokensPerFrame;
        Assert.Equal((long)Frames * GridH * GridW, drivingTokens);
        Assert.Equal((long)c.NumLayers * c.InnerDim * bytesPerElement, cache.StoredBytes / drivingTokens);
        Assert.Equal((long)c.NumLayers * drivingTokens * c.InnerDim * bytesPerElement, cache.StoredBytes);
    }

    /// <summary>The reference geometry the port has to fit: 480×832, 81 pixel frames → 21 driving latent frames of
    /// 30×52 patches. Storing K and V was 1.6384 MiB per driving token (53.7 GiB); the block input is half that, and
    /// BF16 halves it again. Pinned here because the numbers are what a run gets sized against.</summary>
    [Fact]
    public void DrivingCacheFootprint_AtTheReferenceGeometry()
    {
        const int blocks = 40, dim = 5120;
        const int drivingFrames = (81 - 1) / 4 + 1;
        int refSeq = drivingFrames * (480 / 16) * (832 / 16);
        Assert.Equal(32_760, refSeq);

        long storedKvPerToken = 2L * blocks * dim * 4;
        long inputF32PerToken = (long)blocks * dim * 4;
        long inputBf16PerToken = (long)blocks * dim * 2;
        Assert.Equal(1_638_400, storedKvPerToken);
        Assert.Equal(819_200, inputF32PerToken);
        Assert.Equal(409_600, inputBf16PerToken);

        Assert.Equal(53.67, refSeq * storedKvPerToken / 1e9, 2);
        Assert.Equal(26.84, refSeq * inputF32PerToken / 1e9, 2);
        Assert.Equal(13.42, refSeq * inputBf16PerToken / 1e9, 2);
    }
}
