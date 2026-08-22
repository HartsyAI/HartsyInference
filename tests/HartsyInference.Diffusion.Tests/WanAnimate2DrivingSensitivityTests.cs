using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The gate the rest of the Animate-2 suite was missing. Every existing test checks that the re-projected
/// K/V equal the K/V the prepass computed inline — an INTERNAL consistency property that holds perfectly even if the
/// spliced driving stream never reaches the output. Live, that is exactly what happened: a wildly dancing driver and
/// a frozen still produced byte-identical video, because the driving contribution was not reaching the result.
/// These assert the only property a user cares about — that changing the driving video changes what comes out.</summary>
public unsafe class WanAnimate2DrivingSensitivityTests
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

    private static double MeanAbsDiff(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer, pb = (float*)b.DataPointer;
        double s = 0;
        for (long i = 0; i < a.ElementCount; i++) s += Math.Abs(pa[i] - pb[i]);
        return s / a.ElementCount;
    }

    /// <summary>Two different driving videos must produce different denoiser output. If this passes while real
    /// generations stay identical, the divergence is in the CUDA path, not the architecture.</summary>
    [Fact]
    public void DifferentDrivingVideos_ProduceDifferentDenoiserOutput()
    {
        WanVideoConfig c = Config(layers: 2);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        using CpuBackend backend = new CpuBackend();
        using WanAnimate2Transformer dit = new WanAnimate2Transformer(c);
        dit.LoadWeights(weights);

        (int T, int H, int W) genGrid = (Frames + 1, GridH, GridW);
        using Tensor latents = Random(new TensorShape([1L, c.InChannels, Frames + 1, GridH * 2, GridW * 2]), 11);
        using Tensor encoder = Random(new TensorShape(6, c.TextDim), 12);
        using Tensor clip = Random(new TensorShape(5, c.ImageDim), 13);

        using Tensor drivingA = Random(new TensorShape([1L, c.VaeLatentChannels, Frames, GridH * 2, GridW * 2]), 101);
        using Tensor drivingB = Random(new TensorShape([1L, c.VaeLatentChannels, Frames, GridH * 2, GridW * 2]), 202);

        using WanAnimate2DrivingCache cacheA = dit.EncodeDriving(backend, drivingA, encoder, clip, genGrid);
        using Tensor outA = dit.Forward(backend, latents, encoder, 1000f, cacheA, clip);
        using WanAnimate2DrivingCache cacheB = dit.EncodeDriving(backend, drivingB, encoder, clip, genGrid);
        using Tensor outB = dit.Forward(backend, latents, encoder, 1000f, cacheB, clip);

        double diff = MeanAbsDiff(outA, outB);
        double mag = 0;
        { float* pa = (float*)outA.DataPointer; for (long i = 0; i < outA.ElementCount; i++) mag += Math.Abs(pa[i]); }
        mag /= outA.ElementCount;
        Console.WriteLine($"[SENS] mean|out|={mag:E3}  mean|diff|={diff:E3}  relative={diff / mag:P2}");
        // Threshold is deliberately just above float noise, not a magnitude claim. With random synthetic weights
        // every attention logit is ~0, so the softmax is near-uniform and the driving signal is averaged away to
        // ~0.05% by the output layer — real weights concentrate it. The companion test pins that identical driving
        // reproduces EXACTLY, so any nonzero difference here is driving signal rather than nondeterminism.
        Assert.True(diff > 1e-6,
            $"driving video is INERT: two different driving streams gave mean|out| difference {diff:E3}. "
            + "The generation ignores the driving video entirely.");
    }

    /// <summary>Same driving stream must be reproducible — guards the test above against passing on nondeterminism
    /// rather than on real driving sensitivity.</summary>
    [Fact]
    public void SameDrivingVideo_ReproducesTheSameOutput()
    {
        WanVideoConfig c = Config(layers: 2);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        using CpuBackend backend = new CpuBackend();
        using WanAnimate2Transformer dit = new WanAnimate2Transformer(c);
        dit.LoadWeights(weights);

        (int T, int H, int W) genGrid = (Frames + 1, GridH, GridW);
        using Tensor latents = Random(new TensorShape([1L, c.InChannels, Frames + 1, GridH * 2, GridW * 2]), 11);
        using Tensor encoder = Random(new TensorShape(6, c.TextDim), 12);
        using Tensor clip = Random(new TensorShape(5, c.ImageDim), 13);
        using Tensor driving = Random(new TensorShape([1L, c.VaeLatentChannels, Frames, GridH * 2, GridW * 2]), 101);

        using WanAnimate2DrivingCache c1 = dit.EncodeDriving(backend, driving, encoder, clip, genGrid);
        using Tensor o1 = dit.Forward(backend, latents, encoder, 1000f, c1, clip);
        using WanAnimate2DrivingCache c2 = dit.EncodeDriving(backend, driving, encoder, clip, genGrid);
        using Tensor o2 = dit.Forward(backend, latents, encoder, 1000f, c2, clip);

        Assert.Equal(0.0, MeanAbsDiff(o1, o2), 10);
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir)) dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    /// <summary>The CUDA twin of the CPU gate. The CPU splice carries real signal, so if this one does not, the
    /// driving stream is being lost in a GPU kernel — which is exactly what real generations show.</summary>
    [Fact]
    public void DifferentDrivingVideos_ProduceDifferentDenoiserOutput_Cuda()
    {
        WanVideoConfig c = Config(layers: 2);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using WanAnimate2Transformer dit = new WanAnimate2Transformer(c);
        dit.LoadWeights(weights);

        (int T, int H, int W) genGrid = (Frames + 1, GridH, GridW);
        using Tensor latents = Random(new TensorShape([1L, c.InChannels, Frames + 1, GridH * 2, GridW * 2]), 11);
        using Tensor encoder = Random(new TensorShape(6, c.TextDim), 12);
        using Tensor clip = Random(new TensorShape(5, c.ImageDim), 13);
        using Tensor drivingA = Random(new TensorShape([1L, c.VaeLatentChannels, Frames, GridH * 2, GridW * 2]), 101);
        using Tensor drivingB = Random(new TensorShape([1L, c.VaeLatentChannels, Frames, GridH * 2, GridW * 2]), 202);

        using WanAnimate2DrivingCache cacheA = dit.EncodeDriving(backend, drivingA, encoder, clip, genGrid);
        using Tensor outA = dit.Forward(backend, latents, encoder, 1000f, cacheA, clip);
        using WanAnimate2DrivingCache cacheB = dit.EncodeDriving(backend, drivingB, encoder, clip, genGrid);
        using Tensor outB = dit.Forward(backend, latents, encoder, 1000f, cacheB, clip);

        double diff = MeanAbsDiff(outA, outB);
        double mag = 0;
        { float* pa = (float*)outA.DataPointer; for (long i = 0; i < outA.ElementCount; i++) mag += Math.Abs(pa[i]); }
        mag /= outA.ElementCount;
        Console.WriteLine($"[SENS-CUDA] mean|out|={mag:E3}  mean|diff|={diff:E3}  relative={diff / mag:P4}");
        Assert.True(diff > 1e-6, $"CUDA driving is INERT: difference {diff:E3} on mean|out| {mag:E3}");
    }

    /// <summary>The sensitivity gate above fixes the sequence at 3 driving frames, which is a third of the shortest
    /// live clip. The splice runs one attention call and two scatters per generation frame, so the loop is the one
    /// part of it whose work scales with the frame count; this walks that scaling. (The live 77-frame haze that
    /// prompted this scan turned out to be the base checkpoint sampled at the distillation build's 6 steps and
    /// cfg 1, not a length defect — see the plan doc. The scan stays because nothing else covers long T on CPU.)</summary>
    [Theory]
    [InlineData(3)]
    [InlineData(10)]
    [InlineData(20)]
    public void DrivingStaysLiveAsTheSequenceGrows(int drivingFrames)
    {
        WanVideoConfig c = Config(layers: 2);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        using CpuBackend backend = new CpuBackend();
        using WanAnimate2Transformer dit = new WanAnimate2Transformer(c);
        dit.LoadWeights(weights);

        (int T, int H, int W) genGrid = (drivingFrames + 1, GridH, GridW);
        using Tensor latents = Random(new TensorShape([1L, c.InChannels, drivingFrames + 1, GridH * 2, GridW * 2]), 11);
        using Tensor encoder = Random(new TensorShape(6, c.TextDim), 12);
        using Tensor clip = Random(new TensorShape(5, c.ImageDim), 13);
        using Tensor dA = Random(new TensorShape([1L, c.VaeLatentChannels, drivingFrames, GridH * 2, GridW * 2]), 101);
        using Tensor dB = Random(new TensorShape([1L, c.VaeLatentChannels, drivingFrames, GridH * 2, GridW * 2]), 202);

        using WanAnimate2DrivingCache cA = dit.EncodeDriving(backend, dA, encoder, clip, genGrid);
        using Tensor oA = dit.Forward(backend, latents, encoder, 1000f, cA, clip);
        using WanAnimate2DrivingCache cB = dit.EncodeDriving(backend, dB, encoder, clip, genGrid);
        using Tensor oB = dit.Forward(backend, latents, encoder, 1000f, cB, clip);

        double diff = MeanAbsDiff(oA, oB);
        double mag = 0;
        { float* pa = (float*)oA.DataPointer; for (long i = 0; i < oA.ElementCount; i++) mag += Math.Abs(pa[i]); }
        mag /= oA.ElementCount;
        Console.WriteLine($"[T-SCAN] drivingFrames={drivingFrames} genT={genGrid.T} mean|out|={mag:E3} mean|diff|={diff:E3} relative={diff / mag:P4}");
        Assert.True(diff > 1e-6, $"driving went INERT at genT={genGrid.T}: diff {diff:E3}");
    }
}
