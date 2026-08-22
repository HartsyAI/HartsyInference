using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wan-Animate-2 renders cleanly at 21 output frames (latent T=7) and degenerates into hazy mush at 77
/// (latent T=21) with resolution, steps, seed and driving clip all held fixed — so the defect is a function of the
/// frame count alone. The splice path's per-frame slice/scatter/attention loop runs once per generation latent
/// frame, so these scan T on the REAL token grid (hw = 40x24 = 960, the live 384x640 case) and pin the CUDA splice
/// to the CPU one at every T. A GPU primitive that is benign at 7 frames and wrong at 21 shows up here and nowhere
/// in the fixed-T tests.</summary>
public unsafe class WanAnimate2LongSequenceParityTests
{
    // The live failing geometry: 384x640 → latent 48x80 → token grid 40x24, hw = 960.
    private const int GridH = 40;
    private const int GridW = 24;

    private static WanVideoConfig Config() => new WanVideoConfig
    {
        NumHeads = 4,
        HeadDim = 128,
        InChannels = 36,
        OutChannels = 16,
        VaeLatentChannels = 16,
        FfnDim = 512,
        NumLayers = 1,
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

    private static (double MaxAbs, double RelL2) Compare(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer, pb = (float*)b.DataPointer;
        double max = 0, num = 0, den = 0;
        for (long i = 0; i < a.ElementCount; i++)
        {
            double d = pa[i] - pb[i];
            max = Math.Max(max, Math.Abs(d));
            num += d * d;
            den += (double)pa[i] * pa[i];
        }
        return (max, Math.Sqrt(num / Math.Max(den, 1e-30)));
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir)) dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    /// <summary>One denoise forward through the splice, CPU vs CUDA, at the generation frame counts the live runs
    /// use: 7 renders correctly, 21 does not.</summary>
    [Theory]
    [InlineData(7)]
    [InlineData(11)]
    [InlineData(16)]
    [InlineData(21)]
    public void SplicedForward_MatchesCpu_AtEveryGenerationLength(int genFrames)
    {
        WanVideoConfig c = Config();
        int drivingFrames = genFrames - 1;
        (int T, int H, int W) genGrid = (genFrames, GridH, GridW);

        using Tensor latents = Random(new TensorShape([1L, c.InChannels, genFrames, GridH * 2, GridW * 2]), 11);
        using Tensor encoder = Random(new TensorShape(6, c.TextDim), 12);
        using Tensor clip = Random(new TensorShape(5, c.ImageDim), 13);
        using Tensor driving = Random(new TensorShape([1L, c.VaeLatentChannels, drivingFrames, GridH * 2, GridW * 2]), 101);

        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        Tensor cpuOut, cudaOut;
        using (CpuBackend cpu = new CpuBackend())
        using (WanAnimate2Transformer dit = new WanAnimate2Transformer(c))
        {
            dit.LoadWeights(weights);
            using WanAnimate2DrivingCache cache = dit.EncodeDriving(cpu, driving, encoder, clip, genGrid);
            cpuOut = dit.Forward(cpu, latents, encoder, 1000f, cache, clip);
        }
        using (CudaBackend gpu = new CudaBackend(0, PtxDir()))
        using (WanAnimate2Transformer dit = new WanAnimate2Transformer(c))
        {
            dit.LoadWeights(weights);
            using WanAnimate2DrivingCache cache = dit.EncodeDriving(gpu, driving, encoder, clip, genGrid);
            cudaOut = dit.Forward(gpu, latents, encoder, 1000f, cache, clip);
        }
        (double max, double rel) = Compare(cpuOut, cudaOut);
        cpuOut.Dispose();
        cudaOut.Dispose();
        Console.WriteLine($"[A2-LEN] genFrames={genFrames} maxAbs={max:E3} relL2={rel:E3}");
        Assert.True(rel < 2e-3, $"CUDA splice diverged from CPU at genFrames={genFrames}: relL2 {rel:E3}, maxAbs {max:E3}.");
    }
}
