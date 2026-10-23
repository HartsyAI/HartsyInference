using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>When the free VRAM is below a full Qwen-Image VAE decode's peak, <see cref="QwenImageVaeDecoder.DecodeWithinBudget"/>
/// must tile instead of failing. Real weights; the free-VRAM reading is forced low through the test seam. Not run
/// on the CPU lane.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("CudaSerial")]
public sealed unsafe class QwenImageVaeBudgetGpuTests
{
    private const long ConstrainedFreeBytes = 352L << 20;
    private const double MeanAbsDiffFloor = 0.05;

    private readonly ITestOutputHelper _output;

    public QwenImageVaeBudgetGpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ConstrainedFreeVram_TilesAndMatchesFullDecode()
    {
        string vaePath = TestPaths.QwenImage.Vae;
        if (!File.Exists(vaePath) || !CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: Qwen-Image VAE checkpoint or CUDA unavailable.");
            return;
        }

        using SafeTensorsLoader loader = new();
        loader.Load(vaePath);
        Dictionary<string, Tensor> weights = new();
        foreach (KeyValuePair<string, Tensor> entry in loader.GetAllTensors())
        {
            weights[entry.Key] = entry.Value.DType == DType.F32 ? entry.Value : entry.Value.CastTo(DType.F32);
        }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        using CudaBackend cuda = new CudaBackend(0, Directory.Exists(ptxDir)
            ? ptxDir
            : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx"));
        using QwenImageVaeDecoder vae = new(VaeConfig.QwenImage);
        vae.LoadWeights(weights);

        using Tensor latent = new Tensor(new TensorShape(1, QwenImageVaeDecoder.LatentChannels, 64, 64), DType.F32);
        float* lat = (float*)latent.DataPointer;
        Random rng = new(3);
        for (long i = 0; i < latent.ElementCount; i++) lat[i] = (float)(rng.NextDouble() * 2 - 1);

        using Tensor full = vae.Decode(cuda, latent);
        vae.FreeBytesForTests = () => ConstrainedFreeBytes;
        using Tensor tiled = vae.DecodeWithinBudget(cuda, latent);

        Assert.True(vae.LastTileLatent > 0, "the constrained decode ran full-resolution instead of tiling");
        Assert.Equal(full.Shape, tiled.Shape);
        double meanAbsDiff = MeanAbsDiff(full, tiled);
        _output.WriteLine($"tile {vae.LastTileLatent}, mean |full - tiled| = {meanAbsDiff:F5}");
        Assert.True(meanAbsDiff < MeanAbsDiffFloor, $"tiled decode drifts from full decode: {meanAbsDiff}");
    }

    private static double MeanAbsDiff(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        double sum = 0;
        for (long i = 0; i < a.ElementCount; i++) sum += Math.Abs(pa[i] - pb[i]);
        return sum / a.ElementCount;
    }
}
