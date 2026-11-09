using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ThreeD.Models.CoLa3D;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary>Real-weight check of <c>cola3d_free_q20.pt</c>. Integration lane: needs Models/CoLa3D above the test bin directory.</summary>
[Trait("Category", "Integration")]
public sealed class CoLa3DCheckpointTests
{
    [Fact]
    public void FreeQ20Checkpoint_KeyMapCoversEveryTensorAndTheDecomposerRuns()
    {
        string path = Path.Combine(FindRepoRoot(), "Models", "CoLa3D", "cola3d_free_q20.pt");
        Assert.True(
            File.Exists(path),
            $"CoLa3D integration weights not found at {path}. Place cola3d_free_q20.pt in Models/CoLa3D (a junction to the main checkout works) to run this test.");

        CoLa3DConfig config = CoLa3DTestSupport.FreeQ20();
        using PytorchPickleLoader loader = new();
        loader.Load(path, recursiveFlatten: true);
        Assert.Equal(0, loader.SkippedTensorCount);
        Assert.Equal(251, loader.Descriptors.Count);

        List<string> unmapped = loader.Descriptors.Keys.Where(k => CoLa3DKeyMap.TryMap(k) is null).Order(StringComparer.Ordinal).ToList();
        Assert.True(unmapped.Count == 0, $"Checkpoint tensors with no key-map entry: {string.Join(", ", unmapped.Take(10))}");

        Dictionary<string, Tensor> named = CoLa3DKeyMap.Remap(config, loader.GetAllTensors());
        Assert.Equal(loader.Descriptors.Count, named.Count);

        (HartsyInference.Core.Sparse.SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [6], seed: 1);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, named);
            using CoLa3DOutput output = decomposer.Forward(backend, latent, utonia);

            Assert.Equal(new TensorShape(6, 768), output.PointFeatures.Feats.Shape);
            Assert.Equal(new TensorShape(6, 20), output.MaskLogits.Shape);
            Assert.Equal(new TensorShape(1, 20, 768), output.QueryFeatures.Shape);
            foreach (float value in output.MaskLogits.AsReadOnlySpan<float>())
                Assert.True(float.IsFinite(value), "Non-finite mask logit from the real checkpoint.");
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
        }
    }

    private static string FindRepoRoot()
    {
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "Models", "CoLa3D")))
        {
            dir = dir.Parent;
        }
        return dir?.FullName ?? AppContext.BaseDirectory;
    }
}
