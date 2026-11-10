using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ThreeD.Models.CoLa3D;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary>Real-weight checks of the three q20 decomposers. Integration lane: needs Models/CoLa3D above the test bin directory.</summary>
[Trait("Category", "Integration")]
public sealed class CoLa3DCheckpointTests
{
    [Theory]
    [InlineData(CoLa3DConfig.QuerySourceKind.Fixed, 251)]
    [InlineData(CoLa3DConfig.QuerySourceKind.PromptMask, 273)]
    [InlineData(CoLa3DConfig.QuerySourceKind.PromptPoint, 256)]
    public void Checkpoint_KeyMapCoversEveryTensorAndTheDecomposerRuns(CoLa3DConfig.QuerySourceKind source, int expectedTensors)
    {
        string fileName = CoLa3DCheckpointConverter.DecomposerFileName(source);
        string path = Path.Combine(FindRepoRoot(), "Models", "CoLa3D", fileName);
        Assert.True(
            File.Exists(path),
            $"CoLa3D integration weights not found at {path}. Place {fileName} in Models/CoLa3D (a junction to the main checkout works) to run this test.");

        CoLa3DConfig config = CoLa3DConfig.ForQuerySource(source);
        using PytorchPickleLoader loader = new();
        loader.Load(path, recursiveFlatten: true);
        Assert.Equal(0, loader.SkippedTensorCount);
        Assert.Equal(expectedTensors, loader.Descriptors.Count);

        List<string> unmapped = loader.Descriptors.Keys.Where(k => CoLa3DKeyMap.TryMap(k) is null).Order(StringComparer.Ordinal).ToList();
        Assert.True(unmapped.Count == 0, $"Checkpoint tensors with no key-map entry: {string.Join(", ", unmapped.Take(10))}");

        Dictionary<string, Tensor> named = CoLa3DKeyMap.Remap(config, loader.GetAllTensors());
        Assert.Equal(loader.Descriptors.Count, named.Count);

        (SparseTensor latent, Tensor utonia) = CoLa3DTestSupport.RandomLatent(config, [6], seed: 1);
        try
        {
            using CpuBackend backend = new();
            using CoLa3DDecomposer decomposer = new(config, named);
            using Tensor? masks = source == CoLa3DConfig.QuerySourceKind.PromptMask ? DiskMask() : null;
            CoLa3DQueryPrompts? prompts = PromptFor(source, masks, latent);
            using CoLa3DOutput output = decomposer.Forward(backend, latent, utonia, prompts);

            int queries = source == CoLa3DConfig.QuerySourceKind.Fixed ? config.MaxParts : 1;
            Assert.Equal(new TensorShape(6, 768), output.PointFeatures.Feats.Shape);
            Assert.Equal(new TensorShape(6, queries), output.MaskLogits.Shape);
            Assert.Equal(new TensorShape(1, queries, 768), output.QueryFeatures.Shape);
            foreach (float value in output.MaskLogits.AsReadOnlySpan<float>())
                Assert.True(float.IsFinite(value), "Non-finite mask logit from the real checkpoint.");
        }
        finally
        {
            latent.Feats.Dispose();
            utonia.Dispose();
        }
    }

    // One prompt for the first voxel of the synthetic latent: a 16x16 disk mask, or that voxel as the point.
    private static CoLa3DQueryPrompts? PromptFor(CoLa3DConfig.QuerySourceKind source, Tensor? masks, SparseTensor latent) => source switch
    {
        CoLa3DConfig.QuerySourceKind.Fixed => null,
        CoLa3DConfig.QuerySourceKind.PromptMask => new CoLa3DQueryPrompts { Batch = 1, Queries = 1, Masks = masks },
        _ => new CoLa3DQueryPrompts { Batch = 1, Queries = 1, Points = [latent.Coords[1], latent.Coords[2], latent.Coords[3]] },
    };

    private static Tensor DiskMask()
    {
        Tensor masks = new(new TensorShape(1, 1, 16, 16), DType.F32);
        Span<float> values = masks.AsSpan<float>();
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
                values[y * 16 + x] = (x - 8) * (x - 8) + (y - 8) * (y - 8) <= 25 ? 1f : 0f;
        }

        return masks;
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
