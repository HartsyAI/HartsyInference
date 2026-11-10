using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary>
/// Real-weight check of <c>geometry_decoder.pt</c>: its 292 tensors are the shape-VAE decoder's keys with the same shapes, so
/// the shape-VAE key map covers it. Integration lane: needs Models/CoLa3D above the test bin directory.
/// </summary>
[Trait("Category", "Integration")]
public sealed class CoLa3DGeometryDecoderKeyMapTests
{
    [Fact]
    public void GeometryDecoder_EveryDecoderTensorIsPresentWithTheShapeVaeShape()
    {
        string path = Path.Combine(FindRepoRoot(), "Models", "CoLa3D", "geometry_decoder.pt");
        Assert.True(
            File.Exists(path),
            $"CoLa3D integration weights not found at {path}. Place geometry_decoder.pt in Models/CoLa3D (a junction to the main checkout works) to run this test.");

        using PytorchPickleLoader loader = new();
        loader.Load(path, recursiveFlatten: true);
        Assert.Equal(0, loader.SkippedTensorCount);

        IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = ShapeVaeKeyMap.Decoder(ShapeVaeConfig.NextDcF16C32.Decoder);
        Assert.Equal(specs.Count, loader.Descriptors.Count);
        foreach (ShapeVaeKeyMap.TensorSpec spec in specs)
        {
            Assert.True(loader.Descriptors.TryGetValue(spec.CheckpointKey, out PytorchTensorDescriptor? descriptor), $"Missing {spec.CheckpointKey}.");
            Assert.Equal(new TensorShape(spec.Shape), descriptor!.Shape);
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
