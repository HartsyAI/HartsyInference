using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.PyTorch;

public sealed class RepackedCheckpointTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("repacked-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Theory]
    [InlineData("unet/diffusion_pytorch_model.bin", "unet/diffusion_pytorch_model.safetensors")]
    [InlineData("model.fp16.ckpt", "model.fp16.safetensors")]
    [InlineData("naf_release.pth", "naf_release.safetensors")]
    [InlineData("already.safetensors", "already.safetensors")]
    public void TheRepackSitsBesideThePickleUnderTheSameNameWithTheSafetensorsExtension(string pickle, string repack)
    {
        Assert.Equal(repack.Replace('/', Path.DirectorySeparatorChar), RepackedCheckpoint.SiblingOf(pickle.Replace('/', Path.DirectorySeparatorChar)));
    }

    [Fact]
    public void ResolvePrefersTheRepackAndFallsBackToThePickle()
    {
        string pickle = Path.Combine(_folder, "w.bin");
        string repack = Path.Combine(_folder, "w.safetensors");

        Assert.Equal(pickle, RepackedCheckpoint.Resolve(pickle));
        Assert.False(RepackedCheckpoint.Exists(pickle));
        File.WriteAllText(pickle, "p");
        Assert.True(RepackedCheckpoint.Exists(pickle));
        Assert.Equal(pickle, RepackedCheckpoint.Resolve(pickle));
        File.WriteAllText(repack, "r");
        Assert.Equal(repack, RepackedCheckpoint.Resolve(pickle));
        File.Delete(pickle);
        Assert.True(RepackedCheckpoint.Exists(pickle));
        Assert.Equal(repack, RepackedCheckpoint.Resolve(pickle));
    }
}
