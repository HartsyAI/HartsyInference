using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Tests.Models.HairDeck;

public sealed class HairDeckModelRootTests : IDisposable
{
    private readonly string _base = Path.Combine(Path.GetTempPath(), "hairdeck-root-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_base))
        {
            Directory.Delete(_base, recursive: true);
        }
    }

    [Fact]
    public void ResolveModelRoot_FromTheCatalogPrimaryCheckpoint_IsTheFolderHoldingConfigJson()
    {
        string root = MakeModelFolder();
        string checkpoint = Path.Combine(root, "ckpt", "image_model", "model.safetensors");

        Assert.Equal(root, HairDeckModel.ResolveModelRoot(checkpoint));
    }

    [Fact]
    public void ResolveModelRoot_FromTheFolderItself_IsTheFolder()
    {
        string root = MakeModelFolder();

        Assert.Equal(root, HairDeckModel.ResolveModelRoot(root));
    }

    [Fact]
    public void ResolveModelRoot_WithoutConfigJson_FallsBackToThePathsFolder()
    {
        string folder = Path.Combine(_base, "empty");
        Directory.CreateDirectory(folder);
        string file = Path.Combine(folder, "model.safetensors");
        File.WriteAllBytes(file, []);

        Assert.Equal(folder, HairDeckModel.ResolveModelRoot(file));
    }

    private string MakeModelFolder()
    {
        string root = Path.Combine(_base, "3D", "HairDeck");
        Directory.CreateDirectory(Path.Combine(root, "ckpt", "image_model"));
        File.WriteAllText(Path.Combine(root, "config.json"), "{}");
        File.WriteAllBytes(Path.Combine(root, "ckpt", "image_model", "model.safetensors"), []);
        return root;
    }
}
