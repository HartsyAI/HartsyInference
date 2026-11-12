using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshServiceXPartFolderTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "xpart-folder-" + Guid.NewGuid().ToString("N"));

    public MeshServiceXPartFolderTests() => Directory.CreateDirectory(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private void Touch(string relative)
    {
        string path = Path.Combine(_root, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [1]);
    }

    [Fact]
    public void XPartFolder_TakesTheFolderHoldingAllFourCheckpointsOrOneAbove()
    {
        foreach (string file in XPartPipeline.Files)
            Touch(file);

        Assert.Equal(Path.GetFullPath(_root), MeshService.XPartFolder(_root));
        Assert.Equal(Path.GetFullPath(_root), MeshService.XPartFolder(Path.Combine(_root, "model")));
        Assert.Equal(Path.GetFullPath(_root), MeshService.XPartFolder(Path.Combine(_root, "model", "model.safetensors")));
    }

    [Fact]
    public void XPartFolder_NamesTheMissingCheckpoints()
    {
        Touch("p3sam/p3sam.safetensors");

        FileNotFoundException error = Assert.Throws<FileNotFoundException>(() => MeshService.XPartFolder(_root));
        Assert.Contains("model/model.safetensors", error.Message);
        Assert.Contains("conditioner/conditioner.safetensors", error.Message);
        Assert.DoesNotContain("p3sam/p3sam.safetensors", error.Message);
    }
}
