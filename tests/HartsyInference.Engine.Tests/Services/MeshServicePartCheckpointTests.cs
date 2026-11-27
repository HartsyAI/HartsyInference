using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshServicePartCheckpointTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "p3sam-checkpoint-" + Guid.NewGuid().ToString("N"));

    public MeshServicePartCheckpointTests() => Directory.CreateDirectory(Path.Combine(_root, "p3sam"));

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private string Touch(params string[] parts)
    {
        string path = Path.Combine([_root, .. parts]);
        File.WriteAllBytes(path, [1]);
        return path;
    }

    [Fact]
    public void PartCheckpoint_TakesAFileAsItIs()
    {
        string file = Touch("p3sam", "p3sam.safetensors");

        Assert.Equal(file, MeshService.PartCheckpoint(file));
    }

    [Fact]
    public void PartCheckpoint_FindsTheFileInAFolderOrItsP3SamSubfolder()
    {
        string file = Touch("p3sam", "p3sam.safetensors");

        Assert.Equal(file, MeshService.PartCheckpoint(_root));
        Assert.Equal(file, MeshService.PartCheckpoint(Path.Combine(_root, "p3sam")));
    }

    [Fact]
    public void PartCheckpoint_ReportsAFolderWithoutTheCheckpoint()
    {
        Assert.Throws<FileNotFoundException>(() => MeshService.PartCheckpoint(_root));
    }
}
