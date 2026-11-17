using HartsyInference.Cli.Dispatch;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>The 3D command's <c>--view side=path</c> and <c>--preprocess</c> values.</summary>
public sealed class MeshViewArgumentTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("hartsy-views-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string Picture(string name)
    {
        string path = Path.Combine(_folder, name);
        File.WriteAllBytes(path, PngEncoder.Encode(new byte[2 * 2 * 3], [255, 0, 255, 0], 2, 2));
        return path;
    }

    [Fact]
    public void EachViewIsReadWithItsSideAndAlpha()
    {
        IReadOnlyList<MeshViewImage>? views = GenerationDispatch.LoadMeshViews($"left={Picture("l.png")}\nBACK=\"{Picture("b.png")}\"");

        Assert.Equal([MeshView.Left, MeshView.Back], views!.Select(v => v.View));
        Assert.Equal([255, 0, 255, 0], views![0].Image.Alpha);
        Assert.Null(GenerationDispatch.LoadMeshViews(null));
    }

    [Theory]
    [InlineData("front=x.png")]
    [InlineData("1=x.png")]
    [InlineData("left")]
    public void AnEntryThatIsNotASideAndAPathIsRejected(string entry) =>
        Assert.Throws<ArgumentException>(() => GenerationDispatch.LoadMeshViews(entry));

    [Fact]
    public void AMissingFileIsNamed() =>
        Assert.Throws<FileNotFoundException>(() => GenerationDispatch.LoadMeshViews($"right={Path.Combine(_folder, "none.png")}"));

    [Fact]
    public void PreprocessIsNoneOrAuto()
    {
        Assert.Equal(MeshPreprocess.None, GenerationDispatch.ParseMeshPreprocess(null));
        Assert.Equal(MeshPreprocess.Auto, GenerationDispatch.ParseMeshPreprocess("AUTO"));
        Assert.Throws<ArgumentException>(() => GenerationDispatch.ParseMeshPreprocess("1"));
        Assert.Throws<ArgumentException>(() => GenerationDispatch.ParseMeshPreprocess("crop"));
    }
}
