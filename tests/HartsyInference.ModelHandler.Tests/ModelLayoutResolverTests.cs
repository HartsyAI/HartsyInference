using HartsyInference.ModelHandler.Registry;
using Xunit;

namespace HartsyInference.ModelHandler.Tests;

/// <summary>Layout resolution for the three on-disk shapes: single file, sharded directory, and
/// diffusers-layout directory. Uses empty placeholder files in a temp directory (the resolver only
/// inspects names/structure, not file contents).</summary>
public sealed class ModelLayoutResolverTests : IDisposable
{
    private readonly string _root;

    public ModelLayoutResolverTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "hi-layout-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void Resolve_SingleFile()
    {
        string file = Touch(Path.Combine(_root, "model.safetensors"));
        ModelLayout layout = ModelLayoutResolver.Resolve(file);

        Assert.Equal(ModelLayoutKind.SingleFile, layout.Kind);
        Assert.Equal(file, layout.RepresentativeFile);
        Assert.Single(layout.SafeTensorsFiles);
    }

    [Fact]
    public void Resolve_FlatDirWithOneSafetensors_IsSingleFile()
    {
        string dir = NewDir("bundled");
        string file = Touch(Path.Combine(dir, "sdxl.safetensors"));
        ModelLayout layout = ModelLayoutResolver.Resolve(dir);

        Assert.Equal(ModelLayoutKind.SingleFile, layout.Kind);
        Assert.Equal(file, layout.RepresentativeFile);
    }

    [Fact]
    public void Resolve_ShardedDir()
    {
        string dir = NewDir("sharded");
        Touch(Path.Combine(dir, "model-00001-of-00002.safetensors"));
        Touch(Path.Combine(dir, "model-00002-of-00002.safetensors"));
        ModelLayout layout = ModelLayoutResolver.Resolve(dir);

        Assert.Equal(ModelLayoutKind.Sharded, layout.Kind);
        Assert.Equal(2, layout.SafeTensorsFiles.Count);
        // Representative is the first shard (sorted).
        Assert.EndsWith("model-00001-of-00002.safetensors", layout.RepresentativeFile);
    }

    [Fact]
    public void Resolve_DiffusersLayout_PicksTransformerAsRepresentative()
    {
        string dir = NewDir("diffusers");
        Touch(Path.Combine(dir, "model_index.json"));
        string transformerDir = NewDir(Path.Combine(dir, "transformer"));
        string tFile = Touch(Path.Combine(transformerDir, "diffusion_pytorch_model.safetensors"));
        string vaeDir = NewDir(Path.Combine(dir, "vae"));
        Touch(Path.Combine(vaeDir, "diffusion_pytorch_model.safetensors"));

        ModelLayout layout = ModelLayoutResolver.Resolve(dir);

        Assert.Equal(ModelLayoutKind.Diffusers, layout.Kind);
        Assert.Equal(tFile, layout.RepresentativeFile);
        Assert.Equal(2, layout.SafeTensorsFiles.Count);
    }

    [Fact]
    public void Resolve_DiffusersLayout_UnetSubdir()
    {
        string dir = NewDir("diffusers-unet");
        string unetDir = NewDir(Path.Combine(dir, "unet"));
        string uFile = Touch(Path.Combine(unetDir, "diffusion_pytorch_model.safetensors"));

        ModelLayout layout = ModelLayoutResolver.Resolve(dir);

        Assert.Equal(ModelLayoutKind.Diffusers, layout.Kind);
        Assert.Equal(uFile, layout.RepresentativeFile);
    }

    [Fact]
    public void Resolve_MissingPath_Throws()
    {
        Assert.Throws<FileNotFoundException>(() =>
            ModelLayoutResolver.Resolve(Path.Combine(_root, "does-not-exist")));
    }

    private string NewDir(string nameOrPath)
    {
        string path = Path.IsPathRooted(nameOrPath) ? nameOrPath : Path.Combine(_root, nameOrPath);
        Directory.CreateDirectory(path);
        return path;
    }

    private static string Touch(string path)
    {
        File.WriteAllText(path, "");
        return path;
    }
}
