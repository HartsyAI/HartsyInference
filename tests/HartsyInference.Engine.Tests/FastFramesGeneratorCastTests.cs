using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The FastFrames generator reader on a safetensors file: on CPU the tensors come back F32 as owned copies, and disposing
/// the loaders frees those copies; off CPU the file is passed through as stored.</summary>
public sealed class FastFramesGeneratorCastTests : IDisposable
{
    private static readonly float[] Values = [1f, 2f, 3f, 4f, 5f, 6f];
    private readonly string _dir = Directory.CreateTempSubdirectory("ff-cast-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    /// <summary>A two-by-three F16 generator under one original Wan key, the dtype a BF16 or F16 release arrives in.</summary>
    private string WriteHalfGenerator()
    {
        string path = Path.Combine(_dir, "generator.safetensors");
        using Tensor source = new Tensor(new TensorShape(2, 3), DType.F32);
        Values.AsSpan().CopyTo(source.AsSpan<float>());
        using Tensor half = source.CastTo(DType.F16);
        SafeTensorsWriter.Save(path, new Dictionary<string, Tensor> { ["blocks.0.weight"] = half });
        return path;
    }

    [Fact]
    public void OnCpuTheGeneratorComesBackF32AndItsCopiesAreFreedWithTheLoaders()
    {
        string path = WriteHalfGenerator();
        List<IDisposable> loaders = [];

        Dictionary<string, Tensor> weights = FastFramesRecipe.ReadGenerator(path, DType.F32, castForCpu: true, loaders);

        Tensor copy = weights["blocks.0.weight"];
        Assert.Equal(DType.F32, copy.DType);
        Assert.Equal(Values, copy.AsSpan<float>().ToArray());
        foreach (IDisposable loader in loaders)
        {
            loader.Dispose();
        }
        Assert.Throws<ObjectDisposedException>(() => copy.AsSpan<float>().ToArray());
    }

    [Fact]
    public void OffCpuTheGeneratorIsPassedThroughInItsStoredDtypeWithoutCopies()
    {
        string path = WriteHalfGenerator();
        List<IDisposable> loaders = [];

        Dictionary<string, Tensor> weights = FastFramesRecipe.ReadGenerator(path, DType.F16, castForCpu: false, loaders);

        Assert.Equal(DType.F16, weights["blocks.0.weight"].DType);
        // The only loader is the file's own; no owned cast copies were registered.
        Assert.Single(loaders);
        foreach (IDisposable loader in loaders)
        {
            loader.Dispose();
        }
    }
}
