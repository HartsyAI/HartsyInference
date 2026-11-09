using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.PyTorch;

/// <summary>Real-weight load of the CoLa3D SS flow checkpoint (complex RoPE buffer). Integration lane: needs Models/CoLa3D.</summary>
[Trait("Category", "Integration")]
public sealed class CoLa3DCheckpointLoadTests
{
    [Fact]
    public void SsFlowQ8_LoadsAllTensorsAndReadsComplexRopePhasesAsRealPairs()
    {
        string path = Path.Combine(FindRepoRoot(), "Models", "CoLa3D", "ss_flow_q8.pt");
        Assert.True(File.Exists(path), $"Integration weights missing: {path}");

        PytorchPickleLoader loader = new();
        loader.Load(path);

        Assert.Equal(641, loader.Descriptors.Count);
        Assert.Equal(0, loader.SkippedTensorCount);
        PytorchTensorDescriptor rope = loader.Descriptors["rope_phases"];
        Assert.Equal(DType.F32, rope.DType);
        Assert.Equal(new TensorShape(4096, 64, 2), rope.Shape);
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
