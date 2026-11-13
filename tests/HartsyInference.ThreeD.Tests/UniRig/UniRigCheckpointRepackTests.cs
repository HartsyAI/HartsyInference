using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.UniRig;
using HartsyInference.ThreeD.Pipelines;
using Xunit;

namespace HartsyInference.ThreeD.Tests.UniRig;

/// <summary>A UniRig checkpoint whose pickle was repacked to safetensors beside it (<c>model.ckpt</c> and <c>model.safetensors</c>): the same tensors under the same keys, read without a pickle.</summary>
public sealed class UniRigCheckpointRepackTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("unirig-repack-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string WriteRepack(string relativePickle)
    {
        string pickle = Path.Combine(_folder, relativePickle);
        Directory.CreateDirectory(Path.GetDirectoryName(pickle)!);
        using Tensor weight = new(new TensorShape(2, 2), DType.F32);
        weight.AsSpan<float>().Fill(3f);
        using Tensor steps = new(new TensorShape(1), DType.I64);
        using Tensor outside = new(new TensorShape(1), DType.F32);
        SafeTensorsWriter.Save(Path.ChangeExtension(pickle, ".safetensors"), new Dictionary<string, Tensor>
        {
            ["state_dict.model.mesh_encoder.weight"] = weight,
            ["state_dict.model.mesh_encoder.norm.num_batches_tracked"] = steps,
            ["callbacks.ignored"] = outside,
        });
        return pickle;
    }

    [Fact]
    public void Load_ReadsTheRepackWhenThePickleIsGone_AndKeepsTheEnvelopeAndStepCounterRules()
    {
        string pickle = WriteRepack(UniRigPipeline.SkinCheckpoint);

        using UniRigCheckpoint checkpoint = UniRigCheckpoint.Load(pickle);

        KeyValuePair<string, Tensor> only = Assert.Single(checkpoint.Tensors);
        Assert.Equal("mesh_encoder.weight", only.Key);
        Assert.Equal([3f, 3f, 3f, 3f], only.Value.AsSpan<float>().ToArray());
    }

    [Fact]
    public void ThePipelineAcceptsAFolderHoldingOnlyRepacks()
    {
        WriteRepack(UniRigPipeline.SkeletonCheckpoint);
        WriteRepack(UniRigPipeline.SkinCheckpoint);

        using HartsyInference.Core.Backends.Cpu.CpuBackend backend = new();

        _ = new UniRigPipeline(backend, _folder);
    }
}
