using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins which files a Lumina-2 selection expands to. The sharded diffusers release and a quantized repack of
/// it live in the same folder often enough that "an index exists here" cannot be the test: expanding a selected GGUF
/// into the index's safetensors loads a different model, or runs out of memory, with nothing to say the selection was
/// ignored.</summary>
public sealed class Lumina2ShardResolutionTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("hartsy-lumina2-shards").FullName;

    public void Dispose()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
            // A leftover temp directory is not worth failing a passing test over.
        }
    }

    /// <summary>Writes the two-shard diffusers release plus its index, and returns both shard paths.</summary>
    private (string First, string Second) WriteShardedRelease()
    {
        using Tensor first = new Tensor(new TensorShape(4, 8), DType.F32);
        using Tensor second = new Tensor(new TensorShape(4, 8), DType.F32);
        string firstPath = Path.Combine(_dir, "diffusion_pytorch_model-00001-of-00002.safetensors");
        string secondPath = Path.Combine(_dir, "diffusion_pytorch_model-00002-of-00002.safetensors");
        SafeTensorsWriter.Save(firstPath, new Dictionary<string, Tensor> { ["x_embedder.weight"] = first });
        SafeTensorsWriter.Save(secondPath, new Dictionary<string, Tensor> { ["layers.0.attn.to_q.weight"] = second });
        File.WriteAllText(Path.Combine(_dir, "diffusion_pytorch_model.safetensors.index.json"),
            """
            {
              "metadata": { "total_size": 1024 },
              "weight_map": {
                "x_embedder.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
                "layers.0.attn.to_q.weight": "diffusion_pytorch_model-00002-of-00002.safetensors"
              }
            }
            """);
        return (firstPath, secondPath);
    }

    [Fact]
    public void ResolveShardPaths_KeepsAGgufRepackParkedBesideTheShardedRelease()
    {
        WriteShardedRelease();
        string repack = Path.Combine(_dir, "lumina2-Q4_K_M.gguf");
        using (Tensor weight = new Tensor(new TensorShape(4, 8), DType.F32))
        using (GgufWriter writer = new GgufWriter(repack))
        {
            writer.SetMetadata("general.architecture", "lumina2");
            writer.AddTensor("x_embedder.weight", weight);
            writer.Flush();
        }

        Assert.Equal([repack], Lumina2CheckpointConverter.ResolveShardPaths(repack));
    }

    [Fact]
    public void ResolveShardPaths_ExpandsAShardTheIndexLists()
    {
        (string first, string second) = WriteShardedRelease();
        Assert.Equal([first, second], Lumina2CheckpointConverter.ResolveShardPaths(first));
        Assert.Equal([first, second], Lumina2CheckpointConverter.ResolveShardPaths(second));
    }

    [Fact]
    public void ResolveShardPaths_KeepsASingleFileCheckpointStoredBesideAnotherModelsIndex()
    {
        WriteShardedRelease();
        using Tensor weight = new Tensor(new TensorShape(4, 8), DType.F32);
        string single = Path.Combine(_dir, "lumina2-fp8_scaled.safetensors");
        SafeTensorsWriter.Save(single, new Dictionary<string, Tensor> { ["x_embedder.weight"] = weight });

        Assert.Equal([single], Lumina2CheckpointConverter.ResolveShardPaths(single));
    }
}
