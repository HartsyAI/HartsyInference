using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.BlockScale;

namespace HartsyInference.LLM.Ernie5;

/// <summary>ERNIE-5.5 weights read from a safetensors directory: <see cref="BlockFp8CheckpointWeights"/> under this family's name. The grouped
/// output projection is stored as <c>self_attn.linear_o_group_proj</c> with <c>linear_o_group_proj_scale_inv</c> (no <c>.weight</c>); the checkpoint
/// reader pairs any F8 tensor with its <c>_scale_inv</c> key, so it reads the same way.</summary>
public sealed class Ernie5CheckpointWeights : IErnie5WeightSource, IDisposable
{
    private readonly BlockFp8CheckpointWeights _weights;

    private Ernie5CheckpointWeights(BlockFp8CheckpointWeights weights) => _weights = weights;

    /// <summary>Opens <paramref name="directory"/>'s <c>model.safetensors.index.json</c> shards; FP8 scale blocks are
    /// <paramref name="block"/> on each edge.</summary>
    public static Ernie5CheckpointWeights Open(string directory, int block = Fp8BlockF32Codec.DefaultBlock) =>
        new(BlockFp8CheckpointWeights.Open(directory, "ERNIE-5.5", block));

    /// <inheritdoc />
    public IEnumerable<string> Names => _weights.Names;

    /// <inheritdoc />
    public Tensor Read(string name) => _weights.Read(name);

    /// <inheritdoc />
    public void Dispose() => _weights.Dispose();
}
