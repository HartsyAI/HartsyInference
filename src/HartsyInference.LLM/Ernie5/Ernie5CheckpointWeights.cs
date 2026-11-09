using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.LLM.Ernie5;

/// <summary>ERNIE-5.5 weights read from a safetensors directory through <see cref="HfQuantCheckpoint"/>: BF16 and F32 tensors as mapped
/// views, FP8 e4m3 weights with their 128x128 F32 <c>weight_scale_inv</c> grid dequantized to F32 by <see cref="Fp8BlockF32Codec"/>.</summary>
/// <remarks>The grouped output projection is stored as <c>self_attn.linear_o_group_proj</c> with <c>linear_o_group_proj_scale_inv</c> (no
/// <c>.weight</c>); the checkpoint reader pairs any F8 tensor with its <c>_scale_inv</c> key, so it reads the same way.</remarks>
public sealed class Ernie5CheckpointWeights : IErnie5WeightSource, IDisposable
{
    private HfQuantCheckpoint? _checkpoint;

    private Ernie5CheckpointWeights(HfQuantCheckpoint checkpoint) => _checkpoint = checkpoint;

    /// <summary>Opens <paramref name="directory"/>'s <c>model.safetensors.index.json</c> shards; FP8 scale blocks are
    /// <paramref name="block"/> on each edge.</summary>
    public static Ernie5CheckpointWeights Open(string directory, int block = Fp8BlockF32Codec.DefaultBlock) =>
        new(HfQuantCheckpoint.Open(directory, fp8BlockSize: block));

    /// <inheritdoc />
    public IEnumerable<string> Names => Checkpoint.Weights.Keys;

    /// <inheritdoc />
    public Tensor Read(string name)
    {
        HfQuantCheckpoint checkpoint = Checkpoint;
        if (!checkpoint.Weights.TryGetValue(name, out HfQuantWeight? weight))
            throw new KeyNotFoundException($"The ERNIE-5.5 checkpoint has no weight '{name}'.");
        return weight.Storage switch
        {
            HfWeightStorage.Plain => checkpoint.Borrow(name),
            HfWeightStorage.Fp8Block => checkpoint.DequantizeF32(name),
            _ => throw new InvalidDataException(
                $"'{name}' is stored as {weight.Storage}; an ERNIE-5.5 checkpoint holds only float and 128x128 block-FP8 weights."),
        };
    }

    private HfQuantCheckpoint Checkpoint => _checkpoint ?? throw new ObjectDisposedException(nameof(Ernie5CheckpointWeights));

    /// <inheritdoc />
    public void Dispose() => Interlocked.Exchange(ref _checkpoint, null)?.Dispose();
}
