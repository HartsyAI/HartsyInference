using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.LLM.Ernie5;

/// <summary>ERNIE-5.5 weights read from a safetensors directory through <see cref="HfQuantCheckpoint"/>: BF16 and F32 tensors as mapped
/// views, and FP8 e4m3 weights as their stored tensor with the 128x128 F32 <c>weight_scale_inv</c> grid on
/// <see cref="Tensor.QuantInfo"/>. A linear on such a weight runs <c>IBackend.Fp8BlockLinear</c> on any backend, so the FP8 weights stay
/// packed; a caller that needs F32 decodes them with <see cref="Ernie5Layer.ReadF32"/>.</summary>
/// <remarks>The grouped output projection is stored as <c>self_attn.linear_o_group_proj</c> with <c>linear_o_group_proj_scale_inv</c> (no
/// <c>.weight</c>); the checkpoint reader pairs any F8 tensor with its <c>_scale_inv</c> key, so it reads the same way.
/// Each weight keeps its own scale view, cached by weight name so repeated reads share one view; all are disposed with this source.</remarks>
public sealed class Ernie5CheckpointWeights : IErnie5WeightSource, IDisposable
{
    private HfQuantCheckpoint? _checkpoint;
    private readonly Dictionary<string, Tensor> _scales = new(StringComparer.Ordinal);

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
            HfWeightStorage.Fp8Block => ReadFp8(checkpoint, name),
            _ => throw new InvalidDataException(
                $"'{name}' is stored as {weight.Storage}; an ERNIE-5.5 checkpoint holds only float and 128x128 block-FP8 weights."),
        };
    }

    private Tensor ReadFp8(HfQuantCheckpoint checkpoint, string name)
    {
        (Tensor weight, Tensor scale) = checkpoint.BorrowFp8Block(name);
        if (_scales.TryGetValue(name, out Tensor? kept)) scale.Dispose();
        else _scales[name] = kept = scale;
        weight.QuantInfo = new QuantWeightInfo
        {
            Format = "fp8_block_scale_inv", Fp8BlockScaleInv = kept, Fp8BlockSize = checkpoint.Fp8BlockSize,
        };
        return weight;
    }

    private HfQuantCheckpoint Checkpoint => _checkpoint ?? throw new ObjectDisposedException(nameof(Ernie5CheckpointWeights));

    /// <inheritdoc />
    public void Dispose()
    {
        HfQuantCheckpoint? checkpoint = Interlocked.Exchange(ref _checkpoint, null);
        if (checkpoint is null) return;
        foreach (Tensor scale in _scales.Values) scale.Dispose();
        _scales.Clear();
        checkpoint.Dispose();
    }
}
