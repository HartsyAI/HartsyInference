using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.LLM.Transformer;

/// <summary>Weights read from a safetensors directory (index shards or one <c>model.safetensors</c>) through <see cref="HfQuantCheckpoint"/>: BF16 and
/// F32 tensors as mapped views, FP8 e4m3 weights as their stored tensor with the F32 block-scale grid on <see cref="Tensor.QuantInfo"/>. A linear on such
/// a weight runs <c>IBackend.Fp8BlockLinear</c> on any backend, so FP8 weights stay packed; a caller that needs F32 decodes them with
/// <see cref="ProjectionOps.EnsureF32"/>.</summary>
/// <remarks>Each weight keeps its own scale view, cached by weight name so repeated reads share one view; all are disposed with this source.</remarks>
public sealed class BlockFp8CheckpointWeights : IModelWeightSource, IDisposable
{
    private readonly string _model;
    private HfQuantCheckpoint? _checkpoint;
    private readonly Dictionary<string, Tensor> _scales = new(StringComparer.Ordinal);

    private BlockFp8CheckpointWeights(HfQuantCheckpoint checkpoint, string model)
    {
        _checkpoint = checkpoint;
        _model = model;
    }

    /// <summary>Opens <paramref name="directory"/>; FP8 scale blocks are <paramref name="block"/> on each edge. <paramref name="model"/> names the
    /// family in errors.</summary>
    public static BlockFp8CheckpointWeights Open(string directory, string model, int block = Fp8BlockF32Codec.DefaultBlock)
    {
        ArgumentException.ThrowIfNullOrEmpty(model);
        return new BlockFp8CheckpointWeights(HfQuantCheckpoint.Open(directory, fp8BlockSize: block), model);
    }

    /// <inheritdoc />
    public IEnumerable<string> Names => Checkpoint.Weights.Keys;

    /// <inheritdoc />
    public Tensor Read(string name)
    {
        HfQuantCheckpoint checkpoint = Checkpoint;
        if (!checkpoint.Weights.TryGetValue(name, out HfQuantWeight? weight))
            throw new KeyNotFoundException($"The {_model} checkpoint has no weight '{name}'.");
        return weight.Storage switch
        {
            HfWeightStorage.Plain => checkpoint.Borrow(name),
            HfWeightStorage.Fp8Block => ReadFp8(checkpoint, name),
            _ => throw new InvalidDataException(
                $"'{name}' is stored as {weight.Storage}; a {_model} checkpoint holds only float and block-FP8 weights."),
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

    private HfQuantCheckpoint Checkpoint => _checkpoint ?? throw new ObjectDisposedException(nameof(BlockFp8CheckpointWeights));

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
