using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Strictly converted PDD trunk updates and the temporary tensors created for fused H3 targets.</summary>
public sealed class MiniMaxH3PddTrunkConversion : IDisposable
{
    private readonly IReadOnlyList<Tensor> _ownedTensors;
    private int _disposed;

    /// <summary>Creates a conversion result whose allocated fused matrices remain owned here.</summary>
    public MiniMaxH3PddTrunkConversion(IReadOnlyList<LoraLayer> layers,
        IReadOnlyList<LoraFullWeightDiff> fullWeightDiffs, IReadOnlyList<Tensor> ownedTensors)
    {
        Layers = layers ?? throw new ArgumentNullException(nameof(layers));
        FullWeightDiffs = fullWeightDiffs ?? throw new ArgumentNullException(nameof(fullWeightDiffs));
        _ownedTensors = ownedTensors ?? throw new ArgumentNullException(nameof(ownedTensors));
    }

    /// <summary>Canonical H3 low-rank updates, including fused QKV and corrected SwiGLU row order.</summary>
    public IReadOnlyList<LoraLayer> Layers { get; }

    /// <summary>Mandatory weight/DC-bias diffs produced when dense AdaLN updates were rebased to a curve table.</summary>
    public IReadOnlyList<LoraFullWeightDiff> FullWeightDiffs { get; }

    /// <summary>Creates a non-owning standard LoRA view for the existing merge machinery.</summary>
    public LoraFile CreateLoraView(string filePath, IReadOnlyDictionary<string, string>? metadata)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return new LoraFile
        {
            FilePath = filePath,
            Format = LoraFormat.DiffusersBareDit,
            Layers = Layers,
            FullWeightDiffs = FullWeightDiffs,
            Metadata = metadata,
        };
    }

    /// <summary>Releases only matrices allocated by conversion; mmap-backed source tensors remain externally owned.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor tensor in _ownedTensors) tensor.Dispose();
    }
}
