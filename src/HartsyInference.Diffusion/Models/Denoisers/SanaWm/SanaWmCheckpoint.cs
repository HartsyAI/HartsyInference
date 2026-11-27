using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>A loaded SANA-WM DiT checkpoint. Owns every tensor in <see cref="Weights"/>; disposing it releases them all.</summary>
public sealed class SanaWmCheckpoint : IDisposable
{
    private int _disposed;

    internal SanaWmCheckpoint(SanaWmConfig config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        Config = config;
        Weights = weights;
    }

    /// <summary>Dimensions derived from the checkpoint header.</summary>
    public SanaWmConfig Config { get; }

    /// <summary>Converted tensors keyed by upstream name, valid until this checkpoint is disposed.</summary>
    public IReadOnlyDictionary<string, Tensor> Weights { get; }

    /// <summary>Releases every owned tensor. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor tensor in Weights.Values)
            tensor.Dispose();
    }
}
