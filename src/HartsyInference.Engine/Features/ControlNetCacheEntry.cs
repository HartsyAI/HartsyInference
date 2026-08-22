using HartsyInference.Diffusion.Adapters;

namespace HartsyInference.Engine.Features;

/// <summary>A loaded UNet-family ControlNet kept alive for one generation: the mmap-backed checkpoint plus the constructed adapter, disposed together.</summary>
public sealed class ControlNetCacheEntry : IDisposable
{
    /// <summary>Path the checkpoint was loaded from.</summary>
    public required string FilePath { get; init; }

    /// <summary>The mmap-backed checkpoint; keeps the adapter's weight tensors valid.</summary>
    public required ControlNetFile File { get; init; }

    /// <summary>The constructed, weights-loaded adapter.</summary>
    public required ControlNet Adapter { get; init; }

    private bool _disposed;

    /// <summary>Disposes the adapter then the backing file.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        Adapter.Dispose();
        File.Dispose();
    }
}
