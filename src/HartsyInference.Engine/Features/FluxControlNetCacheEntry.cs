using HartsyInference.Diffusion.Adapters;

namespace HartsyInference.Engine.Features;

/// <summary>A loaded Flux DiT ControlNet kept alive for one generation: the mmap-backed checkpoint plus its adapter.</summary>
public sealed class FluxControlNetCacheEntry : IDisposable
{
    /// <summary>The mmap-backed checkpoint; keeps the adapter's weight tensors valid.</summary>
    public required ControlNetFile File { get; init; }

    /// <summary>The constructed, weights-loaded Flux adapter.</summary>
    public required FluxControlNet Adapter { get; init; }

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
