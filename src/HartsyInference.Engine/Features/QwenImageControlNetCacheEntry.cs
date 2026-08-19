using HartsyInference.Diffusion.Adapters;

namespace HartsyInference.Engine.Features;

/// <summary>A loaded Qwen-Image DiT ControlNet: the mmap-backed checkpoint plus the constructed adapter.</summary>
public sealed class QwenImageControlNetCacheEntry : IDisposable
{
    /// <summary>The mmap-backed checkpoint; keeps the adapter's weight tensors valid.</summary>
    public required ControlNetFile File { get; init; }

    /// <summary>The constructed, weights-loaded Qwen adapter.</summary>
    public required QwenImageControlNet Adapter { get; init; }

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
