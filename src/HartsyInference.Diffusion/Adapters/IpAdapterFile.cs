using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>A loaded IP-Adapter checkpoint (safetensors, or the torch-pickle <c>.bin</c> FaceID ships) with auto-detected variant (Standard / Plus / FaceID), base model, and parsed weight dictionary. The underlying loader (mmap for safetensors, owned tensor copies for pickle) is owned by this instance.</summary>
public sealed class IpAdapterFile : IDisposable
{
    private IDisposable? _loader;
    private int _disposed;

    /// <summary>Path of the loaded checkpoint file.</summary>
    public required string FilePath { get; init; }

    /// <summary>Auto-detected base model architecture (Sd15, Sdxl, or Flux).</summary>
    public required IpAdapterBaseModel BaseModel { get; init; }

    /// <summary>Auto-derived config for the detected variant + base model.</summary>
    public required IpAdapterConfig Config { get; init; }

    /// <summary>All parsed tensors keyed by their original safetensors key.</summary>
    public required IReadOnlyDictionary<string, Tensor> Weights { get; init; }

    internal void AttachLoader(IDisposable loader) => _loader = loader;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _loader?.Dispose();
        _loader = null;
    }
}
