using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>A loaded ControlNet safetensors file with auto-detected base model and parsed weight dictionary. The file's mmap-backed data is owned by this instance — disposing this object invalidates the tensors.</summary>
public sealed class ControlNetFile : IDisposable
{
    private SafeTensorsLoader? _loader;
    private int _disposed;

    /// <summary>Path of the loaded safetensors file.</summary>
    public required string FilePath { get; init; }

    /// <summary>Auto-detected base model architecture (Sd15, Sdxl, or Flux).</summary>
    public required ControlNetBaseModel BaseModel { get; init; }

    /// <summary>Heuristically detected conditioning mode from filename keywords. Falls back to <see cref="ControlNetMode.Depth"/> when no match found — caller may override.</summary>
    public required ControlNetMode Mode { get; init; }

    /// <summary>Auto-derived config based on the detected base model and the standard preset for that family.</summary>
    public required ControlNetConfig Config { get; init; }

    /// <summary>All parsed tensors keyed by their original safetensors key.</summary>
    public required IReadOnlyDictionary<string, Tensor> Weights { get; init; }

    internal void AttachLoader(SafeTensorsLoader loader) => _loader = loader;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _loader?.Dispose();
        _loader = null;
    }
}
