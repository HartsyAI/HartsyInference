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

    /// <summary>Full DiT config derived from the checkpoint header when <see cref="BaseModel"/> is
    /// <see cref="ControlNetBaseModel.Flux"/> (block depths, union mode count, guidance); null otherwise.
    /// Feed it to <see cref="FluxControlNet(FluxControlNetConfig)"/>.</summary>
    public FluxControlNetConfig? FluxConfig { get; init; }

    /// <summary>All parsed tensors keyed by diffusers-format name. For LDM-layout checkpoints (<c>control_model.*</c>) the keys have been converted; for diffusers-layout files they are the original safetensors keys.</summary>
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
