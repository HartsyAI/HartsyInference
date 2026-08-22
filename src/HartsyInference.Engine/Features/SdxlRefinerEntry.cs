using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Features;

/// <summary>A loaded SDXL refiner UNet plus the safetensors loader backing its weights, cached per refiner checkpoint so repeat generations do not re-read the ~6 GB file.</summary>
public sealed class SdxlRefinerEntry : IDisposable
{
    /// <summary>Path the refiner was loaded from; the cache key.</summary>
    public required string FilePath { get; init; }

    /// <summary>The refiner UNet, built with <see cref="UNetConfig.SdxlRefiner"/>.</summary>
    public required UNet Unet { get; init; }

    /// <summary>Loader owning the refiner weights' mmap; disposing it invalidates the UNet.</summary>
    public required SafeTensorsLoader Loader { get; init; }

    private bool _disposed;

    /// <summary>Releases the backing mmap, which invalidates the UNet's borrowed weight tensors.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        try
        {
            Loader.Dispose();
        }
        catch (Exception ex)
        {
            Logs.Error($"[Features][Refiner] Failed to dispose refiner '{FilePath}'.", ex);
        }
    }
}
