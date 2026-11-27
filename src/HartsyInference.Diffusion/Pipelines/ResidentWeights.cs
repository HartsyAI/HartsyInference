using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Whether one model's weights are preloaded on a backend; a release frees them unless the VRAM profile keeps models resident.</summary>
public sealed class ResidentWeights
{
    private readonly IBackend _backend;
    private readonly Func<IEnumerable<Tensor>> _weights;

    /// <summary>Tracks <paramref name="weights"/> on <paramref name="backend"/>; nothing is preloaded yet.</summary>
    public ResidentWeights(IBackend backend, Func<IEnumerable<Tensor>> weights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(weights);
        _backend = backend;
        _weights = weights;
    }

    /// <summary>True between a <see cref="Preload"/> that returned and the release that freed the weights.</summary>
    public bool IsResident { get; private set; }

    /// <summary>Uploads the weights as one bulk preload. A preload that throws may have uploaded some of them, so every weight is freed
    /// again before the exception propagates, and the model is left marked not resident. A failure while freeing is dropped so the
    /// preload's exception is the one the caller sees.</summary>
    public void Preload()
    {
        try
        {
            _backend.PreloadWeights(_weights());
        }
        catch
        {
            try
            {
                _backend.FreeWeights(_weights());
            }
            catch
            {
                // The original exception matters more than a failure to undo a partial upload.
            }
            IsResident = false;
            throw;
        }
        IsResident = true;
    }

    /// <summary>Frees the device weights unless the VRAM profile keeps models resident; <paramref name="force"/> frees them anyway, for a text encoder that needs the room.</summary>
    public void Release(bool force = false)
    {
        if (!IsResident || (!force && VramLevers.KeepResident(_backend)))
            return;
        _backend.FreeWeights(_weights());
        IsResident = false;
    }
}
