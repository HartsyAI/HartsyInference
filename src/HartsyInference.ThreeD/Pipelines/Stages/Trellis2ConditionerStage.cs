using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>DINOv3 conditioning through <see cref="Trellis2ImageConditioner"/>, with the encoder's weights resident on the backend
/// only while it runs.</summary>
public sealed class Trellis2ConditionerStage : ITrellis2Conditioner, IDisposable
{
    private readonly Trellis2ImageConditioner _conditioner;

    /// <param name="conditioner">This stage owns it.</param>
    public Trellis2ConditionerStage(Trellis2ImageConditioner conditioner)
        => _conditioner = conditioner ?? throw new ArgumentNullException(nameof(conditioner));

    /// <inheritdoc/>
    public Tensor Encode(IBackend backend, Trellis2PreparedImage image, int resolution)
    {
        ArgumentNullException.ThrowIfNull(backend);
        backend.PreloadWeights(_conditioner.EnumerateWeights());
        try
        {
            return _conditioner.Encode(backend, image.Rgb, image.Width, image.Height, resolution);
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(_conditioner.EnumerateWeights());
        }
    }

    public void Dispose() => _conditioner.Dispose();
}
