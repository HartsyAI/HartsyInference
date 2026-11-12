using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>The input preparation a 3D model applies, on its own: what <see cref="MeshPreprocess.Auto"/> does inside a generation, returned as an image.</summary>
public interface IMeshPreparationService
{
    /// <summary>Prepares <paramref name="image"/> for <paramref name="spec"/>'s model: mattes it when it has no alpha, then applies the model's framing and compositing.</summary>
    /// <exception cref="NotSupportedException">The model has no wired pipeline, or the matting checkpoint is needed and not installed. A model that takes no image, or prepares its own, gets the image back with a note.</exception>
    Task<PreparedMeshImage> PrepareImageAsync(ModelSpec spec, ImageData image, CancellationToken cancel = default);
}
