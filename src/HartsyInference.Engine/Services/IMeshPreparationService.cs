using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>The input preparation a 3D model applies, on its own: what <see cref="MeshPreprocess.Auto"/> does inside a generation, returned as an image. TRELLIS.2 and Pixal3D ignore <c>Auto</c> because they matte and crop in every run; for them this is that same preprocessing run on its own.</summary>
public interface IMeshPreparationService
{
    /// <summary>Prepares <paramref name="image"/> for <paramref name="spec"/>'s model: mattes it when it has no alpha, then applies the model's framing and compositing.</summary>
    /// <exception cref="NotSupportedException">The model has no wired pipeline, or the matting checkpoint is needed and not installed. A model that takes no image (UniRig, CoLa3D, P3-SAM, X-Part) or prepares its own where no preview is possible (HairDeck, SAM 3D) gets the image back with a note; TRELLIS.2 and Pixal3D return their own crop as the image and the cutout it was cropped from.</exception>
    Task<PreparedMeshImage> PrepareImageAsync(ModelSpec spec, ImageData image, CancellationToken cancel = default);
}
