using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed image-generation surface: detect the model's architecture, apply the request's composition (LoRA/ControlNet/IP-Adapter/Refiner/img2img/inpaint/regional), and generate.</summary>
public interface IImagesService
{
    /// <summary>Generates one image for <paramref name="request"/> against the model named by <paramref name="spec"/>.</summary>
    Task<ImageResult> GenerateAsync(ModelSpec spec, ImageRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default);
}
