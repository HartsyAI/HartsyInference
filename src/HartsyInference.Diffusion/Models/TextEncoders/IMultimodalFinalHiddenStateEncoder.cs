using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>An <see cref="IFinalHiddenStateEncoder"/> that also reads images: each image's vision-tower tokens replace
/// that image's run of placeholder tokens in the prompt, as in Qwen-VL chat templates. The caller sizes the placeholder
/// runs with <see cref="CountImageTokens"/> before tokenizing.</summary>
public interface IMultimodalFinalHiddenStateEncoder : IFinalHiddenStateEncoder
{
    /// <summary>Whether a vision tower is loaded; false means image input is refused.</summary>
    bool HasVision { get; }

    /// <summary>The size the vision tower resizes a <paramref name="width"/>×<paramref name="height"/> image to.</summary>
    (int Width, int Height) VisionInputSize(int width, int height);

    /// <summary>Placeholder tokens one <paramref name="width"/>×<paramref name="height"/> image occupies in the prompt.</summary>
    int CountImageTokens(int width, int height);

    /// <summary>Encodes token ids whose placeholder runs hold <paramref name="images"/> (<c>[1, 3, H, W]</c> in
    /// <c>[0, 1]</c>, in order) and returns <c>[1, ids.Count, HiddenSize]</c> F32 on the host.</summary>
    Tensor Encode(IBackend backend, IReadOnlyList<int> tokenIds, IReadOnlyList<Tensor> images);

    /// <summary>The vision tower's weights, so the pipeline can free them after encoding.</summary>
    IEnumerable<Tensor> EnumerateVisionWeights();
}
