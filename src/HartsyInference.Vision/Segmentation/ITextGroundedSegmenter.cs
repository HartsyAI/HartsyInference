using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation;

/// <summary>Model-agnostic text-grounded segmentation: an image and one or more text prompts in, scored instances
/// with boxes and masks out. Implementations own their weights; the caller owns the input tensor.</summary>
public interface ITextGroundedSegmenter : IDisposable
{
    /// <summary>Segments <paramref name="image"/> (<c>[1, 3, H, W]</c>, RGB in <c>[0, 1]</c>) for each prompt. Results follow the order of <paramref name="prompts"/>.</summary>
    IReadOnlyList<TextGroundedPromptResult> Segment(IBackend backend, Tensor image, IReadOnlyList<string> prompts);
}
