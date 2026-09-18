using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Features;

/// <summary>The weight dictionaries a recipe hands <see cref="RecipeLoraMerge"/>, one per model component it owns. Leave a component null when the architecture has none.</summary>
/// <remarks>The dictionaries are mutated in place: a LoRA'd key is replaced by a tensor the returned stack owns, so
/// pass a copy of anything a cache still holds (<see cref="RecipeLoraMerge.ShallowClone"/>).</remarks>
public sealed record LoraMergeTargets
{
    /// <summary>The SD1.5 / SDXL UNet weights.</summary>
    public IDictionary<string, Tensor>? Unet { get; init; }

    /// <summary>The DiT / transformer weights.</summary>
    public IDictionary<string, Tensor>? Transformer { get; init; }

    /// <summary>The CLIP-L text-encoder weights.</summary>
    public IDictionary<string, Tensor>? ClipL { get; init; }

    /// <summary>The CLIP-G text-encoder weights (SDXL only).</summary>
    public IDictionary<string, Tensor>? ClipG { get; init; }

    /// <summary>The T5 / umT5 / LLM text-encoder weights.</summary>
    public IDictionary<string, Tensor>? TextEncoder2 { get; init; }
}
