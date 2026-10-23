using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>What <see cref="Fibo2Pipeline.GenerateEdit"/> conditions on besides the target size: the edit prompt with its vision
/// tokens, the images as the vision tower and the VAE each read them, and the components that run them. The tensors and
/// components are borrowed.</summary>
public sealed record Fibo2EditInputs
{
    /// <summary>Template-wrapped edit prompt ids, each image's <c>&lt;|image_pad|&gt;</c> run already expanded to its vision-token
    /// count (<see cref="Prompting.Fibo2EditPrompt.Tokenize"/>).</summary>
    public required int[] ConditionalTokens { get; init; }

    /// <summary>The vision tower's copy of each image, RGB <c>[3, H, W]</c> in [0, 1], in marker order. Sized so the tower's own
    /// resize is a no-op (ComfyUI's <c>to_qwen_grid</c>).</summary>
    public required IReadOnlyList<Tensor> VisionImages { get; init; }

    /// <summary>The VAE's copy of each image, RGB <c>[1, 3, H, W]</c> in [-1, 1], sides multiples of 16, in marker order.</summary>
    public required IReadOnlyList<Tensor> ReferencePixels { get; init; }

    /// <summary>Runs the Qwen3-VL language tower with the vision tokens spliced in; it shares the pipeline's language weights.</summary>
    public required Qwen3VlMultimodalEncoder VisionEncoder { get; init; }

    /// <summary>The FLUX.2 VAE encoder.</summary>
    public required VaeEncoder VaeEncoder { get; init; }
}
