using HartsyInference.LLM.Multimodal;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>The image route of a Kimi K3.1 checkpoint: the MoonViT-V2 encoder with its projector, the image geometry and the media pad id.</summary>
/// <param name="Encoder">Owned by the <see cref="KimiVisionInput"/> that returned this route; that input disposes it.</param>
public sealed record KimiVisionRoute(IVlmImageEncoder Encoder, KimiImageSpec Image, int PadId);
