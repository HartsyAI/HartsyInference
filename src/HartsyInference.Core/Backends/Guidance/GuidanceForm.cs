namespace HartsyInference.Core.Backends.Guidance;

/// <summary>How a guided prediction is formed from the conditional and unconditional ones. The two forms are the same value
/// in exact arithmetic and differ in the last bit in F32, so a model keeps the form its reference uses.</summary>
public enum GuidanceForm
{
    /// <summary><c>g · cond + (1 − g) · uncond</c>: the flow-match samplers and the autoregressive logit blends.</summary>
    Lerp,

    /// <summary><c>uncond + g · (cond − uncond)</c>: the normalized samplers (Lance, Lumina 2).</summary>
    Extrapolate,
}
