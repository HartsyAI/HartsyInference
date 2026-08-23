namespace HartsyInference.Engine.Recipes;

/// <summary>The composition features an <see cref="IArchitectureRecipe"/> declares it can actually apply; anything a
/// request sets that the resolved recipe does not declare is rejected up front with a precise error.</summary>
[Flags]
public enum ImageFeatures
{
    /// <summary>Text-to-image only; every composition object is rejected.</summary>
    None = 0,

    /// <summary>LoRA stacks merged into the loaded weights.</summary>
    Lora = 1,

    /// <summary>ControlNet conditioning layers.</summary>
    ControlNet = 2,

    /// <summary>Image-prompt (IP-Adapter / Redux / FaceID) conditioning.</summary>
    IpAdapter = 4,

    /// <summary>Second-pass refiner.</summary>
    Refiner = 8,

    /// <summary>Image-to-image init.</summary>
    Img2Img = 16,

    /// <summary>Inpaint masking (implies an init image).</summary>
    Inpaint = 32,

    /// <summary>Regional / segment prompting.</summary>
    Regional = 64,

    /// <summary>Variation-seed noise blending.</summary>
    VariationSeed = 128,

    /// <summary>Reference-image editing: the init image is VAE-encoded and concatenated into the token sequence as in-context reference latents, rather than noised and denoised from.</summary>
    /// <remarks>Deliberately distinct from <see cref="Img2Img"/> even though both are driven by an init image, because the
    /// two obey different contracts: img2img's denoise strength selects a start step, while an edit model conditions on
    /// the reference at full strength and has no such knob. A family declaring only this bit must not silently accept a
    /// creativity value it cannot honour.</remarks>
    RefEdit = 256,

    /// <summary>Seamless/circularly-tileable output (wrap-pad every conv instead of zero-pad).</summary>
    SeamlessTiling = 512,
}
