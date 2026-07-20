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
}
