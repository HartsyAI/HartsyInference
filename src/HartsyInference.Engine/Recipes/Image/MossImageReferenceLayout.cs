namespace HartsyInference.Engine.Recipes.Image;

/// <summary>The reference-image layout a MOSS-Image edit is conditioned with, chosen by <c>reference.layout</c>; each
/// layout matches how one family of reference LoRAs was trained.</summary>
public enum MossImageReferenceLayout
{
    /// <summary>ai-toolkit's edit layout: up to three references, each under a <c>Picture N:</c> label, in the cached or joint
    /// attention form the adapter was trained with.</summary>
    Ostris,

    /// <summary>The colorize LoRA's layout: one page at half size per side, between the image and the text, with its
    /// vision copy at the target's pixel count and no label.</summary>
    Colorize,
}
