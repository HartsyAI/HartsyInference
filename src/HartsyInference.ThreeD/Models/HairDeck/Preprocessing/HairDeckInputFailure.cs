namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>Why a HairDeck input image cannot be preprocessed.</summary>
public enum HairDeckInputFailure
{
    /// <summary>Width, height or a pixel buffer does not match the image size.</summary>
    InvalidImage,

    /// <summary>Neither hair nor face was found, so no head crop can be made.</summary>
    NoHairOrFace,
}
