namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>Outcome of the mask stage. A missing hair or an empty person search still yields a crop; the status says so.</summary>
public enum HairDeckPrepStatus
{
    /// <summary>Person, hair and face were all found.</summary>
    Ok,

    /// <summary>No person was found; the matte covers the whole image (or the input alpha).</summary>
    NoPersonMask,

    /// <summary>No hair was found; the crop is made from the face alone.</summary>
    NoHair,
}
