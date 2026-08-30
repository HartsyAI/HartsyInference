namespace HartsyInference.Engine.Requests;

/// <summary>Provenance of an already-preprocessed Fun ControlNet-Union stream.</summary>
public enum VideoControlKind
{
    /// <summary>Canny edge video.</summary>
    Canny = 0,
    /// <summary>Depth-map video.</summary>
    Depth = 1,
    /// <summary>HED soft-edge video.</summary>
    Hed = 2,
    /// <summary>MLSD line-segment video.</summary>
    Mlsd = 3,
    /// <summary>Pose/skeleton video.</summary>
    Pose = 4,
    /// <summary>User-preprocessed custom control video.</summary>
    Custom = 5,
    /// <summary>49-channel inpaint construction from control, visibility, and masked source latents.</summary>
    Inpaint = 6,
}
