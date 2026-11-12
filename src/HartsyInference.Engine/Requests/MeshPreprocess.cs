namespace HartsyInference.Engine.Requests;

/// <summary>Whether the engine prepares the request's image for the model.</summary>
public enum MeshPreprocess
{
    /// <summary>The image is used as given: the caller has already cut out and framed the object the way the model expects.</summary>
    None = 0,

    /// <summary>An image without alpha is matted (BiRefNet), then the model's own preparation is applied: Hunyuan3D recentres on white at 512 px, TripoSR crops and composites on gray, TRELLIS crops to 518 px and premultiplies alpha, Paint takes the cutout. Models that matte and frame their own input (TRELLIS.2, Pixal3D, HairDeck, SAM 3D) are unchanged.</summary>
    Auto = 1,
}
