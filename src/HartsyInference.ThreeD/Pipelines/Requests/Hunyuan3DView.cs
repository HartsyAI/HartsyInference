namespace HartsyInference.ThreeD.Pipelines.Requests;

/// <summary>The camera a multiview Hunyuan3D-2mv conditioning image was taken from; the value is its view index in the model.</summary>
public enum Hunyuan3DView
{
    /// <summary>Front view.</summary>
    Front = 0,

    /// <summary>Front rotated 90 degrees clockwise.</summary>
    Left = 1,

    /// <summary>Back view.</summary>
    Back = 2,

    /// <summary>Front rotated 270 degrees clockwise.</summary>
    Right = 3,
}
