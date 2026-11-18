namespace HartsyInference.Engine.Requests;

/// <summary>How much room the crop around the object leaves (Idol): the object fills 1 / 1.2 of the frame for <see cref="Object"/> and 1 / 1.7 for <see cref="Photo"/>.</summary>
public enum MeshFraming
{
    /// <summary>A render or a generated picture of one object.</summary>
    Object = 0,

    /// <summary>A photograph.</summary>
    Photo = 1,
}
