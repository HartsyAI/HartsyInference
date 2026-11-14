namespace HartsyInference.Engine.Requests;

/// <summary>A named bundle of a model's quality settings (<see cref="MeshRequest.Steps"/>, <see cref="MeshRequest.GridResolution"/>, <see cref="MeshRequest.TextureSize"/>); a field the request sets itself always wins.</summary>
public enum MeshQuality
{
    /// <summary>The model's own defaults, as today.</summary>
    Default = 0,

    /// <summary>The best result that is known to fit a 12 GB card in under ten minutes. A model without a preset rejects it.</summary>
    High = 1,

    /// <summary>The best result the model can give whatever it costs in time; it may not fit a 12 GB card, and for a model with no finer setting it equals <see cref="High"/>. A model without a preset rejects it.</summary>
    Max = 2,
}
