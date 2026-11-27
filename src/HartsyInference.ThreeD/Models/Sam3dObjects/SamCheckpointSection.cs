namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>Which model's tensors to take from a SAM 3D checkpoint file.</summary>
public enum SamCheckpointSection
{
    /// <summary>The flow model, stored under <c>_base_models.generator.</c> in the generator checkpoints.</summary>
    Generator,

    /// <summary>The condition embedder (DINO, pointmap embedder and fuser), stored under <c>_base_models.condition_embedder.</c> in the same files.</summary>
    ConditionEmbedder,

    /// <summary>A file whose root is the state dict itself, as for the structure decoder and the two SLAT decoders.</summary>
    Root,
}
