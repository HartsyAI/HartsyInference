namespace HartsyInference.Engine.Planning;

/// <summary>Why the engine cannot fetch a model file itself.</summary>
public enum MeshAssetBlockerKind
{
    /// <summary>The repository is gated: the user accepts its terms on Hugging Face and the engine needs an <c>HF_TOKEN</c> with read access.</summary>
    Gated,

    /// <summary>The file is not on Hugging Face (a release asset, a repack the user makes): the user places it in the folder named.</summary>
    ManualPlacement,

    /// <summary>The model has no catalog entry with files: the user supplies the checkpoint folder as the model path.</summary>
    NoCatalogEntry,
}
