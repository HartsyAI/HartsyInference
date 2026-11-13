namespace HartsyInference.Engine.Planning;

/// <summary>Why the engine cannot fetch a model file itself.</summary>
public enum MeshAssetBlockerKind
{
    /// <summary>The repository is gated: the user accepts its terms on Hugging Face and the engine needs an <c>HF_TOKEN</c> with read access.</summary>
    Gated,

    /// <summary>The engine has no way to fetch the file (not on Hugging Face and no URL in the catalog): the user places it in the folder named. No model reports it today: Pixal3D's NAF release asset is fetched by URL now.</summary>
    ManualPlacement,

    /// <summary>The model has no catalog entry with files: the user supplies the checkpoint folder as the model path.</summary>
    NoCatalogEntry,

    /// <summary>The fetch failed: the connection dropped, the server errored, or the file did not match its pinned hash. Trying again may work.</summary>
    FetchFailed,
}
