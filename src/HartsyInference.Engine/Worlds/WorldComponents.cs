namespace HartsyInference.Engine.Worlds;

/// <summary>The checkpoint files one world model loads, resolved from its spec before any weight is read. A component the
/// architecture does not take is null.</summary>
public sealed record WorldComponents
{
    /// <summary>The loader and fixed config the files are built with. The tokenizer and config follow from it.</summary>
    public required WorldArchitecture Architecture { get; init; }

    /// <summary>The backbone checkpoint: the DiT, or DIAMOND's denoiser.</summary>
    public required string Backbone { get; init; }

    /// <summary>The VAE checkpoint (aux <c>vae-path</c>).</summary>
    public string? Vae { get; init; }

    /// <summary>The text-encoder checkpoint: Hunyuan-GameCraft's Llava-Llama-3-8B (aux <c>llava-path</c>) or SANA-WM's
    /// Gemma-2-2B (aux <c>text-encoder-path</c>).</summary>
    public string? TextEncoder { get; init; }

    /// <summary>The text encoder's tokenizer model (SANA-WM's Gemma <c>tokenizer.model</c>, aux <c>tokenizer-path</c>).</summary>
    public string? Tokenizer { get; init; }

    /// <summary>The CLIP-L text-encoder checkpoint (aux <c>clip-path</c>).</summary>
    public string? ClipEncoder { get; init; }

    /// <summary>Where the model's optional refiner stage lives; null for a model without one.</summary>
    public WorldRefinerFiles? Refiner { get; init; }
}

/// <summary>The files of a world model's refiner stage. They are named when the model is resolved but not required to exist
/// until a run asks for the refiner, so a stage-1-only install resolves without them.</summary>
public sealed record WorldRefinerFiles
{
    /// <summary>The refiner transformer checkpoint (aux <c>refiner-path</c>).</summary>
    public required string Transformer { get; init; }

    /// <summary>The refiner's text connectors checkpoint (aux <c>refiner-connectors-path</c>).</summary>
    public required string Connectors { get; init; }

    /// <summary>The refiner's text encoder: one safetensors file, or a folder of <c>model-*-of-*.safetensors</c> shards
    /// (aux <c>refiner-text-encoder-path</c>). Defaults to the engine's FP8-scaled Gemma-3-12B side model, the file that
    /// fits a 24 GB card.</summary>
    public required string TextEncoder { get; init; }

    /// <summary>The refiner text encoder's tokenizer model (aux <c>refiner-tokenizer-path</c>); null finds the Gemma-3
    /// <c>tokenizer.model</c> beside the text encoder when the refiner opens.</summary>
    public string? Tokenizer { get; init; }
}
