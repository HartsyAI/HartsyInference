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
}
