namespace SharpInference.Diffusion.Adapters;

/// <summary>Configuration for IP-Adapter (Image Prompt Adapter). Projects CLIP image embeddings into cross-attention space for image-guided generation. Works alongside text conditioning.</summary>
public record IpAdapterConfig
{
    /// <summary>Base model this adapter was trained for.</summary>
    public required IpAdapterBaseModel BaseModel { get; init; }

    /// <summary>CLIP image encoder model used (ViT-H/14 for most IP-Adapters).</summary>
    public required string ClipImageModel { get; init; }

    /// <summary>CLIP image embedding dimension (1024 for ViT-H/14, 768 for ViT-L/14).</summary>
    public int ImageEmbeddingDim { get; init; } = 1024;

    /// <summary>Number of image tokens projected from the CLIP embedding. IP-Adapter uses 4, IP-Adapter Plus uses 16.</summary>
    public int NumImageTokens { get; init; } = 4;

    /// <summary>Cross-attention dimension of the base model.</summary>
    public int CrossAttentionDim { get; init; } = 768;

    /// <summary>Whether this is an IP-Adapter Plus variant (uses patch features instead of CLS token).</summary>
    public bool IsPlus { get; init; }

    /// <summary>Whether this is a face-specific adapter.</summary>
    public bool IsFaceId { get; init; }

    /// <summary>Conditioning scale applied to IP-Adapter outputs.</summary>
    public float Scale { get; init; } = 1.0f;

    /// <summary>SD 1.5 IP-Adapter preset.</summary>
    public static IpAdapterConfig Sd15 => new()
    {
        BaseModel = IpAdapterBaseModel.Sd15,
        ClipImageModel = "ViT-H/14",
        ImageEmbeddingDim = 1024,
        NumImageTokens = 4,
        CrossAttentionDim = 768,
    };

    /// <summary>SD 1.5 IP-Adapter Plus preset (16 tokens, patch features).</summary>
    public static IpAdapterConfig Sd15Plus => new()
    {
        BaseModel = IpAdapterBaseModel.Sd15,
        ClipImageModel = "ViT-H/14",
        ImageEmbeddingDim = 1024,
        NumImageTokens = 16,
        CrossAttentionDim = 768,
        IsPlus = true,
    };

    /// <summary>SDXL IP-Adapter preset.</summary>
    public static IpAdapterConfig Sdxl => new()
    {
        BaseModel = IpAdapterBaseModel.Sdxl,
        ClipImageModel = "ViT-H/14",
        ImageEmbeddingDim = 1024,
        NumImageTokens = 4,
        CrossAttentionDim = 2048,
    };
}

/// <summary>Base model type for IP-Adapter.</summary>
public enum IpAdapterBaseModel
{
    /// <summary>Stable Diffusion 1.5.</summary>
    Sd15,

    /// <summary>SDXL.</summary>
    Sdxl,

    /// <summary>Flux.</summary>
    Flux,
}
