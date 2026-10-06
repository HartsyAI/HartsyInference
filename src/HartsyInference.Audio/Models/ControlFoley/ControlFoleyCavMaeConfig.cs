namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Geometry of the CAV-MAE-ST visual branch (official <c>CAVMAEST(audio_length=208, norm_pix_loss=False,
/// modality_specific_depth=11, tr_pos=False)</c>).</summary>
public sealed record ControlFoleyCavMaeConfig
{
    /// <summary>The released ViT-B/16 checkpoint: 11 visual-specific blocks plus one unified block, 224-pixel frames.</summary>
    public static ControlFoleyCavMaeConfig Released { get; } = new()
    {
        EmbedDim = 768, Heads = 12, PatchSize = 16, InputSize = 224, ModalitySpecificDepth = 11,
    };

    /// <summary>Token width.</summary>
    public required int EmbedDim { get; init; }

    /// <summary>Attention heads.</summary>
    public required int Heads { get; init; }

    /// <summary>Patch edge in pixels.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Frame edge in pixels.</summary>
    public required int InputSize { get; init; }

    /// <summary>Blocks specific to one modality; the remaining <see cref="TotalDepth"/> minus this are shared.</summary>
    public required int ModalitySpecificDepth { get; init; }

    /// <summary>Total transformer depth (fixed at 12 in the official model).</summary>
    public int TotalDepth { get; init; } = 12;

    /// <summary>MLP expansion ratio.</summary>
    public int MlpRatio { get; init; } = 4;

    /// <summary>LayerNorm epsilon.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Patch grid edge.</summary>
    public int GridSize => InputSize / PatchSize;

    /// <summary>Tokens per frame.</summary>
    public int Tokens => GridSize * GridSize;
}
