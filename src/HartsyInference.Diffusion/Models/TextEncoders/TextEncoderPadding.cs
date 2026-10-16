namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>How a single right-padded prompt is masked inside a causal text encoder, matching HuggingFace
/// transformers with an attention mask: no token attends to a padding key, and padding rows take the rotary position
/// <c>cumsum(mask) − 1</c> masked to <paramref name="PaddedPosition"/>. The padding rows' outputs are still produced,
/// for consumers (FLUX.3) whose transformer attends to them.</summary>
/// <param name="ValidLength">Real tokens before the padding; at least 1.</param>
/// <param name="PaddedPosition">Rotary position every padding row uses (transformers 4.57 uses 1).</param>
public readonly record struct TextEncoderPadding(int ValidLength, int PaddedPosition)
{
    /// <summary>The transformers ≥ 4.57 convention.</summary>
    public static TextEncoderPadding Transformers(int validLength) => new(validLength, 1);

    /// <summary>Rotary position of row <paramref name="row"/>.</summary>
    public int PositionOf(int row) => row < ValidLength ? row : PaddedPosition;
}
