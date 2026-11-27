using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>What a UNet pass hands every 2.5D transformer block: the pass mode, the reference features written by the reference stream and read
/// by the generating stream, the per-material text tokens, the DINO tokens and the rotation tables.</summary>
public sealed class HunyuanPaintBlockState
{
    /// <summary>True in the reference stream: blocks record <c>norm1(hidden)</c> into <see cref="Embeds"/>.</summary>
    public bool Write { get; init; }

    /// <summary>Reference features per transformer-block name, <c>[1, refViews * tokens, channels]</c>; the reference pass fills it, the owner disposes it.</summary>
    public Dictionary<string, Tensor> Embeds { get; init; } = [];

    /// <summary>Scale of the reference attention output: 0 skips it (the unconditional branch), 1 applies it.</summary>
    public float ReferenceScale { get; init; } = 1f;

    /// <summary>Scale of the multiview attention output.</summary>
    public float MultiviewScale { get; init; } = 1f;

    /// <summary>Text tokens <c>[1, tokens, dim]</c> per material (the learned prompts).</summary>
    public required Tensor[] Text { get; init; }

    /// <summary>Projected DINO tokens <c>[1, tokens, dim]</c>, or null.</summary>
    public Tensor? Dino { get; init; }

    /// <summary>Rotation tables keyed by the multiview sequence length (views times latent tokens at a level), or null.</summary>
    public IReadOnlyDictionary<int, (Tensor Cos, Tensor Sin)>? Rotation { get; init; }
}
