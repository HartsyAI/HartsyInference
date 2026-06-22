using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.StyleTts2;

/// <summary>Per-layer weights for one <see cref="StyleTransformer1d"/> block: the AdaLayerNorm style
/// regressor, the self-attention QKV/out projections, the cross-attention QKV/out projections (reading the
/// PLBERT text context), the feed-forward pair, and the three layer-norm affine params. Any field may be
/// null — a null AdaLN regressor falls back to plain affine layer-norm, and null cross-attention weights skip
/// the cross-attention sub-block.</summary>
public sealed record StyleTransformerLayer
{
    public Tensor? AdaLnW { get; init; }
    public Tensor? AdaLnB { get; init; }

    public Tensor? SelfNormW { get; init; }
    public Tensor? SelfNormB { get; init; }
    public Tensor? SelfQW { get; init; }
    public Tensor? SelfQB { get; init; }
    public Tensor? SelfKW { get; init; }
    public Tensor? SelfKB { get; init; }
    public Tensor? SelfVW { get; init; }
    public Tensor? SelfVB { get; init; }
    public Tensor? SelfOW { get; init; }
    public Tensor? SelfOB { get; init; }

    public Tensor? CrossNormW { get; init; }
    public Tensor? CrossNormB { get; init; }
    public Tensor? CrossQW { get; init; }
    public Tensor? CrossQB { get; init; }
    public Tensor? CrossKW { get; init; }
    public Tensor? CrossKB { get; init; }
    public Tensor? CrossVW { get; init; }
    public Tensor? CrossVB { get; init; }
    public Tensor? CrossOW { get; init; }
    public Tensor? CrossOB { get; init; }

    public Tensor? FfNormW { get; init; }
    public Tensor? FfNormB { get; init; }
    public Tensor? FfW1 { get; init; }
    public Tensor? FfB1 { get; init; }
    public Tensor? FfW2 { get; init; }
    public Tensor? FfB2 { get; init; }
}
