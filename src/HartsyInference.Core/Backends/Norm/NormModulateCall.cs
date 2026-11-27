using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Norm;

/// <summary>One <see cref="IBackend.NormModulate"/> invocation: each row of <see cref="Input"/> normalized as <see cref="Kind"/> says,
/// scaled by <see cref="Weight"/>, offset by <see cref="Bias"/>, then modulated as <see cref="Modulation"/> says, into
/// <see cref="Output"/>: <c>y = (norm(x) · w + b) · scale + shift</c> with every absent term dropped. The activations are F32, F16
/// or BF16, independently for input and output; the vectors are F32.</summary>
/// <param name="Output">The result, the same element count as <paramref name="Input"/>.</param>
/// <param name="Input">The activation.</param>
/// <param name="Kind">The normalization.</param>
/// <param name="Axis">Which axis forms a row.</param>
/// <param name="Eps">The normalization epsilon; unused by <see cref="NormKind.None"/>.</param>
/// <param name="Weight">The norm's learned scale, <c>[width]</c>, or null.</param>
/// <param name="Bias">The norm's learned offset, <c>[width]</c>, or null.</param>
/// <param name="Modulation">The adaLN scale and shift, or none.</param>
public readonly record struct NormModulateCall(Tensor Output, Tensor Input, NormKind Kind, NormAxis Axis, float Eps, Tensor? Weight = null, Tensor? Bias = null, ModulationSpec Modulation = default)
{
    /// <summary>Rows along the last axis.</summary>
    public static NormModulateCall Rows(Tensor output, Tensor input, NormKind kind, float eps) => new(output, input, kind, NormAxis.LastDim, eps);

    /// <summary>Rows along the channel axis of <c>[N, C, ...]</c>.</summary>
    public static NormModulateCall Channel(Tensor output, Tensor input, NormKind kind, float eps) => new(output, input, kind, NormAxis.Channel, eps);

    /// <summary>The same call with the norm's learned affine.</summary>
    public NormModulateCall WithAffine(Tensor? weight, Tensor? bias) => this with { Weight = weight, Bias = bias };

    /// <summary>The same call with a modulation.</summary>
    public NormModulateCall WithModulation(in ModulationSpec modulation) => this with { Modulation = modulation };

    /// <summary>The width of a row: the last extent, or the channel count.</summary>
    public int Width => (int)(Axis == NormAxis.LastDim ? Input.Shape[Input.Shape.Rank - 1] : Input.Shape[1]);
}
