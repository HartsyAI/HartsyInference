using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>How a lower-rank spatial or packed mask is broadcast over a latent tensor.</summary>
public enum MaskBroadcastLayout
{
    /// <summary><c>target/source/noise=[B,C,H,W]</c>, <c>mask=[B,1,H,W]</c>; mask broadcasts over C.</summary>
    DenseNchwBroadcast,

    /// <summary><c>target/source/noise=[B,S,F]</c>, <c>mask=[B,S,P]</c>, <c>F=C·P</c>; feature <c>f=c·P+p</c> uses mask entry <c>p</c>.</summary>
    PackedChannelOuter,

    /// <summary><c>target/source/noise=[B,S,F]</c>, <c>mask=[B,S,P]</c>, <c>F=P·C</c>; feature <c>f=p·C+c</c> uses mask entry <c>p</c>.</summary>
    PackedChannelInner,
}

/// <summary>Validated launch geometry for a mask-broadcast affine mix.</summary>
internal readonly record struct MaskedMixGeometry(MaskBroadcastLayout Layout, long ElementCount, long Batch,
    long Channels, long Spatial, long Tokens, long FeatureDimension, long PatchArea);

/// <summary>One source of truth for host and accelerator mix contracts.</summary>
internal static class MixContract
{
    internal static long ValidateAffineMix(
        Tensor output, Tensor x, Tensor y, float xScale, float yScale)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        ValidateF32("AffineMix", output, x, y);

        long count = ValidateGeometry(output, nameof(output), "AffineMix");
        ValidateGeometry(x, nameof(x), "AffineMix");
        ValidateGeometry(y, nameof(y), "AffineMix");
        if (output.Shape != x.Shape || output.Shape != y.Shape)
            throw new ArgumentException(
                $"AffineMix requires identical shapes; got output={output.Shape}, x={x.Shape}, y={y.Shape}.");
        RejectTargetAlias("AffineMix", output, x, nameof(output));
        RejectTargetAlias("AffineMix", output, y, nameof(output));
        ValidateFinite(xScale, nameof(xScale));
        ValidateFinite(yScale, nameof(yScale));
        return count;
    }

    internal static MaskedMixGeometry ValidateMaskedAffineMix(
        Tensor target,
        Tensor source,
        Tensor? noise,
        Tensor mask,
        float sourceScale,
        float noiseScale,
        MaskBroadcastLayout layout)
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(mask);
        if (!Enum.IsDefined(layout))
            throw new ArgumentOutOfRangeException(nameof(layout), layout, "Unknown mask broadcast layout.");

        ValidateFinite(sourceScale, nameof(sourceScale));
        ValidateFinite(noiseScale, nameof(noiseScale));
        if ((noiseScale == 0f) != (noise is null))
            throw new ArgumentException(
                "MaskedAffineMixInPlace requires noise=null exactly when noiseScale is zero.", nameof(noise));

        if (noise is null)
            ValidateF32("MaskedAffineMixInPlace", target, source, mask);
        else
            ValidateF32("MaskedAffineMixInPlace", target, source, noise, mask);

        long count = ValidateGeometry(target, nameof(target), "MaskedAffineMixInPlace");
        ValidateGeometry(source, nameof(source), "MaskedAffineMixInPlace");
        ValidateGeometry(mask, nameof(mask), "MaskedAffineMixInPlace");
        if (noise is not null)
            ValidateGeometry(noise, nameof(noise), "MaskedAffineMixInPlace");

        if (source.Shape != target.Shape)
            throw new ArgumentException(
                $"MaskedAffineMixInPlace source shape {source.Shape} must match target shape {target.Shape}.",
                nameof(source));
        if (noise is not null && noise.Shape != target.Shape)
            throw new ArgumentException(
                $"MaskedAffineMixInPlace noise shape {noise.Shape} must match target shape {target.Shape}.",
                nameof(noise));

        RejectTargetAlias("MaskedAffineMixInPlace", target, source, nameof(target));
        if (noise is not null)
            RejectTargetAlias("MaskedAffineMixInPlace", target, noise, nameof(target));
        RejectTargetAlias("MaskedAffineMixInPlace", target, mask, nameof(target));

        return layout switch
        {
            MaskBroadcastLayout.DenseNchwBroadcast =>
                ValidateDense(target, mask, count),
            MaskBroadcastLayout.PackedChannelOuter or MaskBroadcastLayout.PackedChannelInner =>
                ValidatePacked(target, mask, count, layout),
            _ => throw new InvalidOperationException("Validated mask layout was not dispatchable."),
        };
    }

    private static MaskedMixGeometry ValidateDense(Tensor target, Tensor mask, long count)
    {
        if (target.Shape.Rank != 4)
            throw new ArgumentException(
                $"DenseNchwBroadcast target must be rank-4 [B,C,H,W]; got {target.Shape}.", nameof(target));
        if (mask.Shape.Rank != 4)
            throw new ArgumentException(
                $"DenseNchwBroadcast mask must be rank-4 [B,1,H,W]; got {mask.Shape}.", nameof(mask));

        long batch = target.Shape[0];
        long channels = target.Shape[1];
        long height = target.Shape[2];
        long width = target.Shape[3];
        TensorShape expectedMask = new(batch, 1, height, width);
        if (mask.Shape != expectedMask)
            throw new ArgumentException(
                $"DenseNchwBroadcast mask shape must be {expectedMask}; got {mask.Shape}.", nameof(mask));

        long spatial = CheckedMultiply(height, width, nameof(target), "DenseNchwBroadcast spatial geometry");
        return new MaskedMixGeometry(
            MaskBroadcastLayout.DenseNchwBroadcast, count, batch, channels, spatial, 0, 0, 0);
    }

    private static MaskedMixGeometry ValidatePacked(
        Tensor target, Tensor mask, long count, MaskBroadcastLayout layout)
    {
        if (target.Shape.Rank != 3)
            throw new ArgumentException(
                $"{layout} target must be rank-3 [B,S,F]; got {target.Shape}.", nameof(target));
        if (mask.Shape.Rank != 3)
            throw new ArgumentException(
                $"{layout} mask must be rank-3 [B,S,P]; got {mask.Shape}.", nameof(mask));

        long batch = target.Shape[0];
        long tokens = target.Shape[1];
        long featureDimension = target.Shape[2];
        long patchArea = mask.Shape[2];
        if (mask.Shape[0] != batch || mask.Shape[1] != tokens)
            throw new ArgumentException(
                $"{layout} mask must have leading shape [{batch},{tokens}]; got {mask.Shape}.", nameof(mask));
        if (featureDimension % patchArea != 0)
            throw new ArgumentException(
                $"{layout} feature dimension F={featureDimension} must be divisible by inferred patch area P={patchArea}.",
                nameof(target));

        return new MaskedMixGeometry(
            layout, count, batch, 0, 0, CheckedMultiply(batch, tokens, nameof(target), "Packed token geometry"),
            featureDimension, patchArea);
    }

    private static void RejectTargetAlias(string operation, Tensor target, Tensor input, string parameterName)
    {
        if (target.HasOverlappingHostStorageWithoutSync(input))
            throw new ArgumentException(
                $"{operation} target/output storage must not overlap any input storage.", parameterName);
    }

    private static long ValidateGeometry(Tensor tensor, string parameterName, string operation)
    {
        long count = 1;
        try
        {
            for (int axis = 0; axis < tensor.Shape.Rank; axis++)
            {
                long dimension = tensor.Shape[axis];
                if (dimension <= 0)
                    throw new ArgumentOutOfRangeException(
                        parameterName, dimension, $"{operation} axis {axis} must be positive.");
                count = checked(count * dimension);
            }
        }
        catch (OverflowException)
        {
            throw new ArgumentOutOfRangeException(
                parameterName, $"{operation} geometry exceeds signed 64-bit tensor indexing.");
        }

        if (count <= 0 || tensor.ElementCount != count)
            throw new ArgumentException(
                $"{operation} requires a nonempty tensor with valid contiguous geometry; got {tensor.Shape}.",
                parameterName);
        if (count > long.MaxValue / sizeof(float))
            throw new ArgumentOutOfRangeException(
                parameterName, $"{operation} byte span exceeds signed 64-bit addressable storage.");
        return count;
    }

    private static long CheckedMultiply(long left, long right, string parameterName, string description)
    {
        try
        {
            return checked(left * right);
        }
        catch (OverflowException)
        {
            throw new ArgumentOutOfRangeException(parameterName, $"{description} exceeds signed 64-bit indexing.");
        }
    }

    private static void ValidateF32(string operation, params Tensor[] tensors)
    {
        foreach (Tensor tensor in tensors)
        {
            if (tensor.DType != DType.F32)
                throw new NotSupportedException($"{operation} supports F32 only; got {tensor.DType}.");
        }
    }

    private static void ValidateFinite(float value, string parameterName)
    {
        if (!float.IsFinite(value))
            throw new ArgumentOutOfRangeException(parameterName, value, "Mix coefficient must be finite.");
    }
}
