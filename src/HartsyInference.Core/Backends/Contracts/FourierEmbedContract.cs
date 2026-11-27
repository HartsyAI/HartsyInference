using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.FourierEmbed"/>: F32 <c>[count, InputDims]</c> coordinates and an F32
/// output of <c>count · Width</c> elements sharing no storage; at least one input dimension and at most 30 bands (the
/// frequency <c>2^b</c> stays an exact integer). Returns the point count.</summary>
public static class FourierEmbedContract
{
    public const int MaxBands = 30;

    public static long Validate(Tensor output, Tensor coords, in FourierEmbedSpec spec)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(coords);
        if (output.DType != DType.F32 || coords.DType != DType.F32)
            throw new NotSupportedException($"FourierEmbed takes F32 tensors; got output {output.DType}, coords {coords.DType}.");
        if (spec.InputDims < 1)
            throw new ArgumentOutOfRangeException(nameof(spec), spec.InputDims, "FourierEmbed needs at least one input dimension.");
        if (spec.Bands < 0 || spec.Bands > MaxBands)
            throw new ArgumentOutOfRangeException(nameof(spec), spec.Bands, $"FourierEmbed takes 0 to {MaxBands} bands.");
        if (coords.ElementCount % spec.InputDims != 0)
            throw new ArgumentException($"FourierEmbed coordinates {coords.Shape} are not rows of {spec.InputDims}.", nameof(coords));
        long count = coords.ElementCount / spec.InputDims;
        if (output.ElementCount != count * spec.Width)
            throw new ArgumentException($"FourierEmbed output {output.Shape} must hold {count} rows of {spec.Width}; it has {output.ElementCount} elements.", nameof(output));
        if (output.HasOverlappingHostStorageWithoutSync(coords))
            throw new ArgumentException("FourierEmbed output storage must not overlap the coordinates.", nameof(output));
        return count;
    }
}
