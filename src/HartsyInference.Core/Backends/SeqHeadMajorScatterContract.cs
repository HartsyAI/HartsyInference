using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Validated byte geometry for <see cref="IBackend.ScatterSeqHeadMajor(Tensor, Tensor, int, int)"/>: each
/// head copies <see cref="SliceBytes"/> from <see cref="SourceOffset"/> to <see cref="DestinationOffset"/>.</summary>
internal readonly record struct SeqHeadMajorScatter(int Heads, long InputHeadBytes, long OutputHeadBytes,
    long SeqOffsetBytes, long SliceBytes)
{
    /// <summary>True when there is nothing to copy (no rows, heads or head dim).</summary>
    public bool IsEmpty => Heads == 0 || SliceBytes == 0;

    /// <summary>Byte offset of head <paramref name="head"/>'s first copied row in the input.</summary>
    public long SourceOffset(int head) => head * InputHeadBytes;

    /// <summary>Byte offset of head <paramref name="head"/>'s first written row in the output.</summary>
    public long DestinationOffset(int head) => head * OutputHeadBytes + SeqOffsetBytes;
}

/// <summary>The head-major sequence scatter's argument rules, shared so the host reference and the device copies
/// accept exactly the same calls.</summary>
internal static class SeqHeadMajorScatterContract
{
    internal static SeqHeadMajorScatter Validate(Tensor output, Tensor input, int seqOffset, int rows)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        if (output.DType != input.DType)
            throw new ArgumentException(
                $"ScatterSeqHeadMajor needs matching dtypes; got output {output.DType}, input {input.DType}.", nameof(input));
        if (input.DType.IsQuantized || input.DType.SizeInBytes <= 0)
            throw new NotSupportedException($"ScatterSeqHeadMajor copies byte-addressable dtypes only; got {input.DType}.");
        if (output.Shape.Rank != 4 || input.Shape.Rank != 4 || output.Shape[0] != 1 || input.Shape[0] != 1)
            throw new ArgumentException(
                $"ScatterSeqHeadMajor expects [1, heads, seq, headDim] tensors; got output {output.Shape}, input {input.Shape}.");
        if (output.Shape[1] != input.Shape[1] || output.Shape[3] != input.Shape[3])
            throw new ArgumentException(
                $"ScatterSeqHeadMajor needs matching heads and head dim; got output {output.Shape}, input {input.Shape}.");
        long inputSeq = input.Shape[2], outputSeq = output.Shape[2];
        if (rows < 0 || rows > inputSeq)
            throw new ArgumentOutOfRangeException(nameof(rows), rows, $"rows must be in [0, {inputSeq}].");
        if (seqOffset < 0 || (long)seqOffset + rows > outputSeq)
            throw new ArgumentOutOfRangeException(
                nameof(seqOffset), seqOffset, $"rows [{seqOffset}, {(long)seqOffset + rows}) exceed the output's {outputSeq}.");
        long rowBytes = output.Shape[3] * input.DType.SizeInBytes;
        return new SeqHeadMajorScatter((int)output.Shape[1], inputSeq * rowBytes, outputSeq * rowBytes,
            seqOffset * rowBytes, rows * rowBytes);
    }
}
