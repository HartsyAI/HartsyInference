using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Dequantizes whole rows of a <see cref="LatentSource"/> to F32 for the CPU references.</summary>
internal static class LatentRowReader
{
    /// <summary>Writes row <paramref name="row"/> of <paramref name="source"/> into <paramref name="destination"/>.</summary>
    internal static unsafe void ReadRow(in LatentSource source, int row, Span<float> destination)
    {
        int dim = source.Dim;
        if (source.Encoding == LatentEncoding.F32)
        {
            new ReadOnlySpan<float>((float*)source.Codes!.DataPointer + (long)row * dim, dim).CopyTo(destination);
            return;
        }
        long codeRow = LatentEncodings.CodeBytesPerRow(source.Encoding, dim);
        int scaleRow = LatentEncodings.ScaleBytesPerRow(source.Encoding, dim);
        ReadOnlySpan<byte> codes = new ReadOnlySpan<byte>((byte*)source.Codes!.DataPointer + row * codeRow, (int)codeRow);
        ReadOnlySpan<byte> scales = new ReadOnlySpan<byte>((byte*)source.Scales!.DataPointer + (long)row * scaleRow, scaleRow);
        for (int c = 0; c < dim; c++) destination[c] = LatentCodec.DecodeElement(source.Encoding, codes, scales, c);
    }
}
