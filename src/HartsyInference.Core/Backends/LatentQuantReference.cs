using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>CPU reference for quantizing rows into a <see cref="LatentSource"/> and for in-place quantize-dequantize.</summary>
public static class LatentQuantReference
{
    /// <summary>Checks the operands of <see cref="QuantizeRows"/>; shared by every backend.</summary>
    public static void ValidateQuantize(in LatentSource dest, Tensor rows, Tensor physicalRows)
    {
        dest.Validate(nameof(dest));
        if (dest.Rows == 0) throw new ArgumentException("QuantizeLatentRows needs a destination with rows.", nameof(dest));
        if (rows.DType != DType.F32) throw new NotSupportedException("QuantizeLatentRows supports F32 input rows only.");
        if (physicalRows.DType != DType.I32) throw new ArgumentException("physicalRows must be I32.", nameof(physicalRows));
        if (rows.ElementCount == 0 || rows.ElementCount % dest.Dim != 0)
            throw new ArgumentException($"rows ({rows.ElementCount} elements) must be a non-empty multiple of Dim {dest.Dim}.");
        if (physicalRows.ElementCount != rows.ElementCount / dest.Dim)
            throw new ArgumentException($"physicalRows must hold {rows.ElementCount / dest.Dim} entries.", nameof(physicalRows));
    }

    /// <summary>Quantizes row <c>i</c> of <paramref name="rows"/> into <c>dest</c> row <c>physicalRows[i]</c>.</summary>
    /// <remarks>Negative destinations are skipped. A destination past the last row throws here and is skipped on
    /// devices that cannot throw, so callers must keep them in range. Later rows overwrite earlier ones.</remarks>
    public static unsafe void QuantizeRows(in LatentSource dest, Tensor rows, Tensor physicalRows)
    {
        ValidateQuantize(dest, rows, physicalRows);
        int dim = dest.Dim, count = (int)physicalRows.ElementCount;
        int group = LatentEncodings.GroupSize(dest.Encoding), groups = group == 0 ? 0 : dim / group;
        bool fp4 = LatentEncodings.IsFp4(dest.Encoding);
        float* src = (float*)rows.DataPointer;
        int* phys = (int*)physicalRows.DataPointer;
        long codeRow = LatentEncodings.CodeBytesPerRow(dest.Encoding, dim);
        byte[] scratch = new byte[Math.Max(group, 1)];
        for (int i = 0; i < count; i++)
        {
            int p = phys[i];
            if (p < 0) continue;
            if (p >= dest.Rows) throw new ArgumentOutOfRangeException(nameof(physicalRows), $"Row {i} targets {p}; dest has {dest.Rows}.");
            if (dest.Encoding == LatentEncoding.F32)
            {
                new ReadOnlySpan<float>(src + (long)i * dim, dim).CopyTo(new Span<float>((float*)dest.Codes!.DataPointer + (long)p * dim, dim));
                continue;
            }
            byte* codes = (byte*)dest.Codes!.DataPointer + p * codeRow;
            byte* scales = (byte*)dest.Scales!.DataPointer + (long)p * groups;
            for (int g = 0; g < groups; g++)
            {
                LatentCodec.QuantizeGroup(dest.Encoding, new ReadOnlySpan<float>(src + (long)i * dim + g * group, group), scratch,
                    out scales[g]);
                for (int e = 0; e < group; e++)
                {
                    int col = g * group + e;
                    if (!fp4) codes[col] = scratch[e];
                    else if ((col & 1) == 0) codes[col >> 1] = scratch[e];
                    else codes[col >> 1] |= (byte)(scratch[e] << 4);
                }
            }
        }
    }

    /// <summary>Checks the operands of <see cref="ActQuantDequantInPlace"/>.</summary>
    public static void ValidateActQuant(Tensor x, LatentEncoding encoding)
    {
        if (x.DType != DType.F32) throw new NotSupportedException("ActQuantDequantInPlace supports F32 only.");
        int group = LatentEncodings.GroupSize(encoding);
        if (group != 0 && (x.ElementCount == 0 || x.Shape[x.Shape.Rank - 1] % group != 0))
            throw new ArgumentException($"The last dimension must be a multiple of {group} for {encoding}.", nameof(x));
    }

    /// <summary>Replaces each value with its quantize-then-dequantize round trip, as the reference <c>act_quant(inplace=True)</c>.</summary>
    public static unsafe void ActQuantDequantInPlace(Tensor x, LatentEncoding encoding)
    {
        ValidateActQuant(x, encoding);
        int group = LatentEncodings.GroupSize(encoding);
        if (group == 0) return;
        float* p = (float*)x.DataPointer;
        byte[] codes = new byte[group];
        long groups = x.ElementCount / group;
        for (long g = 0; g < groups; g++)
        {
            Span<float> span = new Span<float>(p + g * group, group);
            float scale = LatentCodec.QuantizeGroup(encoding, span, codes, out byte _);
            for (int e = 0; e < group; e++) span[e] = LatentCodec.DecodeCode(encoding, codes[e]) * scale;
        }
    }
}
