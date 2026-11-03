namespace HartsyInference.Core.Tensors;

/// <summary>ComfyUI's <c>asym_w4a8_int8</c> weight format, 4- and 6-bit, with the <c>convrot</c> Hadamard rotation
/// (comfy-kitchen <c>backends/eager/w4a8_int8.py</c> <c>dequantize_w4a8_int8_weight</c> and <c>tensor/w4a8_packed.py</c>
/// @cdab4787a8). This decodes to a dense BF16 <c>[out, in]</c> weight, so a W4A8 layer runs as a W4A16 one; the INT8
/// activation path is not modelled.</summary>
/// <remarks><para>Storage, one row per output feature:</para>
/// <list type="bullet">
///   <item><c>weight</c>: I8 <c>[N, K·bits/8]</c>. A K/2-byte plane of packed codes (even column in the low nibble), then,
///         at 6 bits, a K/4-byte plane of each code's top two bits (column <c>c</c> in byte <c>c/4</c> at bit
///         <c>2·(c%4)</c>). The bit depth is implied by the width.</item>
///   <item><c>weight_s_rel</c>: <c>[N, K/group]</c> per-group relative scale (FP8 E4M3 or F32).</item>
///   <item><c>weight_s_channel</c>: F32 <c>[N]</c> per-row scale.</item>
///   <item><c>weight_codebook</c>: F32 <c>[16]</c> 4-bit levels; absent at 6 bits, where codes are uniform
///         <c>code − 32</c>.</item>
///   <item><c>weight_correction</c>: optional F32 <c>[K/group, N]</c> per-group additive term.</item>
/// </list>
/// <para>Decode: <c>level = codebook[code]</c> (or <c>code − 2^(bits−1)</c>); <c>q = clamp(round(level·s_rel), −127, 127)</c>
/// on the INT8 grid; <c>w = q·s_channel (+ correction)</c>; then each row is un-rotated per ConvRot group, as for
/// <see cref="Int8ConvRotCodec"/>.</para></remarks>
public static class W4A8ConvRotCodec
{
    /// <summary>The comfy_quant format name.</summary>
    public const string Format = "asym_w4a8_int8";

    /// <summary>The bit depth packed in a row of <paramref name="packedWidth"/> bytes holding <paramref name="k"/> codes:
    /// 4 (K/2 bytes) or 6 (K/2 + K/4 bytes).</summary>
    public static int BitsFor(long packedWidth, long k) =>
        packedWidth * 2 == k ? 4
            : packedWidth * 4 == k * 3 ? 6
            : throw new ArgumentException($"A packed row of {packedWidth} bytes holds neither 4-bit nor 6-bit codes for {k} columns.");

    /// <summary>Decodes a W4A8/W6A8 weight to dense BF16 <c>[N, K]</c>.</summary>
    /// <param name="convRotGroupSize">0 for a weight quantized without rotation.</param>
    public static Tensor DequantToBf16(Tensor weight, Tensor sRel, Tensor sChannel, Tensor? codebook, Tensor? correction,
        int groupSize, int convRotGroupSize)
    {
        ArgumentNullException.ThrowIfNull(sChannel);
        using Tensor grid = DecodeToInt8(weight, sRel, codebook, groupSize);
        int n = (int)grid.Shape[0], k = (int)grid.Shape[1];
        int groups = (int)sRel.Shape[1];
        if (sChannel.ElementCount != n)
            throw new ArgumentException($"W4A8 s_channel must hold {n} row scales; got {sChannel.ElementCount}.", nameof(sChannel));
        if (correction is not null && (correction.Shape.Rank != 2 || correction.Shape[0] != groups || correction.Shape[1] != n))
            throw new ArgumentException($"W4A8 correction must be [{groups}, {n}]; got {correction.Shape}.", nameof(correction));
        if (correction is not null && BitsFor(weight.Shape[1], k) == 6)
            throw new ArgumentException("6-bit W4A8 storage is uniform and symmetric: it takes no codebook or correction.");
        if (convRotGroupSize > 0 && (!Int8ConvRotCodec.IsValidGroupSize(convRotGroupSize) || k % convRotGroupSize != 0))
            throw new ArgumentException($"ConvRot group {convRotGroupSize} does not fit {k} columns.");

        float[] chan = Floats(sChannel);
        float[]? corr = correction is null ? null : Floats(correction);
        ReadOnlySpan<sbyte> codes = grid.AsReadOnlySpan<sbyte>();
        using Tensor dense = new Tensor(new TensorShape(n, k), DType.F32);
        Span<float> output = dense.AsSpan<float>();
        for (int r = 0; r < n; r++)
        {
            Span<float> outRow = output.Slice(r * k, k);
            for (int col = 0; col < k; col++)
            {
                float value = codes[r * k + col] * chan[r];
                if (corr is not null)
                    value += corr[(col / groupSize) * n + r];
                outRow[col] = value;
            }
            if (convRotGroupSize > 0)
                Int8ConvRotCodec.ApplyRotationInPlace(outRow, convRotGroupSize);
        }
        return dense.CastTo(DType.BF16);
    }

    /// <summary>Decodes the packed codes to the INT8 grid the format multiplies on, <c>q = clamp(round(level·s_rel), −127, 127)</c>,
    /// as I8 <c>[N, K]</c>, still in the rotated basis. With no correction term, <c>q</c> times <c>s_channel</c> per row is
    /// the weight, so <c>q</c> with <c>s_channel</c> as its row scale is an <c>int8_tensorwise</c> weight (ConvRot included).</summary>
    public static Tensor DecodeToInt8(Tensor weight, Tensor sRel, Tensor? codebook, int groupSize)
    {
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(sRel);
        if (weight.DType != DType.I8 || weight.Shape.Rank != 2)
            throw new ArgumentException($"W4A8 weight must be I8 rank-2; got {weight.DType} {weight.Shape}.", nameof(weight));
        if (sRel.Shape.Rank != 2 || groupSize < 1)
            throw new ArgumentException($"W4A8 s_rel must be [N, K/group] with a positive group; got {sRel.Shape}, group {groupSize}.");
        int n = (int)weight.Shape[0];
        int groups = (int)sRel.Shape[1];
        int k = groups * groupSize;
        int bits = BitsFor(weight.Shape[1], k);
        if (sRel.Shape[0] != n)
            throw new ArgumentException($"W4A8 s_rel must cover {n} rows; got {sRel.Shape}.", nameof(sRel));
        if (bits == 4 && codebook is { ElementCount: not 16 })
            throw new ArgumentException($"A 4-bit codebook holds 16 levels; got {codebook.ElementCount}.", nameof(codebook));
        if (bits == 6 && codebook is not null)
            throw new ArgumentException("6-bit W4A8 storage is uniform and symmetric: it takes no codebook or correction.");

        float[] rel = Floats(sRel);
        float[]? levels = codebook is null ? null : Floats(codebook);
        ReadOnlySpan<byte> packed = weight.AsReadOnlySpan<byte>();
        int width = (int)weight.Shape[1];
        float offset = 1 << (bits - 1);

        Tensor grid = new Tensor(new TensorShape(n, k), DType.I8);
        Span<sbyte> output = grid.AsSpan<sbyte>();
        for (int r = 0; r < n; r++)
        {
            ReadOnlySpan<byte> row = packed.Slice(r * width, width);
            Span<sbyte> outRow = output.Slice(r * k, k);
            for (int col = 0; col < k; col++)
            {
                byte b = row[col >> 1];
                int code = (col & 1) == 0 ? b & 0xF : b >> 4;
                if (bits == 6)
                    code |= ((row[(k >> 1) + (col >> 2)] >> (2 * (col & 3))) & 0x3) << 4;
                float level = levels is not null ? levels[code] : code - offset;
                // round half to even on the float32 product, as torch's round_() does.
                outRow[col] = (sbyte)Math.Clamp(MathF.Round(level * rel[r * groups + col / groupSize], MidpointRounding.ToEven), -127f, 127f);
            }
        }
        return grid;
    }

    /// <summary>A copy of <paramref name="t"/>'s values as F32, converting (FP8, BF16, F16) through a temporary.</summary>
    private static float[] Floats(Tensor t)
    {
        if (t.DType == DType.F32)
            return t.AsReadOnlySpan<float>().ToArray();
        using Tensor f32 = t.CastTo(DType.F32);
        return f32.AsReadOnlySpan<float>().ToArray();
    }
}
