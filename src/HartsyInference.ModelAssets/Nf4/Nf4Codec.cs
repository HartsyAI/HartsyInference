using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Nf4;

/// <summary>Dequantizer for bitsandbytes <b>NF4</b> (4-bit NormalFloat) weights, as shipped by the <c>bitsandbytes</c> <c>Linear4bit</c> path (Flux.2 Klein nf4, Ideogram 4 nf4, many community quants).
///
/// <para>NF4 is a <b>non-linear</b> 4-bit code: each nibble indexes a fixed 16-entry codebook of normal quantiles (not the linear e2m1 grid used by MXFP4/NVFP4). Reconstruction is <c>x[i] = absmax[i / blockSize] · NF4_LUT[nibble]</c>, where <c>absmax</c> is one positive scale per block of <see cref="DefaultBlockSize"/> consecutive elements. Because the codebook is non-linear, NF4 has <b>no tensor-core dtype</b> — unlike FP4 it cannot feed a native FP4 GEMM, so the engine dequantizes it to F32 at load (the same strategy MXFP4/NVFP4 use). The native FP4 path (<c>BlockScaledGemmExecutor</c>) is e2m1 only.</para>
///
/// <para>Two layers exist on disk: plain quant (an explicit F32 <c>absmax</c>) and bitsandbytes "double quant" where <c>absmax</c> is itself 8-bit quantized against a 256-entry nested codebook plus a scalar offset. <see cref="ReconstructDoubleQuantAbsmax"/> rebuilds the F32 absmax for that case.</para></summary>
public static unsafe class Nf4Codec
{
    /// <summary>Default bitsandbytes NF4 block size (elements per absmax scale).</summary>
    public const int DefaultBlockSize = 64;

    /// <summary>The 16-entry NF4 codebook (bitsandbytes normal-float quantiles, symmetric, includes exact 0).</summary>
    public static readonly float[] Nf4Lut =
    [
        -1.0f,
        -0.6961928009986877f,
        -0.5250730514526367f,
        -0.39491748809814453f,
        -0.28444138169288635f,
        -0.18477343022823334f,
        -0.09105003625154495f,
         0.0f,
         0.07958029955625534f,
         0.16093020141124725f,
         0.24611230194568634f,
         0.33791524171829224f,
         0.44070982933044434f,
         0.5626170039176941f,
         0.7229568362236023f,
         1.0f,
    ];

    /// <summary>Dequantizes a plain (single-quant) NF4 tensor to F32. The packed input holds two nibbles per byte, first element in the high nibble (bitsandbytes order).</summary>
    /// <param name="packed">U8 tensor of <c>ceil(N/2)</c> bytes holding the 4-bit codes.</param>
    /// <param name="absmaxF32">F32 per-block scales, length <c>ceil(N / blockSize)</c>.</param>
    /// <param name="outputShape">Shape of the reconstructed tensor; its element count is <c>N</c>.</param>
    /// <param name="blockSize">Elements per absmax block (default <see cref="DefaultBlockSize"/>).</param>
    public static Tensor Dequantize(Tensor packed, Tensor absmaxF32, TensorShape outputShape, int blockSize = DefaultBlockSize)
    {
        if (packed.DType != DType.U8)
            throw new ArgumentException($"NF4 packed weight must be U8; got {packed.DType}.", nameof(packed));
        if (absmaxF32.DType != DType.F32)
            throw new ArgumentException($"NF4 absmax must be F32; got {absmaxF32.DType}. Use ReconstructDoubleQuantAbsmax for double-quant.", nameof(absmaxF32));
        if (blockSize <= 0)
            throw new ArgumentException("blockSize must be positive.", nameof(blockSize));

        long n = outputShape.ElementCount;
        long expectedBytes = (n + 1) / 2;
        if (packed.Shape.ElementCount < expectedBytes)
            throw new ArgumentException($"NF4 packed tensor has {packed.Shape.ElementCount} bytes; need at least {expectedBytes} for {n} elements.", nameof(packed));

        long expectedBlocks = (n + blockSize - 1) / blockSize;
        if (absmaxF32.Shape.ElementCount < expectedBlocks)
            throw new ArgumentException($"NF4 absmax has {absmaxF32.Shape.ElementCount} blocks; need {expectedBlocks} for {n} elements at blockSize {blockSize}.", nameof(absmaxF32));

        Tensor output = new Tensor(outputShape, DType.F32);
        byte* p = (byte*)packed.DataPointer;
        float* a = (float*)absmaxF32.DataPointer;
        float* o = (float*)output.DataPointer;

        for (long i = 0; i < n; i++)
        {
            byte b = p[i >> 1];
            int nibble = (i & 1) == 0 ? (b >> 4) & 0x0F : b & 0x0F; // high nibble = even element
            o[i] = Nf4Lut[nibble] * a[i / blockSize];
        }

        return output;
    }

    /// <summary>Quantizes F32 weights to plain NF4: one absmax per block, each element rounded to its nearest codebook entry.</summary>
    /// <returns>The packed nibbles (high nibble = even element) and one F32 absmax per block. Dequantize reproduces each element within <see cref="MaxBlockError"/> of its block absmax.</returns>
    public static (byte[] Packed, float[] Absmax) Quantize(ReadOnlySpan<float> values, int blockSize = DefaultBlockSize)
    {
        if (blockSize <= 0)
            throw new ArgumentException("blockSize must be positive.", nameof(blockSize));

        int n = values.Length;
        byte[] packed = new byte[(n + 1) / 2];
        float[] absmax = new float[(n + blockSize - 1) / blockSize];
        for (int b = 0; b < absmax.Length; b++)
        {
            int start = b * blockSize;
            int end = Math.Min(n, start + blockSize);
            float m = 0f;
            for (int i = start; i < end; i++)
                m = MathF.Max(m, MathF.Abs(values[i]));
            absmax[b] = m;

            float inv = m > 0f ? 1f / m : 0f;
            for (int i = start; i < end; i++)
            {
                int code = NearestCode(values[i] * inv);
                packed[i >> 1] |= (byte)((i & 1) == 0 ? code << 4 : code);
            }
        }

        return (packed, absmax);
    }

    /// <summary>Largest reconstruction error for one element, as a fraction of its block absmax: half the widest gap between adjacent codebook entries.</summary>
    public static float MaxBlockError
    {
        get
        {
            float widest = 0f;
            for (int i = 1; i < Nf4Lut.Length; i++)
                widest = MathF.Max(widest, Nf4Lut[i] - Nf4Lut[i - 1]);
            return widest / 2f;
        }
    }

    private static int NearestCode(float normalized)
    {
        int best = 0;
        float bestError = float.MaxValue;
        for (int c = 0; c < Nf4Lut.Length; c++)
        {
            float error = MathF.Abs(normalized - Nf4Lut[c]);
            if (error < bestError)
            {
                bestError = error;
                best = c;
            }
        }
        return best;
    }

    /// <summary>Rebuilds the F32 per-block <c>absmax</c> for a bitsandbytes double-quantized NF4 tensor: <c>absmax[b] = nestedCodebook[ qAbsmax[b] ] · nestedAbsmax[b / nestedBlockSize] + offset</c>. Feed the result to <see cref="Dequantize"/>.</summary>
    /// <param name="qAbsmax">U8 quantized absmax codes, one per primary block.</param>
    /// <param name="nestedAbsmaxF32">F32 scales for the nested blocks.</param>
    /// <param name="nestedCodebookF32">256-entry F32 nested codebook (bitsandbytes <c>nested_quant_map</c>).</param>
    /// <param name="offset">Scalar offset (bitsandbytes <c>offset</c>).</param>
    /// <param name="nestedBlockSize">Primary blocks per nested scale (bitsandbytes default 256).</param>
    public static Tensor ReconstructDoubleQuantAbsmax(
        Tensor qAbsmax, Tensor nestedAbsmaxF32, Tensor nestedCodebookF32, float offset, int nestedBlockSize = 256)
    {
        if (qAbsmax.DType != DType.U8)
            throw new ArgumentException($"NF4 quantized absmax must be U8; got {qAbsmax.DType}.", nameof(qAbsmax));
        if (nestedAbsmaxF32.DType != DType.F32 || nestedCodebookF32.DType != DType.F32)
            throw new ArgumentException("NF4 nested absmax and codebook must be F32.");
        if (nestedCodebookF32.Shape.ElementCount < 256)
            throw new ArgumentException($"NF4 nested codebook must have 256 entries; got {nestedCodebookF32.Shape.ElementCount}.", nameof(nestedCodebookF32));

        long blocks = qAbsmax.Shape.ElementCount;
        Tensor absmax = new Tensor(new TensorShape(blocks), DType.F32);
        byte* q = (byte*)qAbsmax.DataPointer;
        float* nested = (float*)nestedAbsmaxF32.DataPointer;
        float* code = (float*)nestedCodebookF32.DataPointer;
        float* o = (float*)absmax.DataPointer;

        for (long b = 0; b < blocks; b++)
        {
            o[b] = code[q[b]] * nested[b / nestedBlockSize] + offset;
        }

        return absmax;
    }
}
