using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Locks <see cref="CudaBackend.LinearWeightRows"/> to plain <c>Linear</c> against a MATERIALIZED slice of the
/// same weight. The row-range path exists to avoid materializing that slice (a slice is a separate GPU cache identity,
/// so it would upload a second copy of an already-resident weight), and it reaches the GEMM by offsetting a pointer —
/// into the weight itself, or into the full-size cached dtype cast that sibling ranges share. Both offsets are in
/// gemm-dtype units, which is exactly the sort of thing that is silently wrong only on the dtype the real model uses:
/// these cases run F32, BF16 (cast path) and fp8 (the MiniMax-H3 checkpoint's own weight dtype), preloaded and not,
/// because preloading is what switches the cast on.</summary>
[Collection("CudaSerial")]
public sealed unsafe class LinearWeightRowsTests
{
    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor RandomF32(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    private static Tensor RandomBf16(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.BF16);
        ushort* p = (ushort*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++)
            p[i] = (ushort)(BitConverter.SingleToUInt32Bits((float)(rng.NextDouble() * 2.0 - 1.0)) >> 16);
        return t;
    }

    /// <summary>Raw E4M3 bytes. Exponent is capped well below the NaN/Inf encodings so every byte is a finite value
    /// and the comparison can stay bit-exact.</summary>
    private static Tensor RandomFp8(TensorShape shape, int seed, float scale)
    {
        Tensor t = new Tensor(shape, DType.F8E4M3) { Fp8ScaleFactor = scale };
        byte* p = (byte*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++)
        {
            int sign = rng.Next(2) << 7;
            int exp = rng.Next(4, 10) << 3;      // biased exponent, never 0x0F (NaN) after shift
            int mant = rng.Next(8);
            p[i] = (byte)(sign | exp | mant);
        }
        return t;
    }

    private static void AssertBitExact(Tensor expected, Tensor actual, string label)
    {
        Assert.Equal(expected.ElementCount, actual.ElementCount);
        float* a = (float*)expected.DataPointer, b = (float*)actual.DataPointer;
        long mismatches = 0, firstBad = -1;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
            {
                mismatches++;
                if (firstBad < 0) firstBad = i;
            }
        }
        Assert.True(mismatches == 0,
            $"{label}: {mismatches}/{expected.ElementCount} elements differ; first at {firstBad} "
            + (firstBad >= 0 ? $"(materialized-slice {a[firstBad]} vs row-range {b[firstBad]})" : ""));
    }

    /// <param name="dtype">weight dtype under test</param>
    /// <param name="preload">preloading is what makes the dtype-cast cache eligible, so it changes which pointer the
    /// row offset is applied to</param>
    /// <param name="withBias">bias slices by element, not by row — a separate offset with its own failure mode</param>
    [Theory]
    [InlineData("F32", false, false)]
    [InlineData("F32", true, true)]
    [InlineData("BF16", true, false)]
    [InlineData("BF16", true, true)]
    [InlineData("FP8", true, false)]
    [InlineData("FP8", true, true)]
    public void RowRangeMatchesMaterializedSlice(string dtype, bool preload, bool withBias)
    {
        // Shaped like one MiniMax-H3 chunked-attention call, scaled down: a packed [q|k|v] weight read in two parts.
        const int inner = 128, hidden = 96, rows = 4;
        const int m = 64;
        int outDim = inner * 3;

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor weight = dtype switch
        {
            "F32" => RandomF32(new TensorShape(outDim, hidden), 11),
            "BF16" => RandomBf16(new TensorShape(outDim, hidden), 11),
            _ => RandomFp8(new TensorShape(outDim, hidden), 11, 0.35f),
        };
        using Tensor input = RandomF32(new TensorShape(m, hidden), 22);
        using Tensor? bias = withBias ? RandomF32(new TensorShape(outDim), 33) : null;

        if (preload)
        {
            cuda.PreloadWeights(new[] { weight });
        }

        // The two ranges the split actually uses: q = rows [0, inner), k+v = rows [inner, 3*inner).
        foreach ((int off, int count, string name) in new[] { (0, inner, "q"), (inner, inner * 2, "kv") })
        {
            using Tensor viaRowRange = new Tensor(new TensorShape(m, count), DType.F32);
            cuda.LinearWeightRows(viaRowRange, input, weight, bias, off, count);
            _ = viaRowRange.DataPointer;   // force the D2H sync before the next call rebinds anything

            // Independent reference: materialize the slice as its own tensor and run ordinary Linear.
            using Tensor slice = new Tensor(new TensorShape(count, hidden), weight.DType)
            {
                Fp8ScaleFactor = weight.Fp8ScaleFactor,
            };
            cuda.SliceRowsGeneric(slice, weight, off);
            using Tensor? biasSlice = bias is null ? null : new Tensor(new TensorShape(count), DType.F32);
            if (bias is not null)
            {
                Buffer.MemoryCopy((float*)bias.DataPointer + off, biasSlice!.DataPointer,
                    count * sizeof(float), count * sizeof(float));
            }
            using Tensor viaSlice = new Tensor(new TensorShape(m, count), DType.F32);
            cuda.Linear(viaSlice, input, slice, biasSlice);
            _ = viaSlice.DataPointer;

            AssertBitExact(viaSlice, viaRowRange, $"{dtype} preload={preload} bias={withBias} range={name}");
        }

        _ = rows;
    }
}
