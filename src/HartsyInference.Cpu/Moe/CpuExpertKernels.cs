using System.Buffers;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Moe;

/// <summary>
/// Quantized expert kernels for the CPU: one MoE expert <c>down( act(clampGate(Gate·x)) * clampUp(Up·x) )</c> computed
/// directly on packed Q8_0, Q4_K, Q5_K or Q6_K bytes, for 1 to 8 token rows at once. Each projection may use its own format, as
/// a GGUF K-quant mix does (Q4_K_M keeps some down projections in Q6_K).
///
/// <para><b>Layout.</b> Gate and up are <c>[intermediate, hidden]</c> row-major, down is <c>[hidden, intermediate]</c>;
/// each row is <c>dtype.ComputeByteCount(cols)</c> bytes, exactly as <c>ExpertPackWriter</c> writes them and as a GGUF stores
/// one expert's slice of its stacked expert tensor.</para>
///
/// <para><b>Integer path.</b> Each token's input row is quantized once to int8 in 32-element blocks, each with a float
/// scale and the plain sum of its codes. A weight row's dot product is then an exact int32 sum per block, folded into
/// float in a fixed order. The integer sums do not depend on order, so the AVX2 and scalar paths produce identical
/// bits; the float fold is one shared scalar code path.</para>
///
/// <para><b>Allocation.</b> Scratch comes from <see cref="ArrayPool{T}"/>; a steady-state call allocates nothing.</para>
/// </summary>
public static unsafe partial class CpuExpertKernels
{
    /// <summary>Elements per activation quantization block: the Q8_0 block, and one sub-block of a Q4_K super-block.</summary>
    public const int QuantBlock = 32;

    /// <summary>Most token rows a single call accepts.</summary>
    public const int MaxRows = 8;

    /// <summary>Whether the kernels read <paramref name="dtype"/>.</summary>
    public static bool Supports(DType dtype) => dtype == DType.Q8_0 || dtype == DType.Q4_K || dtype == DType.Q5_K || dtype == DType.Q6_K;

    /// <summary>Runs the expert with every projection in <paramref name="dtype"/>, using AVX2 when the CPU supports it.</summary>
    /// <param name="program">Activation and clamp bounds.</param>
    /// <param name="dtype">Packed weight dtype: Q8_0, Q4_K, Q5_K or Q6_K.</param>
    /// <param name="hidden">Model width H; a multiple of 256 for the K-quants and of 32 for Q8_0.</param>
    /// <param name="intermediate">Expert inner width I; same divisibility as <paramref name="hidden"/>.</param>
    /// <param name="gate">Packed gate matrix, <c>[I, H]</c>.</param>
    /// <param name="up">Packed up matrix, <c>[I, H]</c>.</param>
    /// <param name="down">Packed down matrix, <c>[H, I]</c>.</param>
    /// <param name="x"><c>rows × H</c> inputs, row-major.</param>
    /// <param name="rows">Token rows, 1 to <see cref="MaxRows"/>.</param>
    /// <param name="y"><c>rows × H</c> outputs, overwritten.</param>
    /// <exception cref="ArgumentException">A shape, buffer length or divisibility rule is violated.</exception>
    /// <exception cref="NotSupportedException">The dtype is not one <see cref="Supports"/> accepts.</exception>
    public static void Apply(ExpertProgram program, DType dtype, int hidden, int intermediate, ReadOnlySpan<byte> gate,
        ReadOnlySpan<byte> up, ReadOnlySpan<byte> down, ReadOnlySpan<float> x, int rows, Span<float> y) =>
        Run(program, new ExpertDTypes(dtype, dtype, dtype), hidden, intermediate, gate, up, down, x, rows, y, SimdDispatch.IsAvx2Supported);

    /// <summary>Runs the expert with each projection in its own format, using AVX2 when the CPU supports it.</summary>
    /// <inheritdoc cref="Apply(ExpertProgram, DType, int, int, ReadOnlySpan{byte}, ReadOnlySpan{byte}, ReadOnlySpan{byte}, ReadOnlySpan{float}, int, Span{float})"/>
    public static void Apply(ExpertProgram program, ExpertDTypes dtypes, int hidden, int intermediate, ReadOnlySpan<byte> gate,
        ReadOnlySpan<byte> up, ReadOnlySpan<byte> down, ReadOnlySpan<float> x, int rows, Span<float> y) =>
        Run(program, dtypes, hidden, intermediate, gate, up, down, x, rows, y, SimdDispatch.IsAvx2Supported);

    /// <summary>The per-projection expert with its rows spread across CPU threads (<see cref="CpuParallel"/>): each gate/up row and
    /// each down row is an independent dot product, so one expert serving one token can use every core. Bit-identical to
    /// <see cref="Apply(ExpertProgram, ExpertDTypes, int, int, ReadOnlySpan{byte}, ReadOnlySpan{byte}, ReadOnlySpan{byte}, ReadOnlySpan{float}, int, Span{float})"/>:
    /// every output element is the same per-row computation, only on another thread. Allocates the fan-out's closures when it fans
    /// out; the serial overloads remain allocation-free.</summary>
    /// <inheritdoc cref="Apply(ExpertProgram, ExpertDTypes, int, int, ReadOnlySpan{byte}, ReadOnlySpan{byte}, ReadOnlySpan{byte}, ReadOnlySpan{float}, int, Span{float})"/>
    public static void ApplyParallel(ExpertProgram program, ExpertDTypes dtypes, int hidden, int intermediate, ReadOnlySpan<byte> gate,
        ReadOnlySpan<byte> up, ReadOnlySpan<byte> down, ReadOnlySpan<float> x, int rows, Span<float> y) =>
        Run(program, dtypes, hidden, intermediate, gate, up, down, x, rows, y, SimdDispatch.IsAvx2Supported, parallel: true);

    /// <summary>The same computation without SIMD. Public so a test can hold the AVX2 path to it.</summary>
    /// <inheritdoc cref="Apply"/>
    public static void ApplyScalar(ExpertProgram program, DType dtype, int hidden, int intermediate, ReadOnlySpan<byte> gate,
        ReadOnlySpan<byte> up, ReadOnlySpan<byte> down, ReadOnlySpan<float> x, int rows, Span<float> y) =>
        Run(program, new ExpertDTypes(dtype, dtype, dtype), hidden, intermediate, gate, up, down, x, rows, y, simd: false);

    /// <summary>The per-projection computation without SIMD. Public so a test can hold the AVX2 path to it.</summary>
    /// <inheritdoc cref="Apply(ExpertProgram, ExpertDTypes, int, int, ReadOnlySpan{byte}, ReadOnlySpan{byte}, ReadOnlySpan{byte}, ReadOnlySpan{float}, int, Span{float})"/>
    public static void ApplyScalar(ExpertProgram program, ExpertDTypes dtypes, int hidden, int intermediate, ReadOnlySpan<byte> gate,
        ReadOnlySpan<byte> up, ReadOnlySpan<byte> down, ReadOnlySpan<float> x, int rows, Span<float> y) =>
        Run(program, dtypes, hidden, intermediate, gate, up, down, x, rows, y, simd: false);

    private static void Run(ExpertProgram program, ExpertDTypes dtypes, int hidden, int intermediate, ReadOnlySpan<byte> gate,
        ReadOnlySpan<byte> up, ReadOnlySpan<byte> down, ReadOnlySpan<float> x, int rows, Span<float> y, bool simd, bool parallel = false)
    {
        ArgumentNullException.ThrowIfNull(program);
        program.Validated();
        ValidateShapes(dtypes, hidden, intermediate, gate.Length, up.Length, down.Length, x.Length, rows, y.Length);

        int maxDim = Math.Max(hidden, intermediate);
        int maxBlocks = maxDim / QuantBlock;
        sbyte[] codes = ArrayPool<sbyte>.Shared.Rent(rows * maxDim);
        float[] scales = ArrayPool<float>.Shared.Rent(rows * maxBlocks);
        int[] sums = ArrayPool<int>.Shared.Rent(rows * maxBlocks);
        float[] hiddenAct = ArrayPool<float>.Shared.Rent(rows * intermediate);
        try
        {
            fixed (byte* gp = gate)
            fixed (byte* up0 = up)
            fixed (byte* dp = down)
            fixed (float* xp = x)
            fixed (float* yp = y)
            fixed (sbyte* cp = codes)
            fixed (float* sp = scales)
            fixed (int* sumsP = sums)
            fixed (float* ap = hiddenAct)
            {
                Core(program, dtypes, hidden, intermediate, gp, up0, dp, xp, rows, yp, cp, sp, sumsP, ap, simd, parallel);
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(codes);
            ArrayPool<float>.Shared.Return(scales);
            ArrayPool<int>.Shared.Return(sums);
            ArrayPool<float>.Shared.Return(hiddenAct);
        }
    }

    private static void Core(ExpertProgram program, ExpertDTypes dtypes, int hidden, int intermediate, byte* gate, byte* up, byte* down,
        float* x, int rows, float* y, sbyte* codes, float* scales, int* sums, float* hiddenAct, bool simd, bool parallel)
    {
        QuantizeRows(x, rows, hidden, codes, scales, sums);
        if (parallel) GateUpParallel(program, dtypes, hidden, intermediate, (nint)gate, (nint)up, (nint)codes, (nint)scales, (nint)sums, rows, (nint)hiddenAct, simd);
        else GateUpRows(program, dtypes, hidden, intermediate, gate, up, codes, scales, sums, rows, hiddenAct, 0, intermediate, simd);

        QuantizeRows(hiddenAct, rows, intermediate, codes, scales, sums);
        if (parallel) DownParallel(dtypes, hidden, intermediate, (nint)down, (nint)codes, (nint)scales, (nint)sums, rows, (nint)y, simd);
        else DownRows(dtypes, hidden, intermediate, down, codes, scales, sums, rows, y, 0, hidden, simd);
    }

    // The fan-outs live in their own methods: a lambda's captured parameters are allocated on entry to the method that declares
    // them, whichever branch then runs, and the serial path must stay allocation-free. Pointers cross as addresses because a lambda
    // cannot capture a fixed local.
    private static void GateUpParallel(ExpertProgram program, ExpertDTypes dtypes, int hidden, int intermediate, nint gate, nint up,
        nint codes, nint scales, nint sums, int rows, nint hiddenAct, bool simd) =>
        CpuParallel.ForRanges(intermediate, RowsPerRange, 2L * rows * hidden, (start, length) =>
            GateUpRows(program, dtypes, hidden, intermediate, (byte*)gate, (byte*)up, (sbyte*)codes, (float*)scales, (int*)sums, rows,
                (float*)hiddenAct, (int)start, (int)(start + length), simd));

    private static void DownParallel(ExpertDTypes dtypes, int hidden, int intermediate, nint down, nint codes, nint scales, nint sums,
        int rows, nint y, bool simd) =>
        CpuParallel.ForRanges(hidden, RowsPerRange, (long)rows * intermediate, (start, length) =>
            DownRows(dtypes, hidden, intermediate, (byte*)down, (sbyte*)codes, (float*)scales, (int*)sums, rows, (float*)y, (int)start,
                (int)(start + length), simd));

    /// <summary>Output rows a parallel range covers: enough dot products to outweigh handing the range out.</summary>
    private const int RowsPerRange = 64;

    /// <summary>Gate and up rows <c>[first, end)</c> against the quantized input, activated and clamped into <paramref name="hiddenAct"/>.</summary>
    private static void GateUpRows(ExpertProgram program, ExpertDTypes dtypes, int hidden, int intermediate, byte* gate, byte* up,
        sbyte* codes, float* scales, int* sums, int rows, float* hiddenAct, int first, int end, bool simd)
    {
        int hiddenBlocks = hidden / QuantBlock;
        int gateRowBytes = checked((int)dtypes.Gate.ComputeByteCount(hidden));
        int upRowBytes = checked((int)dtypes.Up.ComputeByteCount(hidden));
        for (int i = first; i < end; i++)
        {
            byte* gateRow = gate + (long)i * gateRowBytes;
            byte* upRow = up + (long)i * upRowBytes;
            for (int r = 0; r < rows; r++)
            {
                float g = DotRow(dtypes.Gate, gateRow, codes + r * hidden, scales + r * hiddenBlocks, sums + r * hiddenBlocks, hidden, simd);
                float u = DotRow(dtypes.Up, upRow, codes + r * hidden, scales + r * hiddenBlocks, sums + r * hiddenBlocks, hidden, simd);
                (float clampedGate, float clampedUp) = program.Clamp(g, u);
                hiddenAct[r * intermediate + i] = ExpertProgramReference.Activate(program.Activation, clampedGate) * clampedUp;
            }
        }
    }

    /// <summary>Down rows <c>[first, end)</c> against the quantized hidden activation, into <paramref name="y"/>.</summary>
    private static void DownRows(ExpertDTypes dtypes, int hidden, int intermediate, byte* down, sbyte* codes, float* scales, int* sums,
        int rows, float* y, int first, int end, bool simd)
    {
        int interBlocks = intermediate / QuantBlock;
        int downRowBytes = checked((int)dtypes.Down.ComputeByteCount(intermediate));
        for (int d = first; d < end; d++)
        {
            byte* downRow = down + (long)d * downRowBytes;
            for (int r = 0; r < rows; r++)
            {
                y[r * hidden + d] = DotRow(dtypes.Down, downRow, codes + r * intermediate, scales + r * interBlocks, sums + r * interBlocks,
                    intermediate, simd);
            }
        }
    }

    /// <summary>Quantizes each row of <paramref name="src"/> into int8 codes, one float scale and one code sum per block.
    /// Blocks are <see cref="QuantBlock"/> wide; the row stride is <paramref name="dim"/> codes.</summary>
    private static void QuantizeRows(float* src, int rows, int dim, sbyte* codes, float* scales, int* sums)
    {
        int blocks = dim / QuantBlock;
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocks; b++)
            {
                float* v = src + r * dim + b * QuantBlock;
                float absMax = 0f;
                for (int k = 0; k < QuantBlock; k++)
                {
                    float a = MathF.Abs(v[k]);
                    if (a > absMax) absMax = a;
                }
                float scale = absMax / 127f;
                float inv = scale > 0f ? 1f / scale : 0f;
                sbyte* c = codes + r * dim + b * QuantBlock;
                int sum = 0;
                for (int k = 0; k < QuantBlock; k++)
                {
                    int q = Math.Clamp((int)MathF.Round(v[k] * inv), -127, 127);
                    c[k] = (sbyte)q;
                    sum += q;
                }
                scales[r * blocks + b] = scale;
                sums[r * blocks + b] = sum;
            }
        }
    }

    /// <summary>One weight row against one quantized activation row: the float dot product, folded block by block.</summary>
    private static float DotRow(DType dtype, byte* row, sbyte* act, float* actScale, int* actSum, int dim, bool simd)
    {
        float acc = 0f;
        if (dtype == DType.Q8_0)
        {
            int blocks = dim / QuantBlock;
            for (int b = 0; b < blocks; b++)
            {
                byte* block = row + b * Q8BlockBytes;
                float weightScale = ReadHalf(block);
                sbyte* w = (sbyte*)(block + 2);
                int dot = simd ? Q8BlockDotAvx2(w, act + b * QuantBlock) : Q8BlockDotScalar(w, act + b * QuantBlock);
                acc += weightScale * actScale[b] * dot;
            }
            return acc;
        }

        if (dtype == DType.Q6_K) return DotRowQ6K(row, act, actScale, dim, simd);
        if (dtype == DType.Q5_K) return DotRowQ5K(row, act, actScale, actSum, dim, simd);

        // Q4_K: 256-element super-blocks of eight 32-element sub-blocks, each with a 6-bit scale and minimum.
        int superBlocks = dim / Q4KSuperBlockElems;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = row + sb * Q4KBlockBytes;
            float d = ReadHalf(block);
            float dmin = ReadHalf(block + 2);
            byte* packedScales = block + 4;
            byte* quants = block + 16;
            for (int j = 0; j < 8; j++)
            {
                GetScaleMinK4(j, packedScales, out int sc, out int mm);
                int b = sb * 8 + j;
                byte* subQuants = quants + (j / 2) * QuantBlock;
                int shift = (j % 2 == 0) ? 0 : 4;
                int dot = simd ? Q4SubDotAvx2(subQuants, shift, act + b * QuantBlock) : Q4SubDotScalar(subQuants, shift, act + b * QuantBlock);
                float sub = d * sc * dot - dmin * mm * actSum[b];
                acc += actScale[b] * sub;
            }
        }
        return acc;
    }

    /// <summary>Q5_K: Q4_K's super-block (scales and minimums packed the same way) with a fifth bit per value in <c>qh</c>.
    /// Value <c>i</c> of sub-block <c>j</c> is the nibble of <c>qs</c> plus 16 when bit <c>j</c> of <c>qh[i]</c> is set.</summary>
    private static float DotRowQ5K(byte* row, sbyte* act, float* actScale, int* actSum, int dim, bool simd)
    {
        float acc = 0f;
        int superBlocks = dim / Q4KSuperBlockElems;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = row + sb * Q5KBlockBytes;
            float d = ReadHalf(block);
            float dmin = ReadHalf(block + 2);
            byte* packedScales = block + 4;
            byte* qh = block + 16;
            byte* qs = block + 48;
            for (int j = 0; j < 8; j++)
            {
                GetScaleMinK4(j, packedScales, out int sc, out int mm);
                int b = sb * 8 + j;
                byte* subQuants = qs + (j / 2) * QuantBlock;
                int shift = (j % 2 == 0) ? 0 : 4;
                int dot = simd ? Q5SubDotAvx2(subQuants, shift, qh, j, act + b * QuantBlock)
                    : Q5SubDotScalar(subQuants, shift, qh, j, act + b * QuantBlock);
                acc += actScale[b] * (d * sc * dot - dmin * mm * actSum[b]);
            }
        }
        return acc;
    }

    /// <summary>Q6_K: 256-element super-blocks of two 128-element halves. A value is a low nibble from <c>ql</c> and two high bits
    /// from <c>qh</c>, minus 32, scaled by a signed 8-bit scale per 16 values. A 32-value activation block therefore spans two
    /// scales, and its integer dot is kept as two 16-value sums.</summary>
    private static float DotRowQ6K(byte* row, sbyte* act, float* actScale, int dim, bool simd)
    {
        float acc = 0f;
        int superBlocks = dim / Q4KSuperBlockElems;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = row + sb * Q6KBlockBytes;
            float d = ReadHalf(block + 208);
            sbyte* sc = (sbyte*)(block + 192);
            for (int half = 0; half < 2; half++)
            {
                byte* ql = block + half * 64;
                byte* qh = block + 128 + half * 32;
                for (int g = 0; g < 4; g++)
                {
                    int b = sb * 8 + half * 4 + g;
                    sbyte* a = act + b * QuantBlock;
                    int lo, hi;
                    if (simd) Q6GroupDotAvx2(ql, qh, g, a, out lo, out hi);
                    else Q6GroupDotScalar(ql, qh, g, a, out lo, out hi);
                    int scaleBase = half * 8 + 2 * g;
                    acc += actScale[b] * d * (sc[scaleBase] * lo + sc[scaleBase + 1] * hi);
                }
            }
        }
        return acc;
    }

    private static float ReadHalf(byte* p) => (float)Unsafe.ReadUnaligned<Half>(p);

    /// <summary>The 6-bit scale and minimum of sub-block <paramref name="j"/>, unpacked as in the GGUF K-quant codecs.</summary>
    private static void GetScaleMinK4(int j, byte* q, out int sc, out int mm)
    {
        if (j < 4)
        {
            sc = q[j] & 63;
            mm = q[j + 4] & 63;
        }
        else
        {
            sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            mm = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }
    }

    private static void ValidateShapes(ExpertDTypes dtypes, int hidden, int intermediate, long gateLength, long upLength,
        long downLength, long xLength, int rows, long yLength)
    {
        foreach (DType dtype in (ReadOnlySpan<DType>)[dtypes.Gate, dtypes.Up, dtypes.Down])
        {
            if (!Supports(dtype))
                throw new NotSupportedException($"CPU expert kernels run Q8_0, Q4_K, Q5_K and Q6_K, not {dtype.Name}.");
            if (hidden % dtype.BlockElementCount != 0 || intermediate % dtype.BlockElementCount != 0)
                throw new ArgumentException($"Hidden {hidden} and intermediate {intermediate} must be multiples of {dtype.Name}'s block size.");
        }
        if (rows < 1 || rows > MaxRows) throw new ArgumentOutOfRangeException(nameof(rows), rows, $"Rows must be 1 to {MaxRows}.");
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden), hidden, "Hidden must be positive.");
        if (intermediate <= 0) throw new ArgumentOutOfRangeException(nameof(intermediate), intermediate, "Intermediate must be positive.");
        if (hidden % QuantBlock != 0 || intermediate % QuantBlock != 0)
            throw new ArgumentException($"Hidden {hidden} and intermediate {intermediate} must be multiples of {QuantBlock}.");
        long expectedGate = (long)intermediate * dtypes.Gate.ComputeByteCount(hidden);
        long expectedUp = (long)intermediate * dtypes.Up.ComputeByteCount(hidden);
        long expectedDown = (long)hidden * dtypes.Down.ComputeByteCount(intermediate);
        if (gateLength != expectedGate) throw new ArgumentException($"Gate must hold {expectedGate} bytes; it holds {gateLength}.");
        if (upLength != expectedUp) throw new ArgumentException($"Up must hold {expectedUp} bytes; it holds {upLength}.");
        if (downLength != expectedDown) throw new ArgumentException($"Down must hold {expectedDown} bytes; it holds {downLength}.");
        if (xLength != (long)rows * hidden) throw new ArgumentException($"x must hold {rows} rows of {hidden}; it holds {xLength}.");
        if (yLength != (long)rows * hidden) throw new ArgumentException($"y must hold {rows} rows of {hidden}; it holds {yLength}.");
    }
}
