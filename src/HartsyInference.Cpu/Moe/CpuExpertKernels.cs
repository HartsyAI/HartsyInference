using System.Buffers;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Moe;

/// <summary>
/// Quantized expert kernels for the CPU: one MoE expert <c>down( act(clampGate(Gate·x)) * clampUp(Up·x) )</c> computed
/// directly on packed Q8_0 or Q4_K bytes, for 1 to 8 token rows at once.
///
/// <para><b>Layout.</b> Gate and up are <c>[intermediate, hidden]</c> row-major, down is <c>[hidden, intermediate]</c>;
/// each row is <c>dtype.ComputeByteCount(cols)</c> bytes, exactly as <c>ExpertPackWriter</c> writes them.</para>
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

    /// <summary>Runs the expert, using AVX2 when the CPU supports it.</summary>
    /// <param name="program">Activation and clamp bounds.</param>
    /// <param name="dtype">Packed weight dtype: <see cref="DType.Q8_0"/> or <see cref="DType.Q4_K"/>.</param>
    /// <param name="hidden">Model width H; a multiple of 256 for Q4_K and of 32 for Q8_0.</param>
    /// <param name="intermediate">Expert inner width I; same divisibility as <paramref name="hidden"/>.</param>
    /// <param name="gate">Packed gate matrix, <c>[I, H]</c>.</param>
    /// <param name="up">Packed up matrix, <c>[I, H]</c>.</param>
    /// <param name="down">Packed down matrix, <c>[H, I]</c>.</param>
    /// <param name="x"><c>rows × H</c> inputs, row-major.</param>
    /// <param name="rows">Token rows, 1 to <see cref="MaxRows"/>.</param>
    /// <param name="y"><c>rows × H</c> outputs, overwritten.</param>
    /// <exception cref="ArgumentException">A shape, buffer length or divisibility rule is violated.</exception>
    /// <exception cref="NotSupportedException">The dtype is not Q8_0 or Q4_K.</exception>
    public static void Apply(ExpertProgram program, DType dtype, int hidden, int intermediate, ReadOnlySpan<byte> gate,
        ReadOnlySpan<byte> up, ReadOnlySpan<byte> down, ReadOnlySpan<float> x, int rows, Span<float> y) =>
        Run(program, dtype, hidden, intermediate, gate, up, down, x, rows, y, SimdDispatch.IsAvx2Supported);

    /// <summary>The same computation without SIMD. Public so a test can hold the AVX2 path to it.</summary>
    /// <inheritdoc cref="Apply"/>
    public static void ApplyScalar(ExpertProgram program, DType dtype, int hidden, int intermediate, ReadOnlySpan<byte> gate,
        ReadOnlySpan<byte> up, ReadOnlySpan<byte> down, ReadOnlySpan<float> x, int rows, Span<float> y) =>
        Run(program, dtype, hidden, intermediate, gate, up, down, x, rows, y, simd: false);

    private static void Run(ExpertProgram program, DType dtype, int hidden, int intermediate, ReadOnlySpan<byte> gate,
        ReadOnlySpan<byte> up, ReadOnlySpan<byte> down, ReadOnlySpan<float> x, int rows, Span<float> y, bool simd)
    {
        ArgumentNullException.ThrowIfNull(program);
        program.Validated();
        ValidateShapes(dtype, hidden, intermediate, gate.Length, up.Length, down.Length, x.Length, rows, y.Length);

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
                Core(program, dtype, hidden, intermediate, gp, up0, dp, xp, rows, yp, cp, sp, sumsP, ap, simd);
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

    private static void Core(ExpertProgram program, DType dtype, int hidden, int intermediate, byte* gate, byte* up, byte* down,
        float* x, int rows, float* y, sbyte* codes, float* scales, int* sums, float* hiddenAct, bool simd)
    {
        int hiddenBlocks = hidden / QuantBlock;
        int interBlocks = intermediate / QuantBlock;
        int gateRowBytes = checked((int)dtype.ComputeByteCount(hidden));
        int downRowBytes = checked((int)dtype.ComputeByteCount(intermediate));

        QuantizeRows(x, rows, hidden, codes, scales, sums);
        for (int i = 0; i < intermediate; i++)
        {
            byte* gateRow = gate + (long)i * gateRowBytes;
            byte* upRow = up + (long)i * gateRowBytes;
            for (int r = 0; r < rows; r++)
            {
                float g = DotRow(dtype, gateRow, codes + r * hidden, scales + r * hiddenBlocks, sums + r * hiddenBlocks, hidden, simd);
                float u = DotRow(dtype, upRow, codes + r * hidden, scales + r * hiddenBlocks, sums + r * hiddenBlocks, hidden, simd);
                (float clampedGate, float clampedUp) = program.Clamp(g, u);
                hiddenAct[r * intermediate + i] = ExpertProgramReference.Activate(program.Activation, clampedGate) * clampedUp;
            }
        }

        QuantizeRows(hiddenAct, rows, intermediate, codes, scales, sums);
        for (int d = 0; d < hidden; d++)
        {
            byte* downRow = down + (long)d * downRowBytes;
            for (int r = 0; r < rows; r++)
            {
                y[r * hidden + d] = DotRow(dtype, downRow, codes + r * intermediate, scales + r * interBlocks, sums + r * interBlocks,
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

    private static void ValidateShapes(DType dtype, int hidden, int intermediate, long gateLength, long upLength, long downLength,
        long xLength, int rows, long yLength)
    {
        if (!(dtype == DType.Q8_0 || dtype == DType.Q4_K))
            throw new NotSupportedException($"CPU expert kernels run Q8_0 and Q4_K, not {dtype.Name}.");
        if (rows < 1 || rows > MaxRows) throw new ArgumentOutOfRangeException(nameof(rows), rows, $"Rows must be 1 to {MaxRows}.");
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden), hidden, "Hidden must be positive.");
        if (intermediate <= 0) throw new ArgumentOutOfRangeException(nameof(intermediate), intermediate, "Intermediate must be positive.");
        if (hidden % dtype.BlockElementCount != 0 || intermediate % dtype.BlockElementCount != 0 || hidden % QuantBlock != 0
            || intermediate % QuantBlock != 0)
            throw new ArgumentException($"Hidden {hidden} and intermediate {intermediate} must be multiples of {dtype.Name}'s block size.");
        long expectedGate = (long)intermediate * dtype.ComputeByteCount(hidden);
        long expectedDown = (long)hidden * dtype.ComputeByteCount(intermediate);
        if (gateLength != expectedGate) throw new ArgumentException($"Gate must hold {expectedGate} bytes; it holds {gateLength}.");
        if (upLength != expectedGate) throw new ArgumentException($"Up must hold {expectedGate} bytes; it holds {upLength}.");
        if (downLength != expectedDown) throw new ArgumentException($"Down must hold {expectedDown} bytes; it holds {downLength}.");
        if (xLength != (long)rows * hidden) throw new ArgumentException($"x must hold {rows} rows of {hidden}; it holds {xLength}.");
        if (yLength != (long)rows * hidden) throw new ArgumentException($"y must hold {rows} rows of {hidden}; it holds {yLength}.");
    }
}
