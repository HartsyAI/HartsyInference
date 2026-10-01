using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Kernels;

/// <summary>int8-weight, uint8-activation matrix-vector products (<see cref="Core.Backends.IBackend.LinearI8U8"/>)
/// and the activation codes that feed them (<see cref="Core.Backends.IBackend.QuantizeActivationsU8"/>).
///
/// <para><b>Exact sums.</b> Each output starts as the exact int32 sum of its uint8 × int8 products, which is what
/// RNNoise's default (SSE2) build computes. AVX2's <c>maddubs</c> would be faster, but it saturates each pair of
/// products at int16, and a weight pair whose magnitudes add up to more than 128 against two codes near 255 does
/// overflow it. So the AVX2 path widens the bytes to 16 bits and multiplies with <c>pmaddwd</c>, whose pair sums
/// cannot overflow. The scalar path adds the same products in plain integers. Integer sums do not depend on order,
/// so both paths give the same bits.</para>
///
/// <para><b>Float epilogue.</b> <c>(float)sum·scale</c>, then <c>+ bias</c>, then <c>+ diag·x</c>. Each step is rounded
/// on its own, never fused, the order upstream's <c>compute_linear</c> uses.</para>
///
/// <para>Rows fan out over 8-row blocks through <see cref="CpuParallel"/>, and nothing is allocated per call:
/// the state is a struct and the block body a static delegate.</para></summary>
public static class Int8GemvKernels
{
    /// <summary>Largest K a call accepts. Its sums fit an int32 with room to spare (16384 products of at most
    /// 255 × 128), and one activation row's split form stays at 32 KB of stack.</summary>
    public const int MaxInputs = 1 << 14;

    // Ints of split activations held on the stack per pass over the weights: 64 KB. A batch of activation rows that
    // needs more takes several passes.
    private const int MaxSplitInts = 16 * 1024;

    private static readonly Action<int, Pass> BlockBody = Block;

    /// <summary>Encodes <paramref name="input"/> (F32) into <paramref name="output"/> (U8) with
    /// <see cref="Int8Tiles.QuantizeActivation"/>.</summary>
    public static unsafe void QuantizeActivations(Tensor output, Tensor input)
    {
        if (output.DType != DType.U8 || input.DType != DType.F32)
            throw new ArgumentException($"QuantizeActivationsU8 writes U8 from F32, got {output.DType} from {input.DType}.");
        if (output.ElementCount != input.ElementCount)
            throw new ArgumentException(
                $"QuantizeActivationsU8 needs equal element counts, got {output.ElementCount} and {input.ElementCount}.");
        byte* codes = (byte*)output.DataPointer;
        float* values = (float*)input.DataPointer;
        long count = input.ElementCount;
        for (long i = 0; i < count; i++) codes[i] = Int8Tiles.QuantizeActivation(values[i]);
    }

    /// <summary><c>output[m, n] = scale[n]·Σₖ weight[n, k]·input[m, k] + bias[n] + diag[n]·diagInput[m, n mod K]</c>;
    /// see <see cref="Core.Backends.IBackend.LinearI8U8"/> for the contract.</summary>
    public static unsafe void Linear(Tensor output, Tensor input, Tensor weight, Tensor scale, Tensor? bias,
        Tensor? diag, Tensor? diagInput) =>
        Run(Prepare(output, input, weight, scale, bias, diag, diagInput), Avx2.IsSupported);

    /// <summary>The same product computed without SIMD, which non-AVX2 CPUs run. Public so a test can hold the AVX2
    /// path to it on a machine that has AVX2.</summary>
    public static unsafe void LinearScalar(Tensor output, Tensor input, Tensor weight, Tensor scale, Tensor? bias,
        Tensor? diag, Tensor? diagInput) =>
        Run(Prepare(output, input, weight, scale, bias, diag, diagInput), simd: false);

    private static unsafe Pass Prepare(Tensor output, Tensor input, Tensor weight, Tensor scale, Tensor? bias,
        Tensor? diag, Tensor? diagInput)
    {
        if (weight.DType != DType.I8 || weight.Shape.Rank != 4 || weight.Shape[2] != Int8Tiles.Rows
            || weight.Shape[3] != Int8Tiles.Cols)
            throw new ArgumentException(
                $"LinearI8U8 needs an I8 weight of shape [N/{Int8Tiles.Rows}, K/{Int8Tiles.Cols}, {Int8Tiles.Rows}, "
                + $"{Int8Tiles.Cols}], got {weight.DType} {weight.Shape}.");
        long n = weight.Shape[0] * Int8Tiles.Rows;
        long k = weight.Shape[1] * Int8Tiles.Cols;
        if (k > MaxInputs)
            throw new ArgumentException($"LinearI8U8 takes at most {MaxInputs} inputs, got {k}.");
        if (input.DType != DType.U8 || input.ElementCount % k != 0)
            throw new ArgumentException(
                $"LinearI8U8 needs U8 input rows of {k} codes, got {input.DType} with {input.ElementCount} elements.");
        long m = input.ElementCount / k;
        if (output.DType != DType.F32 || output.ElementCount != m * n)
            throw new ArgumentException(
                $"LinearI8U8 writes F32 [{m}, {n}], got {output.DType} with {output.ElementCount} elements.");
        RequireVector(scale, n, nameof(scale));
        if (bias is not null) RequireVector(bias, n, nameof(bias));
        if ((diag is null) != (diagInput is null))
            throw new ArgumentException("LinearI8U8 takes diag and diagInput together or not at all.");
        if (diag is not null)
        {
            RequireVector(diag, n, nameof(diag));
            if (n % k != 0 || k % Int8Tiles.Rows != 0)
                throw new ArgumentException(
                    $"A diagonal term needs N a multiple of K and K a multiple of {Int8Tiles.Rows}, got N {n}, K {k}.");
            if (diagInput!.DType != DType.F32 || diagInput.ElementCount != m * k)
                throw new ArgumentException(
                    $"diagInput must be F32 [{m}, {k}], got {diagInput.DType} with {diagInput.ElementCount} elements.");
        }

        return new Pass
        {
            Output = (float*)output.DataPointer,
            Input = (byte*)input.DataPointer,
            Weight = (sbyte*)weight.DataPointer,
            Scale = (float*)scale.DataPointer,
            Bias = bias is null ? null : (float*)bias.DataPointer,
            Diag = diag is null ? null : (float*)diag.DataPointer,
            DiagInput = diagInput is null ? null : (float*)diagInput.DataPointer,
            Rows = (int)m,
            N = (int)n,
            K = (int)k,
        };
    }

    private static void RequireVector(Tensor tensor, long count, string name)
    {
        if (tensor.DType != DType.F32 || tensor.ElementCount != count)
            throw new ArgumentException(
                $"LinearI8U8 needs {name} as F32 with {count} elements, got {tensor.DType} with {tensor.ElementCount}.");
    }

    private static unsafe void Run(Pass pass, bool simd)
    {
        int groups = pass.K / Int8Tiles.Cols;
        int rowsPerPass = Math.Max(1, Math.Min(pass.Rows, MaxSplitInts / (2 * groups)));
        int* split = stackalloc int[simd ? 2 * groups * rowsPerPass : 1];
        int blocks = pass.N / Int8Tiles.Rows;
        Pass chunk = pass;
        chunk.Simd = simd;
        chunk.Split = split;
        for (int first = 0; first < pass.Rows; first += rowsPerPass)
        {
            chunk.Rows = Math.Min(rowsPerPass, pass.Rows - first);
            chunk.Input = pass.Input + (long)first * pass.K;
            chunk.Output = pass.Output + (long)first * pass.N;
            chunk.DiagInput = pass.DiagInput is null ? null : pass.DiagInput + (long)first * pass.K;
            if (simd) Split(chunk.Input, chunk.Rows, groups, split);
            CpuParallel.For(blocks, (long)chunk.Rows * pass.N * pass.K, chunk, BlockBody);
        }
    }

    /// <summary>Spreads each group of four codes <c>x0 x1 x2 x3</c> into two ints whose 16-bit halves are
    /// <c>(x0, x2)</c> and <c>(x1, x3)</c>, the operands <c>pmaddwd</c> pairs with a tile's even and odd weight bytes.
    /// Done once per call, so the inner loop only broadcasts.</summary>
    private static unsafe void Split(byte* codes, int rows, int groups, int* split)
    {
        for (int r = 0; r < rows; r++)
        {
            byte* row = codes + (long)r * groups * Int8Tiles.Cols;
            int* target = split + (long)r * groups * 2;
            for (int g = 0; g < groups; g++)
            {
                uint four = Unsafe.ReadUnaligned<uint>(row + g * Int8Tiles.Cols);
                target[2 * g] = (int)(four & 0x00FF00FFu);
                target[2 * g + 1] = (int)((four >> 8) & 0x00FF00FFu);
            }
        }
    }

    /// <summary>One 8-row block of the output, for every activation row of the pass.</summary>
    private static unsafe void Block(int block, Pass p)
    {
        int groups = p.K / Int8Tiles.Cols;
        sbyte* tiles = p.Weight + (long)block * groups * Int8Tiles.Bytes;
        int firstRow = block * Int8Tiles.Rows;
        if (p.Simd)
        {
            int r = 0;
            for (; r + 1 < p.Rows; r += 2)
            {
                (Vector256<int> a, Vector256<int> b) = DotPair(tiles, groups, p.Split + (long)r * groups * 2,
                    p.Split + (long)(r + 1) * groups * 2);
                Store(a, p, r, firstRow);
                Store(b, p, r + 1, firstRow);
            }
            if (r < p.Rows) Store(Dot(tiles, groups, p.Split + (long)r * groups * 2), p, r, firstRow);
            return;
        }

        for (int r = 0; r < p.Rows; r++)
        {
            byte* codes = p.Input + (long)r * p.K;
            for (int q = 0; q < Int8Tiles.Rows; q++)
            {
                int sum = 0;
                sbyte* w = tiles + q * Int8Tiles.Cols;
                for (int g = 0; g < groups; g++, w += Int8Tiles.Bytes)
                {
                    byte* x = codes + g * Int8Tiles.Cols;
                    sum += x[0] * w[0] + x[1] * w[1] + x[2] * w[2] + x[3] * w[3];
                }
                int row = firstRow + q;
                float value = sum * p.Scale[row];
                if (p.Bias is not null) value += p.Bias[row];
                if (p.Diag is not null) value += p.Diag[row] * p.DiagInput[(long)r * p.K + row % p.K];
                p.Output[(long)r * p.N + row] = value;
            }
        }
    }

    /// <summary>Eight rows' exact sums against one activation row.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<int> Dot(sbyte* tiles, int groups, int* split)
    {
        Vector256<int> sum = Vector256<int>.Zero;
        for (int g = 0; g < groups; g++, tiles += Int8Tiles.Bytes)
        {
            Vector256<short> weights = Avx.LoadVector256((short*)tiles);
            Vector256<short> even = Avx2.ShiftRightArithmetic(Avx2.ShiftLeftLogical(weights, 8), 8);
            Vector256<short> odd = Avx2.ShiftRightArithmetic(weights, 8);
            sum = Avx2.Add(sum, Avx2.Add(
                Avx2.MultiplyAddAdjacent(Vector256.Create(split[2 * g]).AsInt16(), even),
                Avx2.MultiplyAddAdjacent(Vector256.Create(split[2 * g + 1]).AsInt16(), odd)));
        }
        return sum;
    }

    /// <summary>Eight rows' exact sums against two activation rows, widening each tile once for both.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe (Vector256<int>, Vector256<int>) DotPair(sbyte* tiles, int groups, int* first, int* second)
    {
        Vector256<int> a = Vector256<int>.Zero, b = a;
        for (int g = 0; g < groups; g++, tiles += Int8Tiles.Bytes)
        {
            Vector256<short> weights = Avx.LoadVector256((short*)tiles);
            Vector256<short> even = Avx2.ShiftRightArithmetic(Avx2.ShiftLeftLogical(weights, 8), 8);
            Vector256<short> odd = Avx2.ShiftRightArithmetic(weights, 8);
            a = Avx2.Add(a, Avx2.Add(
                Avx2.MultiplyAddAdjacent(Vector256.Create(first[2 * g]).AsInt16(), even),
                Avx2.MultiplyAddAdjacent(Vector256.Create(first[2 * g + 1]).AsInt16(), odd)));
            b = Avx2.Add(b, Avx2.Add(
                Avx2.MultiplyAddAdjacent(Vector256.Create(second[2 * g]).AsInt16(), even),
                Avx2.MultiplyAddAdjacent(Vector256.Create(second[2 * g + 1]).AsInt16(), odd)));
        }
        return (a, b);
    }

    /// <summary>The epilogue for eight rows: the same three roundings, in the same order, as the scalar path.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector256<int> sums, Pass p, int r, int firstRow)
    {
        Vector256<float> value = Avx.Multiply(Avx.ConvertToVector256Single(sums), Avx.LoadVector256(p.Scale + firstRow));
        if (p.Bias is not null) value = Avx.Add(value, Avx.LoadVector256(p.Bias + firstRow));
        if (p.Diag is not null)
            value = Avx.Add(value, Avx.Multiply(Avx.LoadVector256(p.Diag + firstRow),
                Avx.LoadVector256(p.DiagInput + (long)r * p.K + firstRow % p.K)));
        Avx.Store(p.Output + (long)r * p.N + firstRow, value);
    }

    /// <summary>What <see cref="Block"/> needs from the call that fans it out.</summary>
    private unsafe struct Pass
    {
        public float* Output;
        public byte* Input;
        public sbyte* Weight;
        public float* Scale;
        public float* Bias;
        public float* Diag;
        public float* DiagInput;
        public int* Split;
        public int Rows;
        public int N;
        public int K;
        public bool Simd;
    }
}
