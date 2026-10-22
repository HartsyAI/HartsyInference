using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>The executable specification of INT8 attention v2 (docs/ATTENTION_CONTRACT.md §5): the kernel's stats pass,
/// quant launch and online recurrence on the CPU. A kernel reproduces its bytes, scales, means and INT32 QK products bit for
/// bit, and its F32 output up to summation order and the hardware exp2.</summary>
public static class Int8AttentionV2Reference
{
    /// <summary>log2(255) rounded down, so a key tile's maximum maps to u8 255 (Comfy Kitchen's S_U8_OFFSET).</summary>
    public const float U8Offset = 7.9943534f;

    /// <summary>log2(448): the running maximum maps to E4M3's largest value (SageAttention's S_FP8_OFFSET).</summary>
    public const float Fp8Offset = 8.807f;

    /// <summary>Where V's per-channel E4M3 scale puts each channel's absmax (SageAttention2++ on sm_89).</summary>
    public const float Fp8ValueRange = 2.25f;

    /// <summary>Keys per partial sum of the K channel mean.</summary>
    public const int MeanChunk = 64;

    /// <summary>log2(e) in F32: folded into the Q scale and applied to biases, sinks and soft-capped scores.</summary>
    public const float Log2E = 1.44269504f;

    private const float Ln2 = 0.693147182f;
    private const float MinNormal = 1.17549435e-38f;

    /// <summary>The head dim a kernel template runs for <paramref name="headDim"/>: 64, 128 or 256, zero-padded.</summary>
    /// <exception cref="AttentionUnsupportedException">The head dim is above 256.</exception>
    public static int PaddedHeadDim(int headDim) => headDim switch
    {
        <= 0 => throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "Head dim must be positive."),
        <= 64 => 64,
        <= 128 => 128,
        <= 256 => 256,
        _ => throw new AttentionUnsupportedException($"INT8 attention v2 runs head dims up to 256; got {headDim}."),
    };

    /// <summary>The largest |INT32 PV partial| of one key tile, <c>BC · 255 · 127</c>: below 2²⁴, exact in F32.</summary>
    public static int PvPartialBound(int blockCols) => blockCols * 255 * 127;

    /// <summary>The largest |INT32 QK product|, <c>127² · Dp</c>: below 2²⁴, exact in F32.</summary>
    public static int QkDotBound(int paddedHeadDim) => 127 * 127 * paddedHeadDim;

    /// <summary>Quantizes and attends <paramref name="rows"/> (all when empty) of every (batch, head) of
    /// <paramref name="call"/>, writing those output rows and, when given, their natural-log LSE <c>[B, Hq, Sq]</c>.</summary>
    /// <exception cref="AttentionUnsupportedException">The call carries something INT8 v2 does not run.</exception>
    public static void Run(in AttentionCall call, Int8AttentionV2Options options, ReadOnlySpan<int> rows = default,
        Span<float> lse = default) => Attend(in call, Quantize(in call, options, rows), lse);

    /// <summary>The stats pass and quant launch for <paramref name="rows"/> (all when empty) of <paramref name="call"/>;
    /// with <paramref name="scales"/>, its means and Q/K/V scales replace the computed ones (a GPU's, to compare bytes).</summary>
    /// <exception cref="AttentionUnsupportedException">The call carries something INT8 v2 does not run.</exception>
    public static Int8AttentionV2Operands Quantize(in AttentionCall call, Int8AttentionV2Options options,
        ReadOnlySpan<int> rows = default, Int8AttentionV2Operands? scales = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        Check(in call, options);
        Int8AttentionV2Operands ops = new(options, call.Batch, call.Heads, call.KvHeads, call.ValidKvLength,
            call.QkHeadDim, RowList(rows, call.QueryTokens));
        if (scales is not null) CopyScales(scales, ops);
        int dp = ops.PaddedHeadDim, kvLen = ops.ValidKeys, segments = ops.Segments;
        int[] bounds = SegmentBounds(options, kvLen);
        float[] k = new float[Math.Max(1, kvLen) * dp], v = new float[k.Length];
        for (int b = 0; b < call.Batch; b++)
            for (int kh = 0; kh < call.KvHeads; kh++)
            {
                int kvh = b * call.KvHeads + kh;
                for (int j = 0; j < kvLen; j++)
                {
                    ReadRow(call.Key, b, kh, j, k.AsSpan(j * dp, dp));
                    ReadRow(call.Value, b, kh, j, v.AsSpan(j * dp, dp));
                }
                Span<float> means = ops.KeyMeans.AsSpan(kvh * segments * dp, segments * dp);
                if (scales is null && options.SmoothKeys)
                    for (int s = 0; s < segments; s++)
                        SegmentMean(k, dp, bounds[s], bounds[s + 1], means.Slice(s * dp, dp));
                if (options.SmoothKeys)
                    for (int s = 0; s < segments; s++)
                        for (int j = bounds[s]; j < bounds[s + 1]; j++)
                            for (int c = 0; c < dp; c++) k[j * dp + c] -= means[s * dp + c];
                QuantizeKeys(ops, kvh, k, scales is null);
                QuantizeValues(ops, kvh, v, scales is null);
            }
        float[] q = new float[dp];
        int group = call.KvGroup, nRows = ops.Rows.Length;
        for (int b = 0; b < call.Batch; b++)
            for (int h = 0; h < call.Heads; h++)
            {
                int kvh = b * call.KvHeads + h / group;
                ReadOnlySpan<float> means = ops.KeyMeans.AsSpan(kvh * segments * dp, segments * dp);
                for (int ri = 0; ri < nRows; ri++)
                {
                    int at = (b * call.Heads + h) * nRows + ri;
                    ReadRow(call.Query, b, h, ops.Rows[ri], q);
                    if (scales is null) ops.QueryScales[at] = Scale(AbsMax(q), 127f);
                    float inv = 1f / ops.QueryScales[at];
                    for (int c = 0; c < dp; c++) ops.Queries[at * dp + c] = ToS8(q[c] * inv);
                    if (!options.SmoothKeys) continue;
                    for (int s = 0; s < segments; s++)
                    {
                        float dot = 0f;
                        for (int c = 0; c < dp; c++) dot = MathF.FusedMultiplyAdd(q[c], means[s * dp + c], dot);
                        ops.QueryMeanDots[at * segments + s] = dot;
                    }
                }
            }
        return ops;
    }

    /// <summary>The online recurrence over <paramref name="operands"/>: writes their rows of <paramref name="call"/>'s
    /// output and, when given, the natural-log LSE into <paramref name="lse"/> <c>[B, Hq, Sq]</c>.</summary>
    /// <exception cref="ArgumentException">The operands were quantized for a different call.</exception>
    public static void Attend(in AttentionCall call, Int8AttentionV2Operands operands, Span<float> lse = default)
    {
        ArgumentNullException.ThrowIfNull(operands);
        Check(in call, operands.Options);
        if (operands.Batch != call.Batch || operands.Heads != call.Heads || operands.KvHeads != call.KvHeads
            || operands.ValidKeys != call.ValidKvLength || operands.HeadDim != call.QkHeadDim
            || operands.Rows.Length > 0 && operands.Rows[^1] >= call.QueryTokens)
            throw new ArgumentException("The operands were quantized for a different call.", nameof(operands));
        long rowsTotal = (long)call.Batch * call.Heads * call.QueryTokens;
        if (!lse.IsEmpty && lse.Length != rowsTotal)
            throw new ArgumentException($"LSE holds {lse.Length} values; [B, Hq, Sq] needs {rowsTotal}.", nameof(lse));
        new Recurrence(in call, operands).Run(in call, lse);
    }

    /// <summary>The INT32 QK product of key <paramref name="key"/> and query row <paramref name="rowIndex"/> (an index into
    /// <see cref="Int8AttentionV2Operands.Rows"/>) of (<paramref name="batch"/>, <paramref name="head"/>).</summary>
    public static int QkDot(Int8AttentionV2Operands operands, int batch, int head, int rowIndex, int key)
    {
        ArgumentNullException.ThrowIfNull(operands);
        int dp = operands.PaddedHeadDim, kh = head / (operands.Heads / operands.KvHeads);
        int qRow = (batch * operands.Heads + head) * operands.Rows.Length + rowIndex;
        int kRow = (batch * operands.KvHeads + kh) * operands.ValidKeys + key;
        ReadOnlySpan<sbyte> q = operands.Queries.AsSpan(qRow * dp, dp), k = operands.Keys.AsSpan(kRow * dp, dp);
        int dot = 0;
        for (int c = 0; c < dp; c++) dot += q[c] * k[c];
        return dot;
    }

    /// <summary>What query rows <c>[queryRow0, +queryRows)</c> of (<paramref name="batch"/>, <paramref name="head"/>)
    /// see of key slots <c>[key0, +keys)</c> under the causal and window bounds, the valid key length and the bias's
    /// exclusions. Rows past Sq are dropped, as in a partial query tile; keys past the valid ones are masked.</summary>
    /// <exception cref="ArgumentException">A malformed operand or bias, a negative tile bound, or (batch, head) outside
    /// the call.</exception>
    /// <exception cref="AttentionUnsupportedException">The call carries something INT8 v2 does not run.</exception>
    public static AttentionTileClass ClassifyTile(in AttentionCall call, int batch, int head, int queryRow0,
        int queryRows, int key0, int keys)
    {
        Check(in call);
        if ((uint)batch >= (uint)call.Batch)
            throw new ArgumentOutOfRangeException(nameof(batch), batch, $"The call has {call.Batch} batches.");
        if ((uint)head >= (uint)call.Heads)
            throw new ArgumentOutOfRangeException(nameof(head), head, $"The call has {call.Heads} heads.");
        if (queryRow0 < 0 || queryRows < 0 || key0 < 0 || keys < 0)
            throw new ArgumentException(
                $"Tile bounds are non-negative; got rows {queryRow0}+{queryRows}, keys {key0}+{keys}.");
        int end = (int)Math.Min((long)key0 + keys, call.ValidKvLength);
        bool[] excluded = new bool[Math.Max(0, end - key0)];
        StageBias(in call, batch, head, key0, new float[excluded.Length], excluded);
        int rowEnd = (int)Math.Min((long)queryRow0 + queryRows, call.QueryTokens);
        return Classify(call.Mask, call.Positions.QueryOffset, call.ValidKvLength, queryRow0, rowEnd, key0, keys,
            excluded);
    }

    /// <summary>E4M3 rounded to nearest even and saturated to ±448, as <c>cvt.rn.satfinite</c>; NaN stays NaN.</summary>
    public static float RoundToE4M3(float x)
    {
        if (float.IsNaN(x)) return float.NaN;
        float a = MathF.Abs(x);
        if (a >= 448f) return MathF.CopySign(448f, x);
        int e = a >= MinNormal ? (int)(BitConverter.SingleToUInt32Bits(a) >> 23) - 127 : -127;
        float quantum = BitConverter.UInt32BitsToSingle((uint)(Math.Max(e, -6) - 3 + 127) << 23);
        return MathF.CopySign(MathF.Round(a / quantum, MidpointRounding.ToEven) * quantum, x);
    }

    /// <summary>The stochastic rounding threshold of P for (batch·Hq + head, query row, key): 24 hash bits, compared
    /// against the fraction of P scaled by 2²⁴.</summary>
    public static uint RoundingBits(uint seed, int batchHead, int row, int key) =>
        Mix(Mix(Mix(seed ^ 0x9E3779B9u ^ (uint)batchHead) ^ (uint)row) ^ (uint)key) >> 8;

    private static uint Mix(uint x)
    {
        x ^= x >> 16;
        x *= 0x7FEB352Du;
        x ^= x >> 15;
        x *= 0x846CA68Bu;
        return x ^ (x >> 16);
    }

    private static void Check(in AttentionCall call, Int8AttentionV2Options options)
    {
        Check(in call);
        if (options.BlockRows is not (64 or 128) || options.BlockCols is not (64 or 128))
            throw new ArgumentException(
                $"Tiles are 64 or 128 rows and keys; got {options.BlockRows}x{options.BlockCols}.", nameof(options));
        for (int i = 0; i < options.KeySegmentStarts.Count; i++)
            if (options.KeySegmentStarts[i] <= (i == 0 ? 0 : options.KeySegmentStarts[i - 1]))
                throw new ArgumentException("Key segment starts must be positive and ascending.", nameof(options));
    }

    private static void Check(in AttentionCall call)
    {
        AttentionHostReference.Validate(in call);
        if (call.Positions.OnDevice)
            throw new AttentionUnsupportedException("The INT8 attention v2 emulator reads host positions only.");
        if (call.QkHeadDim != call.ValueHeadDim)
            throw new AttentionUnsupportedException(
                $"INT8 attention v2 needs one head dim for Q, K and V; got {call.QkHeadDim} and {call.ValueHeadDim}.");
        PaddedHeadDim(call.QkHeadDim);
        if (call.Logits.AlibiSlopes is not null)
            throw new AttentionUnsupportedException("INT8 attention v2 carries no ALiBi slopes.");
        if (call.Logits.Sinks is { } sinks && sinks.DType != DType.F32)
            throw new AttentionUnsupportedException($"INT8 attention v2 reads F32 sinks, not {sinks.DType}.");
        // Validate admits only ranks 2-4 whose key extent is 1 or Skv, so StageBias reads inside the tensor.
        if (call.Mask.Bias is { } bias)
        {
            if (bias.DType != DType.F32)
                throw new AttentionUnsupportedException($"INT8 attention v2 reads an F32 bias, not {bias.DType}.");
            if (bias.Shape[bias.Shape.Rank - 2] != 1 && call.QueryTokens > 1)
                throw new AttentionUnsupportedException(
                    "INT8 attention v2 stages a per-key bias; a bias that varies over query rows is not one.");
        }
    }

    private static int[] RowList(ReadOnlySpan<int> rows, int queryTokens)
    {
        if (rows.IsEmpty) return Enumerable.Range(0, queryTokens).ToArray();
        for (int i = 0; i < rows.Length; i++)
            if (rows[i] < 0 || rows[i] >= queryTokens || i > 0 && rows[i] <= rows[i - 1])
                throw new ArgumentException($"Query rows must ascend inside [0, {queryTokens}); row {i} is {rows[i]}.",
                    nameof(rows));
        return rows.ToArray();
    }

    private static int[] SegmentBounds(Int8AttentionV2Options options, int kvLen)
    {
        int segments = options.Segments;
        int[] bounds = new int[segments + 1];
        for (int s = 1; s < segments; s++) bounds[s] = Math.Min(options.KeySegmentStarts[s - 1], kvLen);
        bounds[segments] = kvLen;
        return bounds;
    }

    private static void CopyScales(Int8AttentionV2Operands from, Int8AttentionV2Operands to)
    {
        if (from.QueryScales.Length != to.QueryScales.Length || from.KeyMeans.Length != to.KeyMeans.Length
            || from.KeyScales.Length != to.KeyScales.Length || from.ValueScales.Length != to.ValueScales.Length
            || !from.Rows.AsSpan().SequenceEqual(to.Rows))
            throw new ArgumentException("The supplied scales belong to another call, rows or variant.", nameof(from));
        foreach (float[] scales in (float[][])[from.QueryScales, from.KeyScales, from.ValueScales])
            foreach (float s in scales)
                if (s < MinNormal)
                    throw new ArgumentException(
                        $"Supplied scale {s}: a stats pass scale is NaN, +∞ or at least the smallest normal float.",
                        nameof(from));
        from.QueryScales.CopyTo(to.QueryScales, 0);
        if (to.Options.SmoothKeys) from.KeyMeans.CopyTo(to.KeyMeans, 0);
        from.KeyScales.CopyTo(to.KeyScales, 0);
        from.ValueScales.CopyTo(to.ValueScales, 0);
    }

    private static void SegmentMean(float[] k, int dp, int start, int end, Span<float> mean)
    {
        for (int c = 0; c < dp; c++)
        {
            float total = 0f;
            // Chunks start at the segment start, not at absolute multiples of 64: the stats pass chunks per segment.
            for (int chunk = start; chunk < end; chunk += MeanChunk)
            {
                float partial = 0f;
                for (int j = chunk; j < Math.Min(chunk + MeanChunk, end); j++) partial += k[j * dp + c];
                total += partial;
            }
            mean[c] = end > start ? total / (end - start) : 0f;
        }
    }

    private static void QuantizeKeys(Int8AttentionV2Operands ops, int kvh, float[] k, bool computeScales)
    {
        int dp = ops.PaddedHeadDim, kvLen = ops.ValidKeys, bc = ops.Options.BlockCols;
        bool perKey = ops.Options.KeyScales == Int8KeyScaleGrouping.PerKey;
        int perHead = perKey ? kvLen : ops.KeyTiles * 4;
        Span<float> scales = ops.KeyScales.AsSpan(kvh * perHead, perHead);
        if (computeScales)
        {
            if (perKey)
                for (int j = 0; j < kvLen; j++) scales[j] = Scale(AbsMax(k.AsSpan(j * dp, dp)), 127f);
            else
            {
                Span<float> absMax = stackalloc float[4];
                for (int t = 0; t < ops.KeyTiles; t++)
                {
                    absMax.Clear();
                    for (int j = t * bc; j < Math.Min(t * bc + bc, kvLen); j++)
                        absMax[(j & 7) >> 1] = MathF.Max(absMax[(j & 7) >> 1], AbsMax(k.AsSpan(j * dp, dp)));
                    for (int c = 0; c < 4; c++) scales[t * 4 + c] = Scale(absMax[c], 127f);
                }
            }
        }
        sbyte[] bytes = ops.Keys;
        for (int j = 0; j < kvLen; j++)
        {
            float inv = 1f / (perKey ? scales[j] : scales[j / bc * 4 + ((j & 7) >> 1)]);
            for (int c = 0; c < dp; c++) bytes[(kvh * kvLen + j) * dp + c] = ToS8(k[j * dp + c] * inv);
        }
    }

    private static void QuantizeValues(Int8AttentionV2Operands ops, int kvh, float[] v, bool computeScales)
    {
        int dp = ops.PaddedHeadDim, kvLen = ops.ValidKeys, blocks = ops.Options.ValueScalesPerTile ? ops.KeyTiles : 1;
        int block = ops.Options.ValueScalesPerTile ? ops.Options.BlockCols : Math.Max(1, kvLen);
        Int8ProbabilityFormat format = ops.Options.Probabilities;
        for (int t = 0; t < blocks; t++)
        {
            Span<float> scales = ops.ValueScales.AsSpan((kvh * blocks + t) * dp, dp);
            int j0 = t * block, j1 = Math.Min(j0 + block, kvLen);
            if (computeScales)
                for (int c = 0; c < dp; c++)
                {
                    float absMax = 0f;
                    for (int j = j0; j < j1; j++) absMax = MathF.Max(absMax, MathF.Abs(v[j * dp + c]));
                    scales[c] = format switch
                    {
                        Int8ProbabilityFormat.Fp8 => Scale(absMax, Fp8ValueRange),
                        Int8ProbabilityFormat.F16 => 1f,
                        _ => Scale(absMax, 127f),
                    };
                }
            for (int c = 0; c < dp; c++)
            {
                float inv = 1f / scales[c];
                for (int j = j0; j < j1; j++)
                {
                    int at = (kvh * kvLen + j) * dp + c;
                    float x = v[j * dp + c];
                    if (format == Int8ProbabilityFormat.Fp8) ops.NarrowValues[at] = RoundToE4M3(x * inv);
                    else if (format == Int8ProbabilityFormat.F16) ops.NarrowValues[at] = (float)(Half)x;
                    else ops.Values[at] = ToS8(x * inv);
                }
            }
        }
    }

    /// <summary><c>absmax / range</c>; NaN for a NaN absmax, 1 below the smallest normal float.</summary>
    private static float Scale(float absMax, float range)
    {
        if (float.IsNaN(absMax)) return float.NaN;
        float s = absMax / range;
        return s >= MinNormal ? s : 1f;
    }

    /// <summary>The largest |x|, NaN when any element is NaN.</summary>
    private static float AbsMax(ReadOnlySpan<float> x)
    {
        float m = 0f;
        foreach (float value in x) m = MathF.Max(m, MathF.Abs(value));
        return m;
    }

    private static sbyte ToS8(float x) => float.IsNaN(x) ? (sbyte)0 : (sbyte)MathF.Round(Math.Clamp(x, -127f, 127f));

    private static int ToU8(float x) => float.IsNaN(x) ? 0 : (int)MathF.Round(Math.Clamp(x, 0f, 255f));

    private static int ToU8Stochastic(float x, uint bits)
    {
        if (float.IsNaN(x)) return 0;
        float lo = MathF.Floor(Math.Clamp(x, 0f, 255f));
        return (int)lo + ((Math.Clamp(x, 0f, 255f) - lo) * 16777216f > bits ? 1 : 0);
    }

    private static float Exp2(float x) => (float)Math.Pow(2.0, x);

    private static unsafe void ReadRow(in AttentionOperand x, int b, int h, int t, Span<float> dst)
    {
        long i = x.Offset + b * x.BatchStride + h * x.HeadStride + t * x.TokenStride;
        int d = x.HeadDim;
        void* p = x.Storage.DataPointer;
        DType dt = x.Storage.DType;
        if (dt == DType.F32) new ReadOnlySpan<float>((float*)p + i, d).CopyTo(dst);
        else if (dt == DType.F16)
            for (int c = 0; c < d; c++) dst[c] = (float)BitConverter.UInt16BitsToHalf(((ushort*)p)[i + c]);
        else if (dt == DType.BF16)
            for (int c = 0; c < d; c++) dst[c] = BitConverter.UInt32BitsToSingle((uint)((ushort*)p)[i + c] << 16);
        else throw new AttentionUnsupportedException($"INT8 attention v2 reads F32, F16 and BF16 operands, not {dt}.");
        dst[d..].Clear();
    }

    /// <summary>One head's bias over keys <c>[key0, +staged.Length)</c> times log2(e); −∞ marks an exclusion.</summary>
    private static unsafe void StageBias(in AttentionCall call, int b, int h, int key0, Span<float> staged,
        Span<bool> excluded)
    {
        staged.Clear();
        excluded.Clear();
        if (call.Mask.Bias is not { } bias) return;
        TensorShape s = bias.Shape;
        int rank = s.Rank;
        long dk = s[rank - 1], dh = rank >= 3 ? s[rank - 3] : 1, db = rank >= 4 ? s[rank - 4] : 1, rows = s[rank - 2];
        long at = (db == 1 ? 0 : b * dh * rows * dk) + (dh == 1 ? 0 : h * rows * dk);
        float* p = (float*)bias.DataPointer + at;
        for (int j = 0; j < staged.Length; j++)
        {
            float x = p[dk == 1 ? 0 : key0 + j];
            excluded[j] = float.IsNegativeInfinity(x);
            staged[j] = excluded[j] ? 0f : x * Log2E;
        }
    }

    /// <summary>The keys a query position may see under the causal, window and block rules, inclusive.</summary>
    private static (long Lo, long Hi) KeyRange(in AttentionMask mask, long qpos)
    {
        long n = Math.Max(1, mask.BlockSize), qi = qpos / n;
        long lo = long.MinValue, hi = long.MaxValue;
        if (mask.Causal) hi = qi * n + n - 1;
        if (mask.WindowLeft is { } left) lo = (qi - left) * n;
        if (mask.WindowRight is { } right) hi = Math.Min(hi, (qi + right) * n + n - 1);
        return (lo, hi);
    }

    private static AttentionTileClass Classify(in AttentionMask mask, int queryOffset, int kvLen, int row0, int rowEnd,
        int key0, int keys, ReadOnlySpan<bool> excluded)
    {
        long keyEnd = (long)key0 + keys;
        int end = (int)Math.Min(keyEnd, kvLen);
        if (end <= key0 || rowEnd <= row0) return AttentionTileClass.Skip;
        bool anyExcluded = excluded.Contains(true);
        if (!excluded.Contains(false)) return AttentionTileClass.Skip;
        bool full = keyEnd <= kvLen && !anyExcluded, none = true;
        for (int r = row0; r < rowEnd; r++)
        {
            (long lo, long hi) = KeyRange(in mask, (long)queryOffset + r);
            if (Math.Max(lo, key0) <= Math.Min(hi, end - 1)) none = false;
            if (lo > key0 || hi < keyEnd - 1) full = false;
        }
        return none ? AttentionTileClass.Skip : full ? AttentionTileClass.Full : AttentionTileClass.Partial;
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Vector<float> acc = Vector<float>.Zero;
        int i = 0, w = Vector<float>.Count;
        for (; i <= a.Length - w; i += w) acc += new Vector<float>(a[i..]) * new Vector<float>(b[i..]);
        float sum = Vector.Sum(acc);
        for (; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static void AddScaled(float p, ReadOnlySpan<float> x, Span<float> y)
    {
        Vector<float> pv = new(p);
        int i = 0, w = Vector<float>.Count;
        for (; i <= x.Length - w; i += w) (new Vector<float>(y[i..]) + pv * new Vector<float>(x[i..])).CopyTo(y[i..]);
        for (; i < x.Length; i++) y[i] += p * x[i];
    }

    /// <summary>One call's recurrence: K/V per kv head, the bias per head, tile classes per query tile.</summary>
    private sealed class Recurrence
    {
        private readonly Int8AttentionV2Operands _ops;
        private readonly int _dp, _bc, _kvLen, _tiles, _segments;
        private readonly float _alpha, _alphaLog2e, _softcap;
        private readonly bool _u8, _stochastic, _fp8;
        private readonly Func<float, float> _exp2;
        private readonly float[] _sinks;
        private readonly int[] _segmentOf;
        private readonly float[] _keys, _values, _bias, _q, _t, _o, _partial, _shift;
        private readonly bool[] _excluded, _visible;
        private readonly AttentionTileClass[] _classes;

        public Recurrence(in AttentionCall call, Int8AttentionV2Operands ops)
        {
            _ops = ops;
            _dp = ops.PaddedHeadDim;
            _bc = ops.Options.BlockCols;
            _kvLen = ops.ValidKeys;
            _tiles = ops.KeyTiles;
            _segments = ops.Segments;
            _alpha = call.Scale;
            _alphaLog2e = call.Scale * Log2E;
            _softcap = call.Logits.Softcap > 0f ? call.Logits.Softcap : 0f;
            _u8 = ops.Options.Probabilities is Int8ProbabilityFormat.U8 or Int8ProbabilityFormat.U8Stochastic;
            _stochastic = ops.Options.Probabilities == Int8ProbabilityFormat.U8Stochastic;
            _fp8 = ops.Options.Probabilities == Int8ProbabilityFormat.Fp8;
            _exp2 = ops.Options.Exp2 ?? Exp2;
            _sinks = call.Logits.Sinks is { } sinks ? sinks.AsReadOnlySpan<float>()[..call.Heads].ToArray() : [];
            int[] bounds = SegmentBounds(ops.Options, _kvLen);
            _segmentOf = new int[_kvLen];
            for (int s = 0; s < _segments; s++) _segmentOf.AsSpan(bounds[s], bounds[s + 1] - bounds[s]).Fill(s);
            _keys = new float[Math.Max(1, _kvLen) * _dp];
            _values = new float[_keys.Length];
            _bias = new float[_kvLen];
            _excluded = new bool[_kvLen];
            _q = new float[_dp];
            _t = new float[_bc];
            _visible = new bool[_bc];
            _o = new float[_dp];
            _partial = new float[_dp];
            _shift = new float[_segments];
            _classes = new AttentionTileClass[_tiles];
        }

        public void Run(in AttentionCall call, Span<float> lse)
        {
            int group = call.KvGroup, br = _ops.Options.BlockRows;
            int[] rows = _ops.Rows;
            for (int b = 0; b < call.Batch; b++)
                for (int kh = 0; kh < call.KvHeads; kh++)
                {
                    LoadKv(b * call.KvHeads + kh);
                    for (int h = kh * group; h < (kh + 1) * group; h++)
                    {
                        StageBias(in call, b, h, 0, _bias, _excluded);
                        for (int ri = 0; ri < rows.Length;)
                        {
                            int tile0 = rows[ri] / br * br, tileEnd = Math.Min(tile0 + br, call.QueryTokens);
                            int qOffset = call.Positions.QueryOffset;
                            for (int t = 0; t < _tiles; t++)
                                _classes[t] = Classify(call.Mask, qOffset, _kvLen, tile0, tileEnd, t * _bc, _bc,
                                    _excluded.AsSpan(t * _bc, Math.Min(_bc, _kvLen - t * _bc)));
                            for (; ri < rows.Length && rows[ri] < tileEnd; ri++)
                            {
                                float rowLse = Row(call.Mask, qOffset, b, h, ri, rows[ri], call.Heads);
                                for (int c = 0; c < call.ValueHeadDim; c++)
                                    AttentionHostReference.Write(call.Output, b, h, rows[ri], c, _o[c]);
                                if (!lse.IsEmpty) lse[(b * call.Heads + h) * call.QueryTokens + rows[ri]] = rowLse;
                            }
                        }
                    }
                }
        }

        private void LoadKv(int kvh)
        {
            int n = _kvLen * _dp;
            ReadOnlySpan<sbyte> k = _ops.Keys.AsSpan(kvh * n, n);
            for (int i = 0; i < n; i++) _keys[i] = k[i];
            if (_u8)
            {
                ReadOnlySpan<sbyte> v = _ops.Values.AsSpan(kvh * n, n);
                for (int i = 0; i < n; i++) _values[i] = v[i];
            }
            else _ops.NarrowValues.AsSpan(kvh * n, n).CopyTo(_values);
        }

        /// <summary>Attends one query row into <c>_o</c>, divided and V-scaled; returns its natural-log LSE.</summary>
        private float Row(in AttentionMask mask, int queryOffset, int b, int h, int ri, int row, int heads)
        {
            int dp = _dp, kvh = b * _ops.KvHeads + h / (heads / _ops.KvHeads);
            int at = (b * heads + h) * _ops.Rows.Length + ri;
            ReadOnlySpan<sbyte> q8 = _ops.Queries.AsSpan(at * dp, dp);
            for (int c = 0; c < dp; c++) _q[c] = q8[c];
            float qScale = _ops.QueryScales[at];
            bool capped = _softcap > 0f;
            float qs = qScale * (capped ? _alpha : _alphaLog2e);
            // Scores leave out the first segment's smoothing offset unless a softcap needs it back.
            float frame = capped ? 0f : _ops.QueryMeanDots[at * _segments] * _alphaLog2e;
            for (int s = 0; s < _segments; s++)
            {
                float meanDot = _ops.QueryMeanDots[at * _segments + s];
                _shift[s] = capped ? meanDot * _alpha : meanDot * _alphaLog2e - frame;
            }
            (long lo, long hi) = KeyRange(in mask, (long)queryOffset + row);
            bool valid = false, poisoned = false;
            float m = 0f, d = 0f;
            Array.Clear(_o);
            int dotBound = QkDotBound(dp);
            for (int t = 0; t < _tiles && !poisoned; t++)
            {
                if (_classes[t] == AttentionTileClass.Skip) continue;
                int j0 = t * _bc, j1 = Math.Min(j0 + _bc, _kvLen), visible = 0;
                float tileMax = float.NegativeInfinity;
                for (int j = j0; j < j1; j++)
                {
                    bool seen = _classes[t] == AttentionTileClass.Full || j >= lo && j <= hi && !_excluded[j];
                    _visible[j - j0] = seen;
                    if (!seen) continue;
                    visible++;
                    float dot = Dot(_q, _keys.AsSpan(j * dp, dp));
                    if (MathF.Abs(dot) > dotBound)
                        throw new InvalidOperationException($"QK product {dot} exceeds the INT32 bound {dotBound}.");
                    float ks = _ops.Options.KeyScales == Int8KeyScaleGrouping.PerKey
                        ? _ops.KeyScales[kvh * _kvLen + j]
                        : _ops.KeyScales[(kvh * _tiles + t) * 4 + ((j & 7) >> 1)];
                    float score;
                    if (capped)
                    {
                        float x = MathF.FusedMultiplyAdd(dot, qs * ks, _shift[_segmentOf[j]]);
                        score = MathF.FusedMultiplyAdd(_softcap * (float)Math.Tanh(x / _softcap), Log2E, _bias[j]);
                    }
                    else score = MathF.FusedMultiplyAdd(dot, qs * ks, _shift[_segmentOf[j]] + _bias[j]);
                    _t[j - j0] = score;
                    tileMax = MathF.Max(tileMax, score);
                }
                if (visible == 0) continue;
                ReadOnlySpan<float> tileVScales =
                    _ops.Options.ValueScalesPerTile ? _ops.ValueScales.AsSpan((kvh * _tiles + t) * dp, dp) : [];
                if (!float.IsFinite(tileMax)) poisoned = true;
                else if (_u8) U8Tile(b * heads + h, row, j0, j1, tileMax, tileVScales, ref valid, ref m, ref d);
                else RunningTile(j0, j1, tileMax, tileVScales, ref valid, ref m, ref d);
            }
            // A −∞ sink carries no mass; a NaN or +∞ one poisons the row (contract §3).
            if (_sinks.Length > 0 && !poisoned && !float.IsNegativeInfinity(_sinks[h]))
            {
                float sink = _sinks[h] * Log2E - frame;
                if (!float.IsFinite(sink)) poisoned = true;
                else
                {
                    float sinkMax = sink - (_u8 ? U8Offset : _fp8 ? Fp8Offset : 0f);
                    float next = valid ? MathF.Max(m, sinkMax) : sinkMax;
                    float oScale = valid ? _exp2(m - next) : 1f;
                    d = d * oScale + _exp2(sink - next);
                    for (int c = 0; c < dp; c++) _o[c] *= oScale;
                    (m, valid) = (next, true);
                }
            }
            if (poisoned)
            {
                Array.Fill(_o, float.NaN);
                return float.NaN;
            }
            if (!valid) return float.NegativeInfinity;
            float inv = 1f / d;
            if (_ops.Options.ValueScalesPerTile)
                for (int c = 0; c < dp; c++) _o[c] *= inv;
            else
            {
                ReadOnlySpan<float> vScales = _ops.ValueScales.AsSpan(kvh * dp, dp);
                for (int c = 0; c < dp; c++) _o[c] = _o[c] * inv * vScales[c];
            }
            return (m + MathF.Log2(d) + frame) * Ln2;
        }

        /// <summary>u8 P against the tile's own maximum; numerator and denominator drain the same bytes.</summary>
        private void U8Tile(int batchHead, int row, int j0, int j1, float tileMax, ReadOnlySpan<float> vScales,
            ref bool valid, ref float m, ref float d)
        {
            float tileM = tileMax - U8Offset, oScale = 1f, tileScale = 1f;
            if (!valid) (m, valid) = (tileM, true);
            else
            {
                float smaller = _exp2(-MathF.Abs(m - tileM));
                oScale = m < tileM ? smaller : 1f;
                tileScale = tileM < m ? smaller : 1f;
                m = MathF.Max(m, tileM);
            }
            Array.Clear(_partial);
            int sum = 0;
            for (int j = j0; j < j1; j++)
            {
                if (!_visible[j - j0]) continue;
                float x = _exp2(_t[j - j0] - tileM);
                int p = _stochastic
                    ? ToU8Stochastic(x, RoundingBits(_ops.Options.RoundingSeed, batchHead, row, j))
                    : ToU8(x);
                if (p == 0) continue;
                sum += p;
                AddScaled(p, _values.AsSpan(j * _dp, _dp), _partial);
            }
            int bound = PvPartialBound(_bc);
            for (int c = 0; c < _dp; c++)
            {
                if (MathF.Abs(_partial[c]) > bound)
                    throw new InvalidOperationException($"PV partial {_partial[c]} exceeds the INT32 bound {bound}.");
                float scale = vScales.IsEmpty ? tileScale : tileScale * vScales[c];
                _o[c] = MathF.FusedMultiplyAdd(_partial[c], scale, _o[c] * oScale);
            }
            d = MathF.FusedMultiplyAdd(sum, tileScale, d * oScale);
        }

        /// <summary>E4M3 or F16 P against the running maximum; the partial sums in key order and drains per tile.</summary>
        private void RunningTile(int j0, int j1, float tileMax, ReadOnlySpan<float> vScales, ref bool valid,
            ref float m, ref float d)
        {
            float tileM = tileMax - (_fp8 ? Fp8Offset : 0f), oScale = 1f;
            if (!valid) (m, valid) = (tileM, true);
            else
            {
                float next = MathF.Max(m, tileM);
                oScale = _exp2(m - next);
                m = next;
            }
            Array.Clear(_partial);
            float sum = 0f;
            for (int j = j0; j < j1; j++)
            {
                if (!_visible[j - j0]) continue;
                float x = _exp2(_t[j - j0] - m);
                float p = _fp8 ? RoundToE4M3(x) : (float)(Half)x;
                sum += p;
                AddScaled(p, _values.AsSpan(j * _dp, _dp), _partial);
            }
            for (int c = 0; c < _dp; c++)
                _o[c] = vScales.IsEmpty
                    ? _o[c] * oScale + _partial[c]
                    : MathF.FusedMultiplyAdd(_partial[c], vScales[c], _o[c] * oScale);
            d = d * oScale + sum;
        }
    }
}
