namespace HartsyInference.Core.Backends;

/// <summary>INT8 attention v2's stats pass and quant launch for one call: the bytes, scales and means its attention
/// kernel reads, in logical layouts (the kernel's physical layouts permute them). A GPU's buffers can be copied in.</summary>
public sealed class Int8AttentionV2Operands
{
    internal Int8AttentionV2Operands(Int8AttentionV2Options options, int batch, int heads, int kvHeads, int validKeys,
        int headDim, int[] rows)
    {
        Options = options;
        Batch = batch;
        Heads = heads;
        KvHeads = kvHeads;
        ValidKeys = validKeys;
        HeadDim = headDim;
        PaddedHeadDim = Int8AttentionV2Reference.PaddedHeadDim(headDim);
        KeyTiles = (validKeys + options.BlockCols - 1) / options.BlockCols;
        Segments = options.Segments;
        Rows = rows;
        int dp = PaddedHeadDim;
        long kvHeadsTotal = (long)batch * kvHeads, rowsTotal = (long)batch * heads * rows.Length;
        Queries = new sbyte[checked((int)(rowsTotal * dp))];
        QueryScales = new float[checked((int)rowsTotal)];
        QueryMeanDots = new float[checked((int)(rowsTotal * Segments))];
        KeyMeans = new float[checked((int)(kvHeadsTotal * Segments * dp))];
        Keys = new sbyte[checked((int)(kvHeadsTotal * validKeys * dp))];
        long keyScales = options.KeyScales == Int8KeyScaleGrouping.ThreadClass ? KeyTiles * 4L : validKeys;
        KeyScales = new float[checked((int)(kvHeadsTotal * keyScales))];
        bool u8 = options.Probabilities is Int8ProbabilityFormat.U8 or Int8ProbabilityFormat.U8Stochastic;
        Values = u8 ? new sbyte[Keys.Length] : [];
        NarrowValues = u8 ? [] : new float[Keys.Length];
        ValueScales = new float[checked((int)(kvHeadsTotal * (options.ValueScalesPerTile ? KeyTiles : 1) * dp))];
    }

    /// <summary>The variant these operands were quantized for.</summary>
    public Int8AttentionV2Options Options { get; }

    /// <summary>Batch.</summary>
    public int Batch { get; }

    /// <summary>Query heads.</summary>
    public int Heads { get; }

    /// <summary>Key/value heads.</summary>
    public int KvHeads { get; }

    /// <summary>Keys attended (the call's valid key length); later key rows are never read.</summary>
    public int ValidKeys { get; }

    /// <summary>The call's head dim.</summary>
    public int HeadDim { get; }

    /// <summary>The kernel's head dim: 64, 128 or 256, the extra channels zero.</summary>
    public int PaddedHeadDim { get; }

    /// <summary>Key tiles of <see cref="Int8AttentionV2Options.BlockCols"/> keys over the valid keys.</summary>
    public int KeyTiles { get; }

    /// <summary>Key segments, each smoothed with its own mean.</summary>
    public int Segments { get; }

    /// <summary>The query rows quantized, ascending.</summary>
    public int[] Rows { get; }

    /// <summary>Q bytes <c>[B, Hq, rows, Dp]</c>.</summary>
    public sbyte[] Queries { get; }

    /// <summary>Per-row Q scales <c>absmax / 127</c>, <c>[B, Hq, rows]</c>; <c>α·log2e</c> folds into them.</summary>
    public float[] QueryScales { get; }

    /// <summary><c>q · μ_s</c> of each unquantized Q row and segment K mean, <c>[B, Hq, rows, segments]</c>.</summary>
    public float[] QueryMeanDots { get; }

    /// <summary>K channel mean per segment, <c>[B, Hkv, segments, Dp]</c>; zero when keys are not smoothed.</summary>
    public float[] KeyMeans { get; }

    /// <summary>Bytes of the smoothed K, <c>[B, Hkv, kvLen, Dp]</c>.</summary>
    public sbyte[] Keys { get; }

    /// <summary>K scales: <c>[B, Hkv, tiles, 4]</c> by thread class, or <c>[B, Hkv, kvLen]</c> per key.</summary>
    public float[] KeyScales { get; }

    /// <summary>s8 V bytes <c>[B, Hkv, kvLen, Dp]</c> for the u8 formats; empty otherwise.</summary>
    public sbyte[] Values { get; }

    /// <summary>V over its channel scale in E4M3 or F16, <c>[B, Hkv, kvLen, Dp]</c>, for those formats.</summary>
    public float[] NarrowValues { get; }

    /// <summary>Per-channel V scales <c>[B, Hkv, Dp]</c>, or <c>[B, Hkv, tiles, Dp]</c> per key tile: <c>absmax / 127</c>
    /// (u8), <c>absmax / 2.25</c> (E4M3), 1 (F16).</summary>
    public float[] ValueScales { get; }

    /// <summary>The index of query row <paramref name="row"/> in <see cref="Rows"/>.</summary>
    /// <exception cref="ArgumentException">The row was not quantized.</exception>
    public int RowIndex(int row)
    {
        int i = Array.BinarySearch(Rows, row);
        return i >= 0 ? i : throw new ArgumentException($"Query row {row} was not quantized.", nameof(row));
    }
}
