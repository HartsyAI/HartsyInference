namespace HartsyInference.Cuda;

/// <summary>Where INT8 attention v2's buffers sit in its workspace for one call shape (Kernels/attention/int8_attn_v2.cu).</summary>
/// <remarks>The stats launch's completion counters sit at offset 0 for every shape; each call zeroes the slots it uses before
/// its stats launch.</remarks>
internal readonly record struct Int8AttentionV2Layout
{
    /// <summary>The head dim the kernels run.</summary>
    public const int HeadDim = 128;

    /// <summary>Keys per tile (BC).</summary>
    public const int BlockCols = 128;

    /// <summary>Keys per partial sum of the K channel mean.</summary>
    public const int MeanChunk = 64;

    /// <summary>Completion counters at the head of the workspace, one per (batch, head).</summary>
    public const int CounterSlots = 65_536;

    private const long Align = 256;

    /// <summary>Lays out the buffers of a call with <paramref name="batch"/> × <paramref name="heads"/> heads,
    /// <paramref name="queryTokens"/> query rows and <paramref name="validKeys"/> keys.</summary>
    public Int8AttentionV2Layout(int batch, int heads, int queryTokens, int validKeys, Int8AttentionV2Pv pv)
    {
        if (batch <= 0 || heads <= 0 || queryTokens <= 0 || validKeys <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), $"INT8 attention v2 needs positive dims; got B={batch} H={heads} Sq={queryTokens} keys={validKeys}.");
        if ((long)batch * heads > CounterSlots)
            throw new ArgumentOutOfRangeException(nameof(heads), $"INT8 attention v2 runs at most {CounterSlots} (batch, head) pairs; got {batch}x{heads}.");
        Pv = pv;
        Tiles = (validKeys + BlockCols - 1) / BlockCols;
        Chunks = (validKeys + MeanChunk - 1) / MeanChunk;
        long bh = (long)batch * heads, kvPad = (long)Tiles * BlockCols;
        KvPad = kvPad;
        long at = CounterSlots * sizeof(uint);
        long Take(ref long cursor, long bytes)
        {
            long start = cursor;
            cursor = (cursor + bytes + Align - 1) / Align * Align;
            return start;
        }
        Q8 = Take(ref at, bh * queryTokens * HeadDim);
        QScale = Take(ref at, bh * queryTokens * sizeof(float));
        K8 = Take(ref at, bh * kvPad * HeadDim);
        KScale = Take(ref at, bh * kvPad * sizeof(float));
        V8 = Take(ref at, bh * HeadDim * kvPad * (pv == Int8AttentionV2Pv.F16 ? sizeof(ushort) : 1));
        VScale = Take(ref at, bh * (pv == Int8AttentionV2Pv.U8TileScales ? Tiles : 1) * HeadDim * sizeof(float));
        KMean = Take(ref at, bh * HeadDim * sizeof(float));
        KPartial = Take(ref at, bh * Chunks * HeadDim * sizeof(float));
        VPartial = Take(ref at, pv == Int8AttentionV2Pv.U8SequenceScales ? bh * Chunks * HeadDim * sizeof(float) : 0);
        Bytes = at;
    }

    /// <summary>Where the u8 V layout stores logical key <paramref name="key"/> of a tile: within each 32-key chunk,
    /// <c>16h + 8s + 2t + e</c> goes to <c>16h + 4t + 2s + e</c>, so a lane's four QK n-tiles form its PV A fragment.</summary>
    public static int PhysicalKey(int key) =>
        (key & ~31) | (key & 16) | (((key >> 1) & 3) << 2) | (((key >> 3) & 1) << 1) | (key & 1);

    /// <summary>The PV format the layout was sized for.</summary>
    public Int8AttentionV2Pv Pv { get; }

    /// <summary>Key tiles of <see cref="BlockCols"/>.</summary>
    public int Tiles { get; }

    /// <summary>Mean chunks of <see cref="MeanChunk"/> keys.</summary>
    public int Chunks { get; }

    /// <summary>Keys rounded up to whole tiles: the K and V buffers' key extent.</summary>
    public long KvPad { get; }

    /// <summary>Q bytes <c>[B·H, Sq, 128]</c>.</summary>
    public long Q8 { get; }

    /// <summary>Q scales <c>[B·H, Sq]</c>.</summary>
    public long QScale { get; }

    /// <summary>Smoothed K bytes <c>[B·H, kvPad, 128]</c>.</summary>
    public long K8 { get; }

    /// <summary>K scales, one per key <c>[B·H, kvPad]</c>.</summary>
    public long KScale { get; }

    /// <summary>V bytes <c>[B·H, 128, kvPad]</c>, keys permuted within 32-key chunks (F16 V in key order for the F16 PV).</summary>
    public long V8 { get; }

    /// <summary>V scales <c>[B·H, 128]</c>, or <c>[B·H, tiles, 128]</c> per tile.</summary>
    public long VScale { get; }

    /// <summary>K channel means <c>[B·H, 128]</c>.</summary>
    public long KMean { get; }

    /// <summary>K channel sums per mean chunk <c>[B·H, chunks, 128]</c>.</summary>
    public long KPartial { get; }

    /// <summary>V channel absmax per mean chunk <c>[B·H, chunks, 128]</c> (sequence scales only).</summary>
    public long VPartial { get; }

    /// <summary>Workspace bytes, counters included.</summary>
    public long Bytes { get; }
}
