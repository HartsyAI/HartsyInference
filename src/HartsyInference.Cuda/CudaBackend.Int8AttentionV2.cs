using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// INT8 attention v2 rows (S.16/S.17, M.4/M.5): admission, the persistent workspace and the three launches.
public sealed partial class CudaBackend
{
    private ulong _int8V2Workspace;
    private long _int8V2WorkspaceBytes;
    private long _int8V2ExecutionCount;

    /// <summary>Number of INT8 attention v2 executions this session.</summary>
    public long Int8AttentionV2ExecutionCount => Interlocked.Read(ref _int8V2ExecutionCount);

    /// <summary>Test and benchmark hook: query rows per INT8 v2 attention block, 64 or 128; 0 follows the card.</summary>
    internal int Int8V2BlockRowsOverride { get; set; }

    /// <summary>Test and benchmark hook: admits the F16-PV rows instead of the u8 ones.</summary>
    internal bool Int8V2F16Pv { get; set; }

    /// <summary>Query rows per INT8 v2 attention block on this card.</summary>
    internal int Int8V2BlockRows => Int8V2BlockRowsOverride is 64 or 128 ? Int8V2BlockRowsOverride
        : _context.Sm is >= CudaArch.Ampere and < 89 ? 64 : 128;

    /// <summary>Bytes the INT8 v2 workspace holds now (its high-water mark since it was last released).</summary>
    internal long Int8V2WorkspaceBytes => _int8V2WorkspaceBytes;

    /// <summary>What the INT8 v2 rows take beyond their declared capability: the knob, F16 or F32 throughout, a positive scale,
    /// both sequences at the engage length, 16-byte aligned rows and the kernels.</summary>
    private bool Int8V2Admits(in AttentionCall c, bool f16Pv)
    {
        if (!EngineKnobs.Int8AttnV2.Value || f16Pv != Int8V2F16Pv || c.Mask.Bias is not null || c.Positions.OnDevice) return false;
        DType dt = c.Query.DType;
        if (dt != DType.F16 && dt != DType.F32 || c.Key.DType != dt || c.Value.DType != dt || c.Output.DType != dt) return false;
        if (c.ValueHeadDim != Int8AttentionV2Layout.HeadDim || !(c.Scale > 0f) || !float.IsFinite(c.Scale)) return false;
        if (Math.Min(c.QueryTokens, c.ValidKvLength) < EngineKnobs.Int8AttnV2MinTokens.Value) return false;
        if (!CudaKernels.GridYzFits(c.Heads, c.Batch) || (long)c.Batch * c.Heads > Int8AttentionV2Layout.CounterSlots) return false;
        if (!Int8V2RowsAligned(c.Query) || !Int8V2RowsAligned(c.Key) || !Int8V2RowsAligned(c.Value) || !Int8V2RowsAligned(c.Output)
            || !c.Output.CoversStorage) return false;
        EnsureKernels();
        return f16Pv ? _kernels!.HasInt8AttentionV2F16Pv : _kernels!.HasInt8AttentionV2;
    }

    /// <summary>Whether a head-major call of this shape with no mask would take S.16; the grouped token-major row yields to it.</summary>
    private bool Int8V2TakesWidened(in AttentionCall c)
    {
        if (!EngineKnobs.Int8AttnV2.Value || Int8V2F16Pv || c.Mask.Bias is not null || c.QkHeadDim != Int8AttentionV2Layout.HeadDim
            || c.ValueHeadDim != Int8AttentionV2Layout.HeadDim || (c.Query.DType != DType.F16 && c.Query.DType != DType.F32) || !(c.Scale > 0f)
            || Math.Min(c.QueryTokens, c.KeyTokens) < EngineKnobs.Int8AttnV2MinTokens.Value || !CudaKernels.GridYzFits(c.Heads, c.Batch))
            return false;
        EnsureKernels();
        return _kernels!.HasInt8AttentionV2;
    }

    /// <summary>Offset and strides in whole 8-element steps: every row the quant and epilogue touch starts 16-byte aligned.</summary>
    private static bool Int8V2RowsAligned(in AttentionOperand o) =>
        o.Offset % 8 == 0 && o.TokenStride % 8 == 0 && (o.Heads == 1 || o.HeadStride % 8 == 0) && (o.Batch == 1 || o.BatchStride % 8 == 0);

    /// <summary>Runs the call; false when a capture finds the workspace too small (the next row runs).</summary>
    private bool RunInt8AttentionV2(in AttentionCall c, bool f16Pv)
    {
        using OpScope _op = EnterOp();
        EnsureKernels();
        Int8AttentionV2Pv pv = f16Pv ? Int8AttentionV2Pv.F16
            : EngineKnobs.Int8AttnV2VTile.Value ? Int8AttentionV2Pv.U8TileScales : Int8AttentionV2Pv.U8SequenceScales;
        Int8AttentionV2Layout layout = new(c.Batch, c.Heads, c.QueryTokens, c.ValidKvLength, pv);
        ulong workspace = EnsureInt8V2Workspace(layout.Bytes);
        if (workspace == 0) return false;
        ulong pQ = 0, pK = 0, pV = 0, pOut = 0;
        bool cached = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(c.Query.Storage);
            pK = GpuTransferHelper.CopyToDevice(c.Key.Storage);
            pV = GpuTransferHelper.CopyToDevice(c.Value.Storage);
            Tensor output = c.Output.Storage;
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            LaunchInt8V2(in c, layout, workspace, pQ, pK, pV, pOut, Int8V2BlockRows, null);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
            Interlocked.Increment(ref _int8V2ExecutionCount);
            return true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    /// <summary>The stats, quant and attention launches of one call over device operands.</summary>
    private void LaunchInt8V2(in AttentionCall c, in Int8AttentionV2Layout layout, ulong workspace, ulong pQ, ulong pK, ulong pV,
        ulong pOut, int blockRows, Int8V2TraceArgs? trace)
    {
        bool f32 = c.Query.DType == DType.F32, f32Out = c.Output.DType == DType.F32;
        ulong elem = f32 ? 4UL : 2UL;
        ulong q = pQ + (ulong)c.Query.Offset * elem, k = pK + (ulong)c.Key.Offset * elem, v = pV + (ulong)c.Value.Offset * elem;
        ulong o = pOut + (ulong)c.Output.Offset * (f32Out ? 4UL : 2UL);
        int keys = c.ValidKvLength;
        using (NvtxRange _ = NvtxRange.PushFine("Int8V2.Stats"))
            _kernels!.LaunchInt8V2Stats(f32, k, Strides(c.Key), v, Strides(c.Value), workspace, in layout, c.Batch, c.Heads, keys,
                _stream.Handle);
        using (NvtxRange _ = NvtxRange.PushFine("Int8V2.Quant"))
            _kernels!.LaunchInt8V2Quant(f32, q, Strides(c.Query), k, Strides(c.Key), v, Strides(c.Value), workspace, in layout, c.Batch,
                c.Heads, c.QueryTokens, keys, _stream.Handle);
        using (NvtxRange _ = NvtxRange.PushFine("Int8V2.Attention"))
            _kernels!.LaunchInt8V2Attention(blockRows, f32Out, o, Strides(c.Output), workspace, in layout, c.Batch, c.Heads, c.QueryTokens,
                keys, c.Scale * Int8AttentionV2Reference.Log2E, _stream.Handle, trace);
    }

    private static CudaKernels.FlashStrides Strides(in AttentionOperand o) =>
        new((ulong)o.BatchStride, (ulong)o.HeadStride, (ulong)o.TokenStride);

    /// <summary>The workspace grown to at least <paramref name="bytes"/>, or 0 while the stream captures and it is smaller: a
    /// capture never allocates it, and a captured graph must never hold a pointer a later growth frees.</summary>
    private ulong EnsureInt8V2Workspace(long bytes)
    {
        if (bytes <= _int8V2WorkspaceBytes) return _int8V2Workspace;
        if (StreamIsCapturing()) return 0;
        StepGraphInvalidateForActivationFree();
        ReleaseInt8V2Workspace();
        ulong buffer = GpuTransferHelper.AllocateDevice((nuint)bytes);
        CudaDriverApi.cuMemsetD8Async(buffer, 0, (nuint)(Int8AttentionV2Layout.CounterSlots * sizeof(uint)), _stream.Handle).ThrowOnError();
        _int8V2Workspace = buffer;
        _int8V2WorkspaceBytes = bytes;
        return buffer;
    }

    /// <summary>Returns the workspace to the stream-ordered pool; the next call grows it again.</summary>
    private void ReleaseInt8V2Workspace()
    {
        ulong buffer = _int8V2Workspace;
        _int8V2Workspace = 0;
        _int8V2WorkspaceBytes = 0;
        if (buffer != 0) GpuTransferHelper.FreeDevice(buffer);
    }

    /// <summary>Test hook: runs one call on <paramref name="pv"/> at <paramref name="blockRows"/> and copies its workspace back, with
    /// the trace of (batch, head, query tile) when <paramref name="traceTarget"/> is set (u8 PV, F32 output).</summary>
    internal unsafe Int8V2Readback RunInt8AttentionV2ForTest(in AttentionCall c, Int8AttentionV2Pv pv, int blockRows,
        (int Batch, int Head, int QueryTile)? traceTarget = null)
    {
        using OpScope _op = EnterOp();
        EnsureKernels();
        Int8AttentionV2Layout layout = new(c.Batch, c.Heads, c.QueryTokens, c.ValidKvLength, pv);
        ulong workspace = EnsureInt8V2Workspace(layout.Bytes);
        if (workspace == 0) throw new InvalidOperationException("The INT8 v2 workspace cannot grow while the stream captures.");
        ulong pQ = 0, pK = 0, pV = 0, pOut = 0, traceBuffer = 0;
        long kvPad = layout.KvPad, rows = blockRows, tiles = layout.Tiles;
        long dotsBytes = rows * kvPad * 4, pBytes = rows * kvPad, maxBytes = rows * tiles * 4, partialBytes = rows * tiles * 128 * 4;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(c.Query.Storage);
            pK = GpuTransferHelper.CopyToDevice(c.Key.Storage);
            pV = GpuTransferHelper.CopyToDevice(c.Value.Storage);
            nuint outBytes = GpuTransferHelper.ByteSize(c.Output.Storage);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            Int8V2TraceArgs? trace = null;
            if (traceTarget is { } target)
            {
                traceBuffer = GpuTransferHelper.AllocateDevice((nuint)(dotsBytes + pBytes + 2 * maxBytes + partialBytes));
                CudaDriverApi.cuMemsetD8Async(traceBuffer, 0, (nuint)(dotsBytes + pBytes + 2 * maxBytes + partialBytes), _stream.Handle).ThrowOnError();
                trace = new Int8V2TraceArgs
                {
                    Dots = traceBuffer, Probabilities = traceBuffer + (ulong)dotsBytes, TileMax = traceBuffer + (ulong)(dotsBytes + pBytes),
                    Sums = traceBuffer + (ulong)(dotsBytes + pBytes + maxBytes), Partials = traceBuffer + (ulong)(dotsBytes + pBytes + 2 * maxBytes),
                    Batch = (uint)target.Batch, Head = (uint)target.Head, QueryTile = (uint)target.QueryTile,
                };
            }
            LaunchInt8V2(in c, layout, workspace, pQ, pK, pV, pOut, blockRows, trace);
            _stream.Synchronize();
            T[] Read<T>(ulong at, long count) where T : unmanaged
            {
                T[] host = new T[count];
                fixed (T* p = host)
                    CudaDriverApi.cuMemcpyDtoH((nint)p, at, (nuint)(count * sizeof(T))).ThrowOnError();
                return host;
            }
            long bh = (long)c.Batch * c.Heads;
            byte[] output = Read<byte>(pOut, (long)outBytes);
            Int8V2Readback result = new()
            {
                Layout = layout,
                Output = output,
                Q8 = Read<sbyte>(workspace + (ulong)layout.Q8, bh * c.QueryTokens * 128),
                QScale = Read<float>(workspace + (ulong)layout.QScale, bh * c.QueryTokens),
                K8 = Read<sbyte>(workspace + (ulong)layout.K8, bh * kvPad * 128),
                KScale = Read<float>(workspace + (ulong)layout.KScale, bh * tiles * 4),
                V8 = pv == Int8AttentionV2Pv.F16 ? [] : Read<sbyte>(workspace + (ulong)layout.V8, bh * 128 * kvPad),
                VScale = pv == Int8AttentionV2Pv.F16 ? []
                    : Read<float>(workspace + (ulong)layout.VScale, bh * (pv == Int8AttentionV2Pv.U8TileScales ? tiles : 1) * 128),
                KMean = Read<float>(workspace + (ulong)layout.KMean, bh * 128),
                Dots = trace is null ? [] : Read<int>(traceBuffer, rows * kvPad),
                Probabilities = trace is null ? [] : Read<byte>(traceBuffer + (ulong)dotsBytes, rows * kvPad),
                TileMax = trace is null ? [] : Read<float>(traceBuffer + (ulong)(dotsBytes + pBytes), rows * tiles),
                Sums = trace is null ? [] : Read<int>(traceBuffer + (ulong)(dotsBytes + pBytes + maxBytes), rows * tiles),
                Partials = trace is null ? [] : Read<int>(traceBuffer + (ulong)(dotsBytes + pBytes + 2 * maxBytes), rows * tiles * 128),
            };
            return result;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(traceBuffer);
        }
    }

    /// <summary>Benchmark hook: mean device milliseconds of the stats, quant and attention launches over
    /// <paramref name="iterations"/> runs of one call on device-resident operands.</summary>
    internal float[] Int8V2StageTimesForTest(in AttentionCall c, Int8AttentionV2Pv pv, int blockRows, int iterations)
    {
        using OpScope _op = EnterOp();
        EnsureKernels();
        Int8AttentionV2Layout layout = new(c.Batch, c.Heads, c.QueryTokens, c.ValidKvLength, pv);
        ulong workspace = EnsureInt8V2Workspace(layout.Bytes);
        if (workspace == 0) throw new InvalidOperationException("The INT8 v2 workspace cannot grow while the stream captures.");
        nint[] events = new nint[4];
        ulong pQ = 0, pK = 0, pV = 0, pOut = 0;
        float[] totals = new float[3];
        try
        {
            for (int i = 0; i < events.Length; i++) CudaDriverApi.cuEventCreate(out events[i], 0).ThrowOnError();
            pQ = GpuTransferHelper.CopyToDevice(c.Query.Storage);
            pK = GpuTransferHelper.CopyToDevice(c.Key.Storage);
            pV = GpuTransferHelper.CopyToDevice(c.Value.Storage);
            pOut = GpuTransferHelper.AllocateDevice(GpuTransferHelper.ByteSize(c.Output.Storage));
            bool f32 = c.Query.DType == DType.F32, f32Out = c.Output.DType == DType.F32;
            ulong elem = f32 ? 4UL : 2UL;
            ulong q = pQ + (ulong)c.Query.Offset * elem, k = pK + (ulong)c.Key.Offset * elem, v = pV + (ulong)c.Value.Offset * elem;
            ulong o = pOut + (ulong)c.Output.Offset * (f32Out ? 4UL : 2UL);
            for (int it = -2; it < iterations; it++)
            {
                CudaDriverApi.cuEventRecord(events[0], _stream.Handle).ThrowOnError();
                _kernels!.LaunchInt8V2Stats(f32, k, Strides(c.Key), v, Strides(c.Value), workspace, in layout, c.Batch, c.Heads,
                    c.ValidKvLength, _stream.Handle);
                CudaDriverApi.cuEventRecord(events[1], _stream.Handle).ThrowOnError();
                _kernels.LaunchInt8V2Quant(f32, q, Strides(c.Query), k, Strides(c.Key), v, Strides(c.Value), workspace, in layout, c.Batch,
                    c.Heads, c.QueryTokens, c.ValidKvLength, _stream.Handle);
                CudaDriverApi.cuEventRecord(events[2], _stream.Handle).ThrowOnError();
                _kernels.LaunchInt8V2Attention(blockRows, f32Out, o, Strides(c.Output), workspace, in layout, c.Batch, c.Heads,
                    c.QueryTokens, c.ValidKvLength, c.Scale * Int8AttentionV2Reference.Log2E, _stream.Handle);
                CudaDriverApi.cuEventRecord(events[3], _stream.Handle).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(events[3]).ThrowOnError();
                if (it < 0) continue;
                for (int s = 0; s < 3; s++)
                {
                    CudaDriverApi.cuEventElapsedTime(out float ms, events[s], events[s + 1]).ThrowOnError();
                    totals[s] += ms / iterations;
                }
            }
            return totals;
        }
        finally
        {
            foreach (nint e in events)
                if (e != 0) CudaDriverApi.cuEventDestroy(e);
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pOut);
        }
    }

    /// <summary>Test hook: the attention kernels' exp2 of each argument.</summary>
    internal unsafe float[] Int8V2Exp2ForTest(float[] arguments)
    {
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (arguments.Length == 0) return [];
        nuint bytes = (nuint)(arguments.Length * sizeof(float));
        ulong input = GpuTransferHelper.AllocateDevice(bytes), output = GpuTransferHelper.AllocateDevice(bytes);
        try
        {
            fixed (float* p = arguments) CudaDriverApi.cuMemcpyHtoDAsync(input, (nint)p, bytes, _stream.Handle).ThrowOnError();
            _kernels!.LaunchInt8V2Ex2Probe(output, input, arguments.Length, _stream.Handle);
            _stream.Synchronize();
            float[] result = new float[arguments.Length];
            fixed (float* p = result) CudaDriverApi.cuMemcpyDtoH((nint)p, output, bytes).ThrowOnError();
            return result;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(input);
            GpuTransferHelper.FreeDevice(output);
        }
    }
}

/// <summary>One INT8 v2 call's workspace and output copied back for a test: logical layouts per <see cref="Int8AttentionV2Layout"/>.</summary>
internal sealed record Int8V2Readback
{
    /// <summary>The layout the call ran.</summary>
    public required Int8AttentionV2Layout Layout { get; init; }

    /// <summary>The output tensor's bytes.</summary>
    public required byte[] Output { get; init; }

    /// <summary>Q bytes <c>[B·H, Sq, 128]</c>.</summary>
    public required sbyte[] Q8 { get; init; }

    /// <summary>Q scales <c>[B·H, Sq]</c>.</summary>
    public required float[] QScale { get; init; }

    /// <summary>K bytes <c>[B·H, kvPad, 128]</c>.</summary>
    public required sbyte[] K8 { get; init; }

    /// <summary>K scales <c>[B·H, tiles, 4]</c>.</summary>
    public required float[] KScale { get; init; }

    /// <summary>V bytes <c>[B·H, 128, kvPad]</c>, keys permuted within 32-key chunks; empty for F16 PV.</summary>
    public required sbyte[] V8 { get; init; }

    /// <summary>V scales <c>[B·H, 128]</c> or <c>[B·H, tiles, 128]</c>; empty for F16 PV.</summary>
    public required float[] VScale { get; init; }

    /// <summary>K channel means <c>[B·H, 128]</c>.</summary>
    public required float[] KMean { get; init; }

    /// <summary>Traced INT32 QK products <c>[BR, kvPad]</c>; empty without a trace.</summary>
    public required int[] Dots { get; init; }

    /// <summary>Traced P bytes <c>[BR, kvPad]</c>.</summary>
    public required byte[] Probabilities { get; init; }

    /// <summary>Traced tile score maxima <c>[BR, tiles]</c>.</summary>
    public required float[] TileMax { get; init; }

    /// <summary>Traced byte sums <c>[BR, tiles]</c>.</summary>
    public required int[] Sums { get; init; }

    /// <summary>Traced INT32 PV partials <c>[BR, tiles, 128]</c>.</summary>
    public required int[] Partials { get; init; }
}
