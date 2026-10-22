using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// INT8 attention v2 rows (S.17/S.18, M.4/M.5): admission, the persistent workspace and the three launches.
public sealed partial class CudaBackend
{
    private ulong _int8V2Workspace;
    private long _int8V2WorkspaceBytes;
    private long _int8V2ExecutionCount;

    /// <summary>Number of INT8 attention v2 executions this session.</summary>
    public long Int8AttentionV2ExecutionCount => Interlocked.Read(ref _int8V2ExecutionCount);

    /// <summary>Query rows per INT8 v2 attention block on this card.</summary>
    internal int Int8V2BlockRows => Int8V2BlockRowsOverride is 64 or 128 ? Int8V2BlockRowsOverride
        : _context.Sm is >= CudaArch.Ampere and < 89 ? 64 : 128;

    /// <summary>Bytes the INT8 v2 workspace holds now (its high-water mark since it was last released).</summary>
    internal long Int8V2WorkspaceBytes => _int8V2WorkspaceBytes;

    /// <summary>The checks the INT8 v2 rows and the grouped row's yield share: the knob, no bias, host positions, F16 or F32
    /// throughout, head dim 128, a finite positive scale, both sequences at the engage length, and a grid and counter slot per
    /// (batch, head).</summary>
    private static bool Int8V2Eligible(in AttentionCall c) =>
        EngineKnobs.Int8AttnV2.Value && c.Mask.Bias is null && !c.Positions.OnDevice
        && (c.Query.DType == DType.F16 || c.Query.DType == DType.F32) && c.Key.DType == c.Query.DType && c.Value.DType == c.Query.DType
        && c.Output.DType == c.Query.DType && c.QkHeadDim == Int8AttentionV2Layout.HeadDim && c.ValueHeadDim == Int8AttentionV2Layout.HeadDim
        && c.Scale > 0f && float.IsFinite(c.Scale) && Math.Min(c.QueryTokens, c.ValidKvLength) >= EngineKnobs.Int8AttnV2MinTokens.Value
        && CudaKernels.GridYzFits(c.Heads, c.Batch) && (long)c.Batch * c.Heads <= Int8AttentionV2Layout.CounterSlots;

    /// <summary>Whether the kernels of the PV format <c>numerics.int8AttnV2F16Pv</c> picks are loaded.</summary>
    private bool Int8V2KernelsLoaded()
    {
        EnsureKernels();
        return EngineKnobs.Int8AttnV2F16Pv.Value ? _kernels!.HasInt8AttentionV2F16Pv : _kernels!.HasInt8AttentionV2;
    }

    /// <summary>What the INT8 v2 rows take beyond their declared capability: <see cref="Int8V2Eligible"/>, the row's PV format,
    /// 16-byte aligned rows and the kernels.</summary>
    private bool Int8V2Admits(in AttentionCall c, bool f16Pv) =>
        f16Pv == EngineKnobs.Int8AttnV2F16Pv.Value && Int8V2Eligible(in c)
        && Int8V2RowsAligned(c.Query) && Int8V2RowsAligned(c.Key) && Int8V2RowsAligned(c.Value) && Int8V2RowsAligned(c.Output)
        && c.Output.CoversStorage && Int8V2KernelsLoaded();

    /// <summary>Whether the head-major widening of this grouped call is one S.17 takes, so the grouped token-major row yields to it.
    /// The widening is contiguous; a widened call S.17 still declines runs on the dense rows, where S.1 follows.</summary>
    private bool Int8V2TakesWidened(in AttentionCall c) => Int8V2Eligible(in c) && Int8V2KernelsLoaded();

    /// <inheritdoc/>
    public bool Int8AttentionV2TakesTokenMajor(int heads, int queryTokens, int keys, int headDim, DType dtype)
    {
        if (!EngineKnobs.Int8AttnV2.Value || headDim != Int8AttentionV2Layout.HeadDim || (dtype != DType.F16 && dtype != DType.F32)
            || Math.Min(queryTokens, keys) < EngineKnobs.Int8AttnV2MinTokens.Value || !CudaKernels.GridYzFits(heads, 1)
            || !Int8V2KernelsLoaded()) return false;
        // A capture cannot grow the workspace: a call it does not fit falls to the next row.
        return !StreamIsCapturing() || new Int8AttentionV2Layout(1, heads, queryTokens, keys, CurrentInt8V2Pv()).Bytes <= _int8V2WorkspaceBytes;
    }

    /// <summary>The PV format the knobs pick.</summary>
    private static Int8AttentionV2Pv CurrentInt8V2Pv() =>
        EngineKnobs.Int8AttnV2F16Pv.Value ? Int8AttentionV2Pv.F16
        : EngineKnobs.Int8AttnV2VTile.Value ? Int8AttentionV2Pv.U8TileScales : Int8AttentionV2Pv.U8SequenceScales;

    /// <summary>Offset and strides in whole 8-element steps: every row the quant and epilogue touch starts 16-byte aligned.</summary>
    private static bool Int8V2RowsAligned(in AttentionOperand o) =>
        o.Offset % 8 == 0 && o.TokenStride % 8 == 0 && (o.Heads == 1 || o.HeadStride % 8 == 0) && (o.Batch == 1 || o.BatchStride % 8 == 0);

    /// <summary>Runs the call; false when a capture finds the workspace too small (the next row runs).</summary>
    private bool RunInt8AttentionV2(in AttentionCall c, bool f16Pv)
    {
        using OpScope _op = EnterOp();
        EnsureKernels();
        Int8AttentionV2Layout layout = new(c.Batch, c.Heads, c.QueryTokens, c.ValidKvLength, CurrentInt8V2Pv());
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
        ZeroInt8V2Counters(workspace, (long)c.Batch * c.Heads);
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

    /// <summary>Zeroes the stats launch's completion counters of <paramref name="slots"/> (batch, head) pairs, so a launch that
    /// stopped part way cannot leave one counting for a later call.</summary>
    private void ZeroInt8V2Counters(ulong workspace, long slots) =>
        CudaDriverApi.cuMemsetD32Async(workspace, 0, (nuint)slots, _stream.Handle).ThrowOnError();

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
}
