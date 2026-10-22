using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// INT8 attention v2 rows (S.17/S.18, M.4/M.5): admission, the persistent workspace and the three launches.
public sealed partial class CudaBackend
{
    private ulong _int8V2Workspace;
    private long _int8V2WorkspaceBytes;
    private long _int8V2GrowthFailedAt = long.MaxValue;
    private long _int8V2ExecutionCount;

    /// <summary>Number of INT8 attention v2 executions this session.</summary>
    public long Int8AttentionV2ExecutionCount => Interlocked.Read(ref _int8V2ExecutionCount);

    /// <summary>Query rows per INT8 v2 attention block on this card.</summary>
    internal int Int8V2BlockRows => Int8V2BlockRowsOverride is 64 or 128 ? Int8V2BlockRowsOverride
        : _context.Sm is >= CudaArch.Ampere and < 89 ? 64 : 128;

    /// <summary>Bytes the INT8 v2 workspace holds now (its high-water mark since it was last released).</summary>
    internal long Int8V2WorkspaceBytes => _int8V2WorkspaceBytes;

    /// <summary>The shape checks every INT8 v2 decision shares: the knob, F16 or F32, head dim 128, both sequences at the engage
    /// length, and a grid and counter slot per (batch, head).</summary>
    private static bool Int8V2ShapeEligible(int batch, int heads, int queryTokens, int keys, int qkDim, int vDim, DType dtype) =>
        EngineKnobs.Int8AttnV2.Value && (dtype == DType.F16 || dtype == DType.F32)
        && qkDim == Int8AttentionV2Layout.HeadDim && vDim == Int8AttentionV2Layout.HeadDim
        && Math.Min(queryTokens, keys) >= EngineKnobs.Int8AttnV2MinTokens.Value
        && CudaKernels.GridYzFits(heads, batch) && (long)batch * heads <= Int8AttentionV2Layout.CounterSlots;

    /// <summary>The checks the INT8 v2 rows and the grouped row's yield share: <see cref="Int8V2ShapeEligible"/>, a call that
    /// does not refuse INT8 (Wan's F16 layers do), no bias, host positions, one dtype throughout and a finite positive scale.</summary>
    private static bool Int8V2Eligible(in AttentionCall c) =>
        Int8V2ShapeEligible(c.Batch, c.Heads, c.QueryTokens, c.ValidKvLength, c.QkHeadDim, c.ValueHeadDim, c.Query.DType)
        && (c.Numerics.Refused & AttentionFormats.Int8) == 0 && c.Mask.Bias is null && !c.Positions.OnDevice && c.Key.DType == c.Query.DType && c.Value.DType == c.Query.DType
        && c.Output.DType == c.Query.DType && c.Scale > 0f && float.IsFinite(c.Scale);

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
    /// Alignment is not asked here: S.17 checks the widened call again at admission, and one it declines runs on the dense rows,
    /// where S.1 follows, as it did before v2.</summary>
    private bool Int8V2TakesWidened(in AttentionCall c) => Int8V2Eligible(in c) && Int8V2KernelsLoaded();

    /// <inheritdoc/>
    public bool Int8AttentionV2TakesTokenMajor(int batch, int heads, int queryTokens, int keys, int headDim, DType dtype)
    {
        if (!Int8V2ShapeEligible(batch, heads, queryTokens, keys, headDim, headDim, dtype) || !Int8V2KernelsLoaded()) return false;
        long bytes = new Int8AttentionV2Layout(batch, heads, queryTokens, keys, CurrentInt8V2Pv()).Bytes;
        // A capture cannot grow the workspace, and a size that failed to allocate is not retried: such a call takes the next row.
        return bytes <= _int8V2WorkspaceBytes || (!StreamIsCapturing() && bytes < _int8V2GrowthFailedAt);
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

    /// <summary>The workspace grown to at least <paramref name="bytes"/>, or 0 (the call takes the next row) while the stream
    /// captures and it is smaller, or when that size did not fit in VRAM since activations were last freed. A capture never
    /// allocates it, and a captured graph must never hold a pointer a later growth frees. Ops are serialized (EnterOp).</summary>
    private ulong EnsureInt8V2Workspace(long bytes)
    {
        if (bytes <= _int8V2WorkspaceBytes) return _int8V2Workspace;
        if (StreamIsCapturing() || bytes >= _int8V2GrowthFailedAt) return 0;
        StepGraphInvalidateForActivationFree();
        // Free first: old and new together would need both at peak, which is what fails near the VRAM limit.
        FreeInt8V2Workspace();
        try
        {
            _int8V2Workspace = GpuTransferHelper.AllocateDevice((nuint)bytes);
        }
        catch (OutOfVramException ex)
        {
            _int8V2GrowthFailedAt = bytes;
            Logs.Warning($"[INT8 attention v2] its {bytes >> 20} MB workspace does not fit ({ex.Message}); such calls take the next row.");
            return 0;
        }
        _int8V2WorkspaceBytes = bytes;
        return _int8V2Workspace;
    }

    /// <summary>Returns the workspace to the stream-ordered pool when activations are freed, and lets a size that did not fit try
    /// again; the next call grows it again.</summary>
    private void ReleaseInt8V2Workspace()
    {
        _int8V2GrowthFailedAt = long.MaxValue;
        FreeInt8V2Workspace();
    }

    private void FreeInt8V2Workspace()
    {
        ulong buffer = _int8V2Workspace;
        _int8V2Workspace = 0;
        _int8V2WorkspaceBytes = 0;
        if (buffer != 0) GpuTransferHelper.FreeDevice(buffer);
    }
}
