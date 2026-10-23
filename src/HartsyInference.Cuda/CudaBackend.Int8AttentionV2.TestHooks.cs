using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// INT8 attention v2 test and benchmark hooks: one call with its workspace read back, per-stage device times, the exp2 probe.
public sealed partial class CudaBackend
{
    /// <summary>Test and benchmark hook: query rows per INT8 v2 attention block, 64 or 128; 0 follows the card.</summary>
    internal int Int8V2BlockRowsOverride { get; set; }

    /// <summary>Test hook: the smallest workspace size that ran out of VRAM since activations were last freed.</summary>
    internal long Int8V2GrowthFailedAtForTest { get => _int8V2GrowthFailedAt; set => _int8V2GrowthFailedAt = value; }

    /// <summary>Test hook: each loaded INT8 v2 entry's registers and local bytes as the driver compiled it for this card.</summary>
    internal (string Name, int Registers, int LocalBytes)[] Int8V2EntryResourcesForTest()
    {
        using OpScope _op = EnterOp();
        EnsureKernels();
        return [.. _kernels!.Int8V2EntryResources()];
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
                KScale = Read<float>(workspace + (ulong)layout.KScale, bh * kvPad),
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
                ZeroInt8V2Counters(workspace, (long)c.Batch * c.Heads);
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
