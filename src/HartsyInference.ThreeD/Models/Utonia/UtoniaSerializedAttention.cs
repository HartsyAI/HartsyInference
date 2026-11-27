using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// Patch attention over one serialization curve: points are attended within consecutive patches of the serialized
/// order, with 3D RoPE on queries and keys. CPU reference; the flash-attention path in Pointcept is the same math
/// in bf16.
/// </summary>
internal sealed class UtoniaSerializedAttention
{
    private readonly IBackend _backend;
    private readonly UtoniaLinear _qkv;
    private readonly UtoniaLinear _proj;
    private readonly int _channels;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _patchSize;
    private readonly float _ropeBase;
    private readonly float _scale;
    private readonly int _orderIndex;
    private readonly long _chunkBytes;

    public UtoniaSerializedAttention(IBackend backend, UtoniaLinear qkv, UtoniaLinear proj, int channels, int heads, int patchSize,
        float ropeBase, int orderIndex, long chunkBytes = UtoniaRowChunks.ChunkBytes)
    {
        if (heads <= 0 || channels % heads != 0)
        {
            throw new HartsyInferenceException($"Utonia attention needs channels {channels} divisible by heads {heads}.");
        }

        _backend = backend;
        _qkv = qkv;
        _proj = proj;
        _channels = channels;
        _heads = heads;
        _headDim = channels / heads;
        _patchSize = patchSize;
        _ropeBase = ropeBase;
        _orderIndex = orderIndex;
        _chunkBytes = chunkBytes;
        _scale = (float)(1.0 / Math.Sqrt(_headDim));
    }

    /// <summary>Attends over <paramref name="feat"/> arranged by <paramref name="geometry"/>'s curve; returns a new <c>[N, C]</c> tensor.</summary>
    public Tensor Forward(Tensor feat, UtoniaPoint geometry)
    {
        if (geometry.Serialized is null)
        {
            throw new HartsyInferenceException("Utonia attention needs a serialized point set.");
        }

        int n = geometry.Count;
        UtoniaPatchPlan plan = UtoniaPatchPlan.Create(geometry.SampleCounts(), _patchSize);
        int[] order = geometry.Serialized.Order[_orderIndex];
        int[] inverse = geometry.Serialized.Inverse[_orderIndex];
        int padded = plan.PaddedCount;
        int[] rows = new int[padded];
        float[] slotCoord = new float[padded * 3];
        for (int slot = 0; slot < padded; slot++)
        {
            int point = order[plan.Pad[slot]];
            rows[slot] = point;
            slotCoord[slot * 3] = geometry.Coord[point * 3];
            slotCoord[slot * 3 + 1] = geometry.Coord[point * 3 + 1];
            slotCoord[slot * 3 + 2] = geometry.Coord[point * 3 + 2];
        }

        int[] slotOfPoint = new int[n];
        for (int p = 0; p < n; p++)
        {
            slotOfPoint[p] = plan.Unpad[inverse[p]];
        }

        // Chunks hold whole segments, so no attention window crosses a chunk and each chunk's qkv stays bounded.
        int maxChunkRows = UtoniaRowChunks.RowsPerChunk(_chunkBytes, 3L * _channels);
        Tensor attended = new(new TensorShape(n, _channels), DType.F32);
        ReadOnlySpan<float> featRows = feat.AsReadOnlySpan<float>();
        Span<float> outRows = attended.AsSpan<float>();
        int segment = 0;
        while (segment < plan.SegmentStart.Length)
        {
            int segmentFirst = segment, slotStart = plan.SegmentStart[segment], slotEnd = slotStart;
            do
            {
                slotEnd += plan.SegmentLength[segment];
                segment++;
            }
            while (segment < plan.SegmentLength.Length && slotEnd + plan.SegmentLength[segment] - slotStart <= maxChunkRows);

            AttendChunk(featRows, rows, slotCoord, plan, segmentFirst, segment, slotStart, slotEnd - slotStart, slotOfPoint, outRows);
        }

        Tensor output = UtoniaRowChunks.Map(attended, _channels, UtoniaRowChunks.RowsPerChunk(_chunkBytes, _channels),
            (src, res) => _proj.Forward(src, res));
        attended.Dispose();
        return output;
    }

    /// <summary>Projects the chunk's slots, rotates them, attends each segment and writes each point's row once.</summary>
    private void AttendChunk(ReadOnlySpan<float> featRows, int[] rows, float[] slotCoord, UtoniaPatchPlan plan, int segmentFirst,
        int segmentEnd, int slotStart, int chunkRows, int[] slotOfPoint, Span<float> outRows)
    {
        Tensor input = new(new TensorShape(chunkRows, _channels), DType.F32);
        Span<float> inputRows = input.AsSpan<float>();
        for (int s = 0; s < chunkRows; s++)
        {
            featRows.Slice(rows[slotStart + s] * _channels, _channels).CopyTo(inputRows.Slice(s * _channels, _channels));
        }

        Tensor qkv = new(new TensorShape(chunkRows, 3L * _channels), DType.F32);
        _qkv.Forward(input, qkv);
        input.Dispose();
        Tensor attnSlots = new(new TensorShape(chunkRows, _channels), DType.F32);
        unsafe
        {
            float* qkvPtr = (float*)qkv.DataPointer;
            if (_ropeBase > 0f)
            {
                UtoniaRope.Apply(qkvPtr, slotCoord.AsSpan(slotStart * 3, chunkRows * 3), _channels, _heads, _ropeBase);
            }

            AttendSegments(qkvPtr, (float*)attnSlots.DataPointer, plan, segmentFirst, segmentEnd, slotStart);
        }

        qkv.Dispose();
        ReadOnlySpan<float> slotRows = attnSlots.AsReadOnlySpan<float>();
        for (int s = 0; s < chunkRows; s++)
        {
            int slot = slotStart + s, point = rows[slot];
            if (slotOfPoint[point] == slot)
            {
                slotRows.Slice(s * _channels, _channels).CopyTo(outRows.Slice(point * _channels, _channels));
            }
        }

        attnSlots.Dispose();
    }

    private unsafe void AttendSegments(float* qkv, float* outSlots, UtoniaPatchPlan plan, int segmentFirst, int segmentEnd,
        int slotBase)
    {
        List<int> lengths = new();
        for (int g = segmentFirst; g < segmentEnd; g++)
        {
            if (!lengths.Contains(plan.SegmentLength[g]))
            {
                lengths.Add(plan.SegmentLength[g]);
            }
        }

        foreach (int len in lengths)
        {
            List<int> segments = new();
            for (int g = segmentFirst; g < segmentEnd; g++)
            {
                if (plan.SegmentLength[g] == len)
                {
                    segments.Add(g);
                }
            }

            int count = segments.Count;
            Tensor q = new(new TensorShape(count, _heads, len, _headDim), DType.F32);
            Tensor k = new(new TensorShape(count, _heads, len, _headDim), DType.F32);
            Tensor v = new(new TensorShape(count, _heads, len, _headDim), DType.F32);
            Tensor o = new(new TensorShape(count, _heads, len, _headDim), DType.F32);
            float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
            for (int p = 0; p < count; p++)
            {
                int start = plan.SegmentStart[segments[p]] - slotBase;
                for (int t = 0; t < len; t++)
                {
                    long rowBase = (long)(start + t) * 3 * _channels;
                    for (int h = 0; h < _heads; h++)
                    {
                        long dst = (((long)p * _heads + h) * len + t) * _headDim;
                        for (int d = 0; d < _headDim; d++)
                        {
                            int src = h * _headDim + d;
                            qp[dst + d] = qkv[rowBase + src];
                            kp[dst + d] = qkv[rowBase + _channels + src];
                            vp[dst + d] = qkv[rowBase + 2 * _channels + src];
                        }
                    }
                }
            }

            _backend.ScaledDotProductAttention(o, q, k, v, null, _scale);
            float* op = (float*)o.DataPointer;
            for (int p = 0; p < count; p++)
            {
                int start = plan.SegmentStart[segments[p]] - slotBase;
                for (int t = 0; t < len; t++)
                {
                    long rowBase = (long)(start + t) * _channels;
                    for (int h = 0; h < _heads; h++)
                    {
                        long src = (((long)p * _heads + h) * len + t) * _headDim;
                        for (int d = 0; d < _headDim; d++)
                        {
                            outSlots[rowBase + h * _headDim + d] = op[src + d];
                        }
                    }
                }
            }

            q.Dispose();
            k.Dispose();
            v.Dispose();
            o.Dispose();
        }
    }
}
