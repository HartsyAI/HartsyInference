using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Deterministic eager F32 oracle for the versioned VSA routing and fine/coarse attention contract.</summary>
/// <remarks>This implementation intentionally favors transparency over throughput. CUDA implements the same session
/// contract with persistent device buffers; this oracle exists for golden fixtures and backend parity tests.</remarks>
public sealed unsafe class VideoSparseAttentionReferenceSession : IVideoSparseAttentionSession
{
    private readonly VideoSparseAttentionPlan _plan;
    private readonly int[] _sourceBlock;
    private bool _disposed;

    /// <summary>Creates an eager reference session after validating the generation layout.</summary>
    public VideoSparseAttentionReferenceSession(VideoSparseAttentionPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        plan.Validate();
        _plan = plan;
        _sourceBlock = new int[plan.SequenceLength];
        Array.Fill(_sourceBlock, -1);
        for (int block = 0; block < plan.BlockOffsets.Length - 1; block++)
        {
            for (int i = plan.BlockOffsets[block]; i < plan.BlockOffsets[block + 1]; i++)
            {
                _sourceBlock[plan.SourceIndices[i]] = block;
            }
        }
    }

    /// <inheritdoc/>
    public VideoSparseAttentionProfileKind Profile => _plan.Profile;

    /// <inheritdoc/>
    public void Execute(Tensor output, Tensor query, Tensor key, Tensor value, Tensor gate)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateTensors(output, query, key, value, gate);
        int heads = (int)query.Shape[1];
        int sequence = (int)query.Shape[2];
        int headDim = (int)query.Shape[3];
        int blocks = _plan.BlockOffsets.Length - 1;
        float[] qCentroids = new float[blocks * heads * headDim];
        float[] kCentroids = new float[qCentroids.Length];
        float[] vCentroids = new float[qCentroids.Length];
        float[] kMeans = new float[heads * headDim];
        BuildCentroids(query, key, value, qCentroids, kCentroids, vCentroids, heads, sequence, headDim);
        BuildKeyMeans(kCentroids, kMeans, blocks, heads, headDim);

        float* pOut = (float*)output.DataPointer;
        float* pQuery = (float*)query.DataPointer;
        float* pKey = (float*)key.DataPointer;
        float* pValue = (float*)value.DataPointer;
        float* pGate = (float*)gate.DataPointer;
        new Span<float>(pOut, checked((int)output.ElementCount)).Clear();
        float scale = 1f / MathF.Sqrt(headDim);
        bool[] selected = new bool[blocks];
        float[] fine = new float[headDim];
        float[] coarse = new float[headDim];
        float[] scores = new float[blocks];
        int[] order = new int[blocks];

        for (int head = 0; head < heads; head++)
        {
            for (int source = 0; source < sequence; source++)
            {
                int queryBlock = _sourceBlock[source];
                if (queryBlock < 0)
                {
                    continue;
                }
                Array.Clear(selected);
                SelectBlocks(selected, scores, order, qCentroids, kCentroids, kMeans,
                    queryBlock, head, heads, headDim);
                Array.Clear(fine);
                Array.Clear(coarse);
                ExactAttention(fine, pQuery, pKey, pValue, source, head, sequence, headDim, scale, selected);
                CoarseAttention(coarse, qCentroids, queryBlock, head, heads, headDim, scale,
                    kCentroids, vCentroids);
                int outputBase = (head * sequence + source) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    pOut[outputBase + d] = fine[d] + pGate[outputBase + d] * coarse[d];
                }
            }
        }
    }

    private void BuildCentroids(Tensor query, Tensor key, Tensor value, float[] qCentroids, float[] kCentroids,
        float[] vCentroids, int heads, int sequence, int headDim)
    {
        float* pQuery = (float*)query.DataPointer;
        float* pKey = (float*)key.DataPointer;
        float* pValue = (float*)value.DataPointer;
        int blocks = _plan.BlockOffsets.Length - 1;
        for (int block = 0; block < blocks; block++)
        {
            int start = _plan.BlockOffsets[block];
            int stop = _plan.BlockOffsets[block + 1];
            float inverseCount = 1f / (stop - start);
            for (int head = 0; head < heads; head++)
            {
                int centroidBase = (block * heads + head) * headDim;
                for (int i = start; i < stop; i++)
                {
                    int source = _plan.SourceIndices[i];
                    int tensorBase = (head * sequence + source) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        qCentroids[centroidBase + d] += pQuery[tensorBase + d] * inverseCount;
                        kCentroids[centroidBase + d] += pKey[tensorBase + d] * inverseCount;
                        vCentroids[centroidBase + d] += pValue[tensorBase + d] * inverseCount;
                    }
                }
            }
        }
    }

    private static void BuildKeyMeans(float[] kCentroids, float[] kMeans, int blocks, int heads, int headDim)
    {
        float inverseBlocks = 1f / blocks;
        for (int block = 0; block < blocks; block++)
        {
            for (int head = 0; head < heads; head++)
            {
                int source = (block * heads + head) * headDim;
                int destination = head * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    kMeans[destination + d] += kCentroids[source + d] * inverseBlocks;
                }
            }
        }
    }

    private void SelectBlocks(bool[] selected, float[] scores, int[] order, float[] qCentroids, float[] kCentroids,
        float[] kMeans, int queryBlock, int head, int heads, int headDim)
    {
        int blocks = selected.Length;
        if (queryBlock < _plan.PrefixSinkBlocks)
        {
            Array.Fill(selected, true);
            return;
        }
        int qBase = (queryBlock * heads + head) * headDim;
        int routeableBlocks = blocks - _plan.PrefixSinkBlocks;
        for (int block = _plan.PrefixSinkBlocks; block < blocks; block++)
        {
            int kBase = (block * heads + head) * headDim;
            scores[block] = _plan.Profile == VideoSparseAttentionProfileKind.ComfySol64V1
                ? QuantizedCentroidDot(qCentroids, qBase, kCentroids, kBase,
                    kMeans, head * headDim, headDim)
                : Dot(qCentroids, qBase, kCentroids, kBase, headDim);
            order[block - _plan.PrefixSinkBlocks] = block;
        }
        Array.Sort(order, 0, routeableBlocks, Comparer<int>.Create((left, right) =>
        {
            int scoreOrder = scores[right].CompareTo(scores[left]);
            return scoreOrder != 0 ? scoreOrder : left.CompareTo(right);
        }));
        int keep = Math.Clamp((int)MathF.Ceiling(routeableBlocks * _plan.KeepFraction), 1, routeableBlocks);
        if (_plan.Profile == VideoSparseAttentionProfileKind.ComfySol64V1)
        {
            if (keep == routeableBlocks)
            {
                for (int block = _plan.PrefixSinkBlocks; block < blocks; block++)
                {
                    selected[block] = true;
                }
            }
            else
            {
                float threshold = scores[order[keep]];
                for (int block = _plan.PrefixSinkBlocks; block < blocks; block++)
                {
                    selected[block] = scores[block] > threshold;
                }
            }
        }
        else
        {
            for (int i = 0; i < keep; i++)
            {
                selected[order[i]] = true;
            }
        }
        for (int block = 0; block < _plan.PrefixSinkBlocks; block++)
        {
            selected[block] = true;
        }
        if (_plan.Profile == VideoSparseAttentionProfileKind.ComfySol64V1)
        {
            selected[queryBlock] = true;
            if (queryBlock > 0)
            {
                selected[queryBlock - 1] = true;
            }
            if (queryBlock + 1 < blocks)
            {
                selected[queryBlock + 1] = true;
            }
        }
    }

    private void ExactAttention(float[] result, float* query, float* key, float* value, int queryRow, int head,
        int sequence, int headDim, float scale, bool[] selected)
    {
        float runningMax = float.NegativeInfinity;
        float runningSum = 0f;
        int queryBase = (head * sequence + queryRow) * headDim;
        for (int block = 0; block < selected.Length; block++)
        {
            if (!selected[block])
            {
                continue;
            }
            for (int i = _plan.BlockOffsets[block]; i < _plan.BlockOffsets[block + 1]; i++)
            {
                int keyRow = _plan.SourceIndices[i];
                int keyBase = (head * sequence + keyRow) * headDim;
                float score = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    score += query[queryBase + d] * key[keyBase + d];
                }
                score *= scale;
                float nextMax = MathF.Max(runningMax, score);
                float oldWeight = runningSum == 0f ? 0f : MathF.Exp(runningMax - nextMax);
                float newWeight = MathF.Exp(score - nextMax);
                float nextSum = runningSum * oldWeight + newWeight;
                for (int d = 0; d < headDim; d++)
                {
                    result[d] = (result[d] * runningSum * oldWeight + value[keyBase + d] * newWeight) / nextSum;
                }
                runningMax = nextMax;
                runningSum = nextSum;
            }
        }
    }

    private static void CoarseAttention(float[] result, float[] queryCentroids, int queryBlock, int head,
        int heads, int headDim, float scale, float[] keyCentroids, float[] valueCentroids)
    {
        int blocks = keyCentroids.Length / (heads * headDim);
        int queryBase = (queryBlock * heads + head) * headDim;
        float runningMax = float.NegativeInfinity;
        float runningSum = 0f;
        for (int block = 0; block < blocks; block++)
        {
            int centroidBase = (block * heads + head) * headDim;
            float score = 0f;
            for (int d = 0; d < headDim; d++)
            {
                score += queryCentroids[queryBase + d] * keyCentroids[centroidBase + d];
            }
            score *= scale;
            float nextMax = MathF.Max(runningMax, score);
            float oldWeight = runningSum == 0f ? 0f : MathF.Exp(runningMax - nextMax);
            float newWeight = MathF.Exp(score - nextMax);
            float nextSum = runningSum * oldWeight + newWeight;
            for (int d = 0; d < headDim; d++)
            {
                result[d] = (result[d] * runningSum * oldWeight + valueCentroids[centroidBase + d] * newWeight) / nextSum;
            }
            runningMax = nextMax;
            runningSum = nextSum;
        }
    }

    private static float Dot(float[] left, int leftOffset, float[] right, int rightOffset, int length)
    {
        float sum = 0f;
        for (int i = 0; i < length; i++)
        {
            sum += left[leftOffset + i] * right[rightOffset + i];
        }
        return sum;
    }

    private static float QuantizedCentroidDot(float[] left, int leftOffset, float[] right, int rightOffset,
        float[] rightMean, int meanOffset, int length)
    {
        float leftAbsMax = 0f;
        float rightAbsMax = 0f;
        for (int i = 0; i < length; i++)
        {
            leftAbsMax = MathF.Max(leftAbsMax, MathF.Abs(left[leftOffset + i]));
            rightAbsMax = MathF.Max(rightAbsMax, MathF.Abs(right[rightOffset + i] - rightMean[meanOffset + i]));
        }
        float leftScale = leftAbsMax == 0f ? 1f : leftAbsMax / 127f;
        float rightScale = rightAbsMax == 0f ? 1f : rightAbsMax / 127f;
        int integerDot = 0;
        for (int i = 0; i < length; i++)
        {
            int l = Math.Clamp((int)MathF.Round(left[leftOffset + i] / leftScale), -127, 127);
            int r = Math.Clamp((int)MathF.Round(
                (right[rightOffset + i] - rightMean[meanOffset + i]) / rightScale), -127, 127);
            integerDot += l * r;
        }
        return integerDot * leftScale * rightScale;
    }

    private void ValidateTensors(Tensor output, Tensor query, Tensor key, Tensor value, Tensor gate)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        ArgumentNullException.ThrowIfNull(gate);
        if (query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32
            || output.DType != DType.F32 || gate.DType != DType.F32)
        {
            throw new NotSupportedException("The eager VSA oracle requires F32 tensors.");
        }
        if (query.Shape.Rank != 4 || query.Shape[0] != 1 || query.Shape[2] != _plan.SequenceLength)
        {
            throw new ArgumentException("VSA Q must have shape [1,H,S,D] matching the generation plan.", nameof(query));
        }
        if (key.Shape != query.Shape || value.Shape != query.Shape || output.Shape != query.Shape)
        {
            throw new ArgumentException("VSA Q/K/V/output shapes must match.");
        }
        if (gate.Shape != query.Shape)
        {
            throw new ArgumentException("VSA gate must have the same [1,H,S,D] shape as Q.", nameof(gate));
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
