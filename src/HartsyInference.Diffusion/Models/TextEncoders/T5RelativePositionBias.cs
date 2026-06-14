using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Computes T5's learned relative position bias for self-attention. Uses bucketed log-linear distance mapping with a learned [numBuckets, numHeads] lookup table. Computed once (layer 0) and broadcast to all encoder layers.</summary>
public sealed unsafe class T5RelativePositionBias
{
    private readonly int _numBuckets;
    private readonly int _maxDistance;
    private readonly int _numHeads;

    /// <summary>Learned bias table [numBuckets, numHeads] loaded from weights.</summary>
    private Tensor? _biasTable;

    /// <summary>Cached bias tensor [1, numHeads, seqLen, seqLen] for the last computed sequence length.</summary>
    private Tensor? _cachedBias;
    private int _cachedSeqLen;

    public T5RelativePositionBias(int numBuckets, int maxDistance, int numHeads)
    {
        _numBuckets = numBuckets;
        _maxDistance = maxDistance;
        _numHeads = numHeads;
    }

    /// <summary>Loads the relative_attention_bias.weight tensor [numBuckets, numHeads]. Auto-casts to F32 if needed (ComputeBias uses float* directly).</summary>
    public void LoadWeights(Tensor biasTable)
    {
        _biasTable = biasTable.DType != DType.F32 ? biasTable.CastTo(DType.F32) : biasTable;
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_biasTable is not null) yield return _biasTable;
    }

    /// <summary>Computes position bias [1, numHeads, seqLen, seqLen]. Cached for repeated calls with same seqLen.</summary>
    public Tensor ComputeBias(int seqLen)
    {
        if (_cachedBias is not null && _cachedSeqLen == seqLen)
            return _cachedBias;

        if (_biasTable is null)
            throw new InvalidOperationException("Bias table weights not loaded.");

        // Compute bucket indices for all (query, key) pairs
        int[] bucketIndices = new int[seqLen * seqLen];
        for (int queryPos = 0; queryPos < seqLen; queryPos++)
        {
            for (int keyPos = 0; keyPos < seqLen; keyPos++)
            {
                int relativePosition = keyPos - queryPos;
                int bucket = ComputeBucket(relativePosition, bidirectional: true, _numBuckets, _maxDistance);
                bucketIndices[queryPos * seqLen + keyPos] = bucket;
            }
        }

        // Look up bias values: output[h, q, k] = biasTable[bucket[q,k], h]
        TensorShape biasShape = new TensorShape(1, _numHeads, seqLen, seqLen);
        Tensor bias = new Tensor(biasShape, DType.F32);

        float* tablePtr = (float*)_biasTable.DataPointer;
        float* outPtr = (float*)bias.DataPointer;

        for (int q = 0; q < seqLen; q++)
        {
            for (int k = 0; k < seqLen; k++)
            {
                int bucket = bucketIndices[q * seqLen + k];
                for (int h = 0; h < _numHeads; h++)
                {
                    // biasTable layout: [numBuckets, numHeads] row-major
                    float val = tablePtr[bucket * _numHeads + h];
                    // output layout: [1, numHeads, seqLen, seqLen]
                    outPtr[h * seqLen * seqLen + q * seqLen + k] = val;
                }
            }
        }

        _cachedBias?.Dispose();
        _cachedBias = bias;
        _cachedSeqLen = seqLen;

        return bias;
    }

    /// <summary>Computes the relative position bucket index using T5's log-linear bucketing scheme. Matches HuggingFace transformers T5Attention._relative_position_bucket exactly.</summary>
    public static int ComputeBucket(int relativePosition, bool bidirectional, int numBuckets, int maxDistance)
    {
        int n = -relativePosition;
        int offset = 0;

        if (bidirectional)
        {
            numBuckets /= 2;
            if (n < 0)
            {
                offset = numBuckets;
                n = -n;
            }
        }
        else
        {
            n = Math.Max(n, 0);
        }

        int maxExact = numBuckets / 2;

        if (n < maxExact)
        {
            return offset + n;
        }

        // Logarithmic bucketing for large distances
        double logVal = Math.Log((double)n / maxExact) / Math.Log((double)maxDistance / maxExact);
        int bucket = maxExact + (int)(logVal * (numBuckets - maxExact));
        bucket = Math.Min(bucket, numBuckets - 1);

        return offset + bucket;
    }
}
