using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>CPU reference for the lightning-indexer scores: ReLU of query-key dots, weighted per head and summed over heads.</summary>
public static class IndexerScoresReference
{
    /// <summary>Checks the operands of <see cref="Apply"/>; shared by every backend.</summary>
    public static void Validate(Tensor scores, Tensor query, in LatentSource keys, Tensor headWeights, Tensor compressLens,
        Tensor? candidates)
    {
        keys.Validate(nameof(keys));
        if (query.DType != DType.F32 || scores.DType != DType.F32 || headWeights.DType != DType.F32)
            throw new NotSupportedException("IndexerScores supports F32 query, weights and scores only.");
        if (compressLens.DType != DType.I32) throw new ArgumentException("compressLens must be I32.", nameof(compressLens));
        if (query.Shape.Rank != 3) throw new ArgumentException("query must be [tokens, heads, dim].", nameof(query));
        int tokens = (int)query.Shape[0], heads = (int)query.Shape[1], dim = (int)query.Shape[2];
        if (tokens < 1 || heads < 1) throw new ArgumentException("query must not be empty.", nameof(query));
        if (keys.Rows > 0 && keys.Dim != dim) throw new ArgumentException($"keys must have Dim {dim}.", nameof(keys));
        if (headWeights.ElementCount != (long)tokens * heads)
            throw new ArgumentException($"headWeights must hold {tokens}x{heads} entries.", nameof(headWeights));
        if (compressLens.ElementCount != tokens) throw new ArgumentException($"compressLens must hold {tokens} entries.", nameof(compressLens));
        if (scores.ElementCount != (long)tokens * keys.Rows)
            throw new ArgumentException($"scores must hold {tokens}x{keys.Rows} entries.", nameof(scores));
        if (candidates is not null && (candidates.DType != DType.U8 || candidates.ElementCount != scores.ElementCount))
            throw new ArgumentException("candidates must be U8 with one flag per score.", nameof(candidates));
    }

    /// <summary>Writes <c>scores[t,n] = scale * sum_h relu(q[t,h] . key[n]) * headWeights[t,h]</c>, heads summed in order.</summary>
    /// <remarks>Entries with <c>n >= compressLens[t]</c>, or a zero candidate flag, are -infinity; this also hides the
    /// still-open compression group from mid-group queries.</remarks>
    public static unsafe void Apply(Tensor scores, Tensor query, in LatentSource keys, Tensor headWeights,
        Tensor compressLens, Tensor? candidates, float scale)
    {
        Validate(scores, query, keys, headWeights, compressLens, candidates);
        int tokens = (int)query.Shape[0], heads = (int)query.Shape[1], dim = (int)query.Shape[2], n = keys.Rows;
        float* q = (float*)query.DataPointer, w = (float*)headWeights.DataPointer, s = (float*)scores.DataPointer;
        int* lens = (int*)compressLens.DataPointer;
        byte* cand = candidates is null ? null : (byte*)candidates.DataPointer;
        float[] row = new float[dim];
        for (int j = 0; j < n; j++)
        {
            LatentRowReader.ReadRow(keys, j, row);
            for (int t = 0; t < tokens; t++)
            {
                long at = (long)t * n + j;
                if (j >= lens[t] || (cand != null && cand[at] == 0))
                {
                    s[at] = float.NegativeInfinity;
                    continue;
                }
                float sum = 0f;
                for (int h = 0; h < heads; h++)
                {
                    float dot = 0f;
                    float* qh = q + ((long)t * heads + h) * dim;
                    for (int d = 0; d < dim; d++) dot += qh[d] * row[d];
                    sum += MathF.Max(dot, 0f) * w[(long)t * heads + h];
                }
                s[at] = sum * scale;
            }
        }
    }
}
