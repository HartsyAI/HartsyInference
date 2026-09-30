using System.Buffers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>CPU reference for attention where one latent row is both key and value, with a per-head sink.</summary>
public static class SparseLatentAttentionReference
{
    /// <summary>Checks the operands of <see cref="Apply"/>; shared by every backend.</summary>
    public static void Validate(Tensor output, Tensor query, in LatentSource window, in LatentSource main,
        Tensor indices, int windowSlots, Tensor sink)
    {
        window.Validate(nameof(window));
        main.Validate(nameof(main));
        if (query.DType != DType.F32 || output.DType != DType.F32 || sink.DType != DType.F32)
            throw new NotSupportedException("SparseLatentAttention supports F32 query, output and sink only.");
        if (indices.DType != DType.I32) throw new ArgumentException("indices must be I32.", nameof(indices));
        if (query.Shape.Rank != 3) throw new ArgumentException("query must be [tokens, heads, dim].", nameof(query));
        if (!output.Shape.Equals(query.Shape)) throw new ArgumentException("output must match the query shape.", nameof(output));
        int tokens = (int)query.Shape[0], heads = (int)query.Shape[1], dim = (int)query.Shape[2];
        if (tokens < 1 || heads < 1 || dim < 1) throw new ArgumentException("query must not be empty.", nameof(query));
        if (indices.Shape.Rank != 2 || indices.Shape[0] != tokens || indices.Shape[1] < 1)
            throw new ArgumentException($"indices must be [{tokens}, k] with k >= 1.", nameof(indices));
        if (sink.ElementCount != heads) throw new ArgumentException($"sink must hold {heads} entries.", nameof(sink));
        if (windowSlots < 0 || window.Rows != windowSlots)
            throw new ArgumentException($"window must hold exactly windowSlots ({windowSlots}) rows.", nameof(window));
        if ((window.Rows > 0 && window.Dim != dim) || (main.Rows > 0 && main.Dim != dim))
            throw new ArgumentException($"Latent sources must have Dim {dim}.");
    }

    /// <summary>Per token and head: softmax over the valid indices' scaled dot products and the head's sink, then the
    /// probability-weighted sum of the latent rows; a row with no valid index yields zeros.</summary>
    /// <remarks>Index <c>i</c> addresses <paramref name="window"/> row <c>i</c> below <paramref name="windowSlots"/> and
    /// <paramref name="main"/> row <c>i - windowSlots</c> above. -1 and any index outside the two sources are skipped.
    /// The sink joins the denominator only. F32 throughout; the maximum includes the sink so nothing overflows.</remarks>
    public static unsafe void Apply(Tensor output, Tensor query, in LatentSource window, in LatentSource main,
        Tensor indices, int windowSlots, Tensor sink, float scale)
    {
        Validate(output, query, window, main, indices, windowSlots, sink);
        int tokens = (int)query.Shape[0], heads = (int)query.Shape[1], dim = (int)query.Shape[2];
        int k = (int)indices.Shape[1];
        long total = (long)windowSlots + main.Rows;
        float* q = (float*)query.DataPointer, o = (float*)output.DataPointer, sk = (float*)sink.DataPointer;
        int* idx = (int*)indices.DataPointer;
        float[] rows = ArrayPool<float>.Shared.Rent(k * dim);
        float[] probs = ArrayPool<float>.Shared.Rent(k);
        try
        {
            for (int t = 0; t < tokens; t++)
            {
                for (int j = 0; j < k; j++)
                {
                    int id = idx[(long)t * k + j];
                    if (id < 0 || id >= total) continue;
                    Span<float> row = rows.AsSpan(j * dim, dim);
                    if (id < windowSlots) LatentRowReader.ReadRow(window, id, row);
                    else LatentRowReader.ReadRow(main, id - windowSlots, row);
                }
                for (int h = 0; h < heads; h++)
                    AttendHead(o + ((long)t * heads + h) * dim, q + ((long)t * heads + h) * dim, idx + (long)t * k, rows, probs,
                        k, dim, total, sk[h], scale);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rows);
            ArrayPool<float>.Shared.Return(probs);
        }
    }

    private static unsafe void AttendHead(float* o, float* q, int* idx, float[] rows, float[] probs, int k, int dim,
        long total, float sink, float scale)
    {
        float max = sink;
        for (int j = 0; j < k; j++)
        {
            if (idx[j] < 0 || idx[j] >= total) continue;
            float dot = 0f;
            for (int d = 0; d < dim; d++) dot += q[d] * rows[j * dim + d];
            probs[j] = dot * scale;
            max = MathF.Max(max, probs[j]);
        }
        float denom = MathF.Exp(sink - max);
        for (int j = 0; j < k; j++)
        {
            if (idx[j] < 0 || idx[j] >= total) continue;
            probs[j] = MathF.Exp(probs[j] - max);
            denom += probs[j];
        }
        for (int d = 0; d < dim; d++)
        {
            float acc = 0f;
            for (int j = 0; j < k; j++)
                if (idx[j] >= 0 && idx[j] < total) acc += probs[j] * rows[j * dim + d];
            o[d] = acc / denom;
        }
    }
}
