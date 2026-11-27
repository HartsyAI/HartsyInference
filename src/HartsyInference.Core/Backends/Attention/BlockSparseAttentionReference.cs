using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>The definition of <see cref="IBackend.BlockSparseAttention"/>, written from <c>flash_attn_bsa_3d</c> of
/// meituan-longcat/LongCat-Video at <c>58eb336edc5c8c63389838727e40b8c18434e54e</c> in top-k mode and checked against its kernel's
/// outputs.</summary>
/// <remarks>
/// <para>Per (batch, head): a block's summary is the mean of its queries or keys; a query block scores every key block by the dot
/// product of the two summaries, unscaled; it keeps the <see cref="BlockSparseAttentionOptions.SelectedBlocks"/> highest and each of
/// its queries attends exactly those blocks' keys, one softmax over them. With no selected block the output rows are zero.</para>
/// <para>The reference ranks with <c>torch.topk</c>, which leaves equal scores in no stated order. Here the lower block index wins a
/// tie and a NaN score ranks above every number, so a NaN in the keys reaches the output while any block is selected.</para>
/// <para>Operands are head-major <c>[B, H, S, D]</c>, F32 or BF16. BF16 operands are widened to F32, exactly, before anything is
/// computed, so the means, the selection and the attention are F32; a BF16 output is the F32 result rounded to nearest even. It runs in
/// plain loops and holds no approximation: a GPU port is compared with it, not the other way round. The pinned kernel run on BF16
/// operands rounds its block means and scores to BF16 and can read other blocks; that behaviour is not reproduced.</para>
/// </remarks>
public static class BlockSparseAttentionReference
{
    /// <summary>Attends <paramref name="query"/> over the key blocks it selects from <paramref name="key"/> and
    /// <paramref name="value"/> into <paramref name="output"/>. With <paramref name="selected"/> (I32
    /// <c>[B, H, QueryBlocks, SelectedBlocks]</c>) also reports each query block's key blocks, ascending.</summary>
    /// <exception cref="ArgumentException">An operand or option does not fit the call.</exception>
    public static void Apply(Tensor output, Tensor query, Tensor key, Tensor value, BlockSparseAttentionOptions options,
        Tensor? selected = null)
    {
        BlockSparseAttentionContract.Validate(output, query, key, value, options, selected);
        if (query.DType == DType.F32 && output.DType == DType.F32)
        {
            Compute(output, query, key, value, options, selected);
            return;
        }
        using Tensor? wideQuery = query.DType == DType.F32 ? null : query.CastTo(DType.F32);
        using Tensor? wideKey = key.DType == DType.F32 ? null : key.CastTo(DType.F32);
        using Tensor? wideValue = value.DType == DType.F32 ? null : value.CastTo(DType.F32);
        using Tensor? wideOutput = output.DType == DType.F32 ? null : new Tensor(output.Shape, DType.F32);
        Compute(wideOutput ?? output, wideQuery ?? query, wideKey ?? key, wideValue ?? value, options, selected);
        if (wideOutput is null) return;
        ReadOnlySpan<float> wide = wideOutput.AsReadOnlySpan<float>();
        Span<ushort> narrow = output.AsSpan<ushort>();
        for (int i = 0; i < wide.Length; i++) narrow[i] = TensorCasts.F32ToBf16BitsRounded(wide[i]);
    }

    /// <summary>The definition on F32 operands the contract has accepted.</summary>
    private static void Compute(Tensor output, Tensor query, Tensor key, Tensor value, BlockSparseAttentionOptions options,
        Tensor? selected)
    {
        int heads = (int)query.Shape[1], dim = (int)query.Shape[3];
        int sequences = (int)query.Shape[0] * heads, queryTokens = (int)query.Shape[2], keyTokens = (int)key.Shape[2];
        int queryBlocks = options.QueryBlocks, keyBlocks = options.KeyBlocks, chosen = options.SelectedBlocks;
        float scale = options.Scale ?? 1f / MathF.Sqrt(dim);
        int queryLength = (int)options.QueryBlock.Tokens, keyLength = (int)options.KeyBlock.Tokens;
        long work = (long)queryTokens * Math.Max(1, chosen) * keyLength * dim * 2 + (long)queryBlocks * keyBlocks * dim;

        // Allocate the host buffers before the fan-out: a tensor's first access is not safe from several threads.
        _ = output.AsSpan<float>();
        _ = query.AsReadOnlySpan<float>();
        _ = key.AsReadOnlySpan<float>();
        _ = value.AsReadOnlySpan<float>();
        if (selected is not null && chosen > 0) _ = selected.AsSpan<int>();

        float[] keyMeans = new float[checked(keyBlocks * dim)];
        for (int sequence = 0; sequence < sequences; sequence++)
        {
            BlockMeans(key.AsReadOnlySpan<float>().Slice(sequence * keyTokens * dim, keyTokens * dim), options, keyMeans, dim);
            int current = sequence;
            CpuParallel.For(queryBlocks, work, block =>
            {
                ReadOnlySpan<float> q = query.AsReadOnlySpan<float>().Slice(current * queryTokens * dim, queryTokens * dim);
                ReadOnlySpan<float> k = key.AsReadOnlySpan<float>().Slice(current * keyTokens * dim, keyTokens * dim);
                ReadOnlySpan<float> v = value.AsReadOnlySpan<float>().Slice(current * keyTokens * dim, keyTokens * dim);
                Span<float> o = output.AsSpan<float>().Slice(current * queryTokens * dim, queryTokens * dim);

                int[] blocks = Select(q, options, block, keyMeans, dim);
                if (selected is not null && chosen > 0)
                    blocks.AsSpan().CopyTo(selected.AsSpan<int>().Slice((current * queryBlocks + block) * chosen, chosen));

                int[] keyRows = new int[chosen * keyLength];
                for (int i = 0; i < keyRows.Length; i++) keyRows[i] = options.KeyRow(blocks[i / keyLength], i % keyLength);
                float[] logits = new float[keyRows.Length];
                for (int offset = 0; offset < queryLength; offset++)
                {
                    int row = options.QueryRow(block, offset);
                    Attend(q.Slice(row * dim, dim), k, v, keyRows, scale, logits, o.Slice(row * dim, dim));
                }
            });
        }
    }

    /// <summary>The mean of each key block's rows, summed in the block's raster order.</summary>
    private static void BlockMeans(ReadOnlySpan<float> keys, BlockSparseAttentionOptions options, Span<float> means, int dim)
    {
        int length = (int)options.KeyBlock.Tokens;
        means.Clear();
        for (int block = 0; block < options.KeyBlocks; block++)
        {
            Span<float> mean = means.Slice(block * dim, dim);
            for (int offset = 0; offset < length; offset++)
            {
                ReadOnlySpan<float> row = keys.Slice(options.KeyRow(block, offset) * dim, dim);
                for (int d = 0; d < dim; d++) mean[d] += row[d];
            }
            for (int d = 0; d < dim; d++) mean[d] /= length;
        }
    }

    /// <summary>The key blocks query block <paramref name="block"/> reads, ascending.</summary>
    private static int[] Select(ReadOnlySpan<float> queries, BlockSparseAttentionOptions options, int block, float[] keyMeans, int dim)
    {
        int length = (int)options.QueryBlock.Tokens;
        float[] mean = new float[dim];
        for (int offset = 0; offset < length; offset++)
        {
            ReadOnlySpan<float> row = queries.Slice(options.QueryRow(block, offset) * dim, dim);
            for (int d = 0; d < dim; d++) mean[d] += row[d];
        }
        for (int d = 0; d < dim; d++) mean[d] /= length;

        float[] scores = new float[options.KeyBlocks];
        int[] order = new int[options.KeyBlocks];
        for (int n = 0; n < scores.Length; n++)
        {
            scores[n] = Dot(mean, keyMeans.AsSpan(n * dim, dim));
            order[n] = n;
        }
        Array.Sort(order, (a, b) =>
        {
            int byScore = Rank(scores[a], scores[b]);
            return byScore != 0 ? byScore : a.CompareTo(b);
        });
        int[] chosen = order[..options.SelectedBlocks];
        Array.Sort(chosen);
        return chosen;
    }

    /// <summary>Negative when <paramref name="a"/> ranks ahead: NaN first, then the higher score.</summary>
    private static int Rank(float a, float b)
    {
        if (float.IsNaN(a) || float.IsNaN(b)) return float.IsNaN(b).CompareTo(float.IsNaN(a));
        return b.CompareTo(a);
    }

    private static void Attend(ReadOnlySpan<float> query, ReadOnlySpan<float> keys, ReadOnlySpan<float> values, int[] keyRows,
        float scale, float[] logits, Span<float> output)
    {
        int dim = query.Length;
        output.Clear();
        float max = float.NegativeInfinity;
        for (int i = 0; i < keyRows.Length; i++)
        {
            logits[i] = scale * Dot(query, keys.Slice(keyRows[i] * dim, dim));
            max = MathF.Max(max, logits[i]);
        }
        // No key, or every key excluded: a zero row. A NaN maximum falls through and propagates.
        if (float.IsNegativeInfinity(max)) return;
        float sum = 0f;
        for (int i = 0; i < keyRows.Length; i++)
        {
            float weight = MathF.Exp(logits[i] - max);
            sum += weight;
            ReadOnlySpan<float> row = values.Slice(keyRows[i] * dim, dim);
            for (int d = 0; d < dim; d++) output[d] += weight * row[d];
        }
        for (int d = 0; d < dim; d++) output[d] /= sum;
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        for (int d = 0; d < a.Length; d++) sum += a[d] * b[d];
        return sum;
    }
}
