namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>Helpers the compressed-pool indexers share (glm5_next and Hy4 Flash): the gated softmax pool of each group of slots, the descending top-k of
/// scores, and the packing of one row per query into a <see cref="DsaKeySelection"/>. The score and selection rules stay with each model, since they differ.</summary>
public static class DsaCompressedPool
{
    /// <summary>Pools each of <paramref name="groups"/> groups of <paramref name="slots"/> slots per channel into <paramref name="pooled"/> (<c>[groups, channels]</c>, F32):
    /// <c>sum_j softmax_j(gate_j + ape_j) · value_j</c>. Slot <c>j</c> of group <c>g</c> reads <c>value</c> at <c>(g·slots + j)·stride + channel</c> from
    /// <paramref name="values"/> and the gate the same way from <paramref name="gates"/>; <paramref name="ape"/> is <c>[slots, channels]</c>.</summary>
    public static void GatedSoftmaxPool(ReadOnlySpan<float> values, ReadOnlySpan<float> gates, int stride, ReadOnlySpan<float> ape, int groups, int slots,
        int channels, Span<float> pooled)
    {
        for (int g = 0; g < groups; g++)
            for (int d = 0; d < channels; d++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < slots; j++) max = MathF.Max(max, gates[(g * slots + j) * stride + d] + ape[j * channels + d]);
                float denominator = 0f, acc = 0f;
                for (int j = 0; j < slots; j++)
                {
                    float e = MathF.Exp(gates[(g * slots + j) * stride + d] + ape[j * channels + d] - max);
                    denominator += e;
                    acc += values[(g * slots + j) * stride + d] * e;
                }
                pooled[g * channels + d] = acc / denominator;
            }
    }

    /// <summary>The indices of the <paramref name="count"/> highest scores, highest first. Equal scores resolve to the lower index; the reference top-k
    /// leaves them undefined, so a parity input keeps the last kept and first dropped score apart.</summary>
    public static int[] TopK(ReadOnlySpan<float> scores, int count)
    {
        float[] copy = scores.ToArray();
        int[] order = [.. Enumerable.Range(0, copy.Length)];
        Array.Sort(order, (a, b) => copy[a] != copy[b] ? copy[b].CompareTo(copy[a]) : a.CompareTo(b));
        return order.Take(count).ToArray();
    }

    /// <summary>Packs one ascending row of cache slots per query into a selection, padded with -1 to the widest row; each row's count is its length.</summary>
    public static DsaKeySelection Pack(IReadOnlyList<List<int>> rows)
    {
        int width = 0;
        foreach (List<int> row in rows) width = Math.Max(width, row.Count);
        int[] packed = new int[rows.Count * width], counts = new int[rows.Count];
        Array.Fill(packed, -1);
        for (int t = 0; t < rows.Count; t++)
        {
            rows[t].CopyTo(packed, t * width);
            counts[t] = rows[t].Count;
        }
        return new DsaKeySelection(packed, width, counts);
    }
}
