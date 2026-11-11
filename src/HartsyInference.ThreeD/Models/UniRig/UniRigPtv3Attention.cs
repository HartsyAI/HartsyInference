using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// Serialized patch attention of UniRig's point transformer: points follow one space-filling curve and attend inside fixed-size
/// patches, with a non-affine LayerNorm over each head's dims on queries and keys and no positional rotation. Patch layout
/// (padding of a sample to whole patches) is shared with <see cref="UtoniaPatchPlan"/>.
/// </summary>
internal sealed class UniRigPtv3Attention(IBackend backend, Tensor qkvWeight, Tensor qkvBias, Tensor projWeight, Tensor projBias,
    int channels, int heads, int patchSize, int orderIndex)
{
    private const float QkNormEps = 1e-5f;

    private readonly int _headDim = channels / heads;

    /// <summary>Attends over <paramref name="input"/> <c>[N, C]</c> arranged by <paramref name="geometry"/>'s curve; returns a new <c>[N, C]</c> tensor.</summary>
    public Tensor Forward(Tensor input, UtoniaPoint geometry)
    {
        int n = geometry.Count;
        using Tensor qkv = UniRigOps.Linear(backend, input, qkvWeight, qkvBias);
        ReadOnlySpan<float> source = qkv.AsReadOnlySpan<float>();
        UtoniaPatchPlan plan = UtoniaPatchPlan.Create(geometry.SampleCounts(), patchSize);
        int[] order = geometry.Serialized!.Order[orderIndex];
        int[] inverse = geometry.Serialized.Inverse[orderIndex];
        int padded = plan.PaddedCount;
        int[] rows = new int[padded];
        for (int slot = 0; slot < padded; slot++)
        {
            rows[slot] = order[plan.Pad[slot]];
        }

        int[] slotOfPoint = new int[n];
        for (int p = 0; p < n; p++)
        {
            slotOfPoint[p] = plan.Unpad[inverse[p]];
        }

        float[] attendedSlots = new float[padded * channels];
        int segmentCount = plan.SegmentStart.Length;
        int[] lengths = plan.SegmentLength.Distinct().ToArray();
        foreach (int length in lengths)
        {
            List<int> group = [];
            for (int g = 0; g < segmentCount; g++)
            {
                if (plan.SegmentLength[g] == length)
                {
                    group.Add(g);
                }
            }

            AttendGroup(source, rows, plan, group, length, attendedSlots);
        }

        float[] attended = new float[n * channels];
        for (int p = 0; p < n; p++)
        {
            Array.Copy(attendedSlots, slotOfPoint[p] * channels, attended, p * channels, channels);
        }

        using Tensor attendedTensor = UniRigOps.FromFloats(attended, new TensorShape(n, channels));
        return UniRigOps.Linear(backend, attendedTensor, projWeight, projBias);
    }

    private void AttendGroup(ReadOnlySpan<float> qkv, int[] rows, UtoniaPatchPlan plan, List<int> group, int length, float[] attendedSlots)
    {
        int count = group.Count;
        TensorShape shape = new(count, heads, length, _headDim);
        using Tensor q = new(shape, DType.F32);
        using Tensor k = new(shape, DType.F32);
        using Tensor v = new(shape, DType.F32);
        using Tensor o = new(shape, DType.F32);
        Span<float> qs = q.AsSpan<float>();
        Span<float> ks = k.AsSpan<float>();
        Span<float> vs = v.AsSpan<float>();
        for (int p = 0; p < count; p++)
        {
            int start = plan.SegmentStart[group[p]];
            for (int t = 0; t < length; t++)
            {
                int row = rows[start + t];
                long rowBase = (long)row * 3 * channels;
                for (int h = 0; h < heads; h++)
                {
                    int dst = (((p * heads) + h) * length + t) * _headDim;
                    NormalizeHead(qkv.Slice((int)rowBase + h * _headDim, _headDim), qs.Slice(dst, _headDim));
                    NormalizeHead(qkv.Slice((int)rowBase + channels + h * _headDim, _headDim), ks.Slice(dst, _headDim));
                    qkv.Slice((int)rowBase + 2 * channels + h * _headDim, _headDim).CopyTo(vs.Slice(dst, _headDim));
                }
            }
        }

        backend.ScaledDotProductAttention(o, q, k, v, null, (float)(1.0 / Math.Sqrt(_headDim)));
        ReadOnlySpan<float> os = o.AsReadOnlySpan<float>();
        for (int p = 0; p < count; p++)
        {
            int start = plan.SegmentStart[group[p]];
            for (int t = 0; t < length; t++)
            {
                for (int h = 0; h < heads; h++)
                {
                    os.Slice((((p * heads) + h) * length + t) * _headDim, _headDim).CopyTo(attendedSlots.AsSpan((start + t) * channels + h * _headDim, _headDim));
                }
            }
        }
    }

    private static void NormalizeHead(ReadOnlySpan<float> source, Span<float> target)
    {
        double mean = 0;
        foreach (float value in source)
        {
            mean += value;
        }

        mean /= source.Length;
        double variance = 0;
        foreach (float value in source)
        {
            variance += (value - mean) * (value - mean);
        }

        variance /= source.Length;
        double scale = 1.0 / Math.Sqrt(variance + QkNormEps);
        for (int i = 0; i < source.Length; i++)
        {
            target[i] = (float)((source[i] - mean) * scale);
        }
    }
}
