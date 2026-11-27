using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// Pads each sample to whole serialized patches and lists the attention segments. Mirrors the padding rule of
/// Pointcept's serialized attention: a sample longer than the patch is padded by copying the previous patch's
/// tail; a shorter one stays unpadded as a single segment.
/// </summary>
public sealed class UtoniaPatchPlan
{
    private UtoniaPatchPlan(int[] pad, int[] unpad, int[] segmentStart, int[] segmentLength)
    {
        Pad = pad;
        Unpad = unpad;
        SegmentStart = segmentStart;
        SegmentLength = segmentLength;
    }

    /// <summary>Padded slot to serialized position (global over the concatenated samples).</summary>
    public int[] Pad { get; }

    /// <summary>Serialized position to padded slot.</summary>
    public int[] Unpad { get; }

    /// <summary>First padded slot of each attention segment.</summary>
    public int[] SegmentStart { get; }

    /// <summary>Length of each attention segment; full patches except a short sample's single segment.</summary>
    public int[] SegmentLength { get; }

    /// <summary>Number of padded slots.</summary>
    public int PaddedCount => Pad.Length;

    /// <summary>Builds the plan from per-sample point counts (contiguous samples, in order).</summary>
    public static UtoniaPatchPlan Create(ReadOnlySpan<int> bincount, int patchSize)
    {
        if (patchSize <= 0)
        {
            throw new HartsyInferenceException("Utonia patch size must be positive.");
        }

        int samples = bincount.Length;
        int[] offset = new int[samples + 1];
        int[] offsetPad = new int[samples + 1];
        int[] countPad = new int[samples];
        for (int i = 0; i < samples; i++)
        {
            int bc = bincount[i];
            countPad[i] = bc > patchSize ? (bc + patchSize - 1) / patchSize * patchSize : bc;
            offset[i + 1] = offset[i] + bc;
            offsetPad[i + 1] = offsetPad[i] + countPad[i];
        }

        int[] pad = new int[offsetPad[samples]];
        for (int s = 0; s < pad.Length; s++)
        {
            pad[s] = s;
        }

        int[] unpad = new int[offset[samples]];
        for (int j = 0; j < unpad.Length; j++)
        {
            unpad[j] = j;
        }

        List<int> segStart = new();
        List<int> segLen = new();
        for (int i = 0; i < samples; i++)
        {
            int bc = bincount[i], bcPad = countPad[i], off = offset[i], offPad = offsetPad[i];
            int shift = offPad - off;
            for (int j = off; j < off + bc; j++)
            {
                unpad[j] += shift;
            }

            if (bc != bcPad)
            {
                int r = bc % patchSize, end = offPad + bcPad;
                int dst = end - patchSize + r, src = end - 2 * patchSize + r;
                for (int t = 0; t < patchSize - r; t++)
                {
                    pad[dst + t] = pad[src + t];
                }
            }

            for (int s = offPad; s < offPad + bcPad; s++)
            {
                pad[s] -= shift;
            }

            if (bc == 0)
            {
                continue;
            }

            if (bc <= patchSize)
            {
                segStart.Add(offPad);
                segLen.Add(bc);
                continue;
            }

            for (int m = 0; m < bcPad / patchSize; m++)
            {
                segStart.Add(offPad + m * patchSize);
                segLen.Add(patchSize);
            }
        }

        return new UtoniaPatchPlan(pad, unpad, segStart.ToArray(), segLen.ToArray());
    }
}
