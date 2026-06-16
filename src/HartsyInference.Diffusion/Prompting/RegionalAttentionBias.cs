using System.Collections.Generic;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Builds the additive attention bias that implements regional conditioning over a joint <c>[conditioning | image]</c> token sequence. Image query tokens are biased toward each region's conditioning columns by <c>log(weight_r · mask_r[token])</c>, so a token inside a region attends to that region's prompt while a token outside it is suppressed (floored at <see cref="MinWeight"/> rather than -inf to keep softmax finite). The output is <c>[1, 1, L, L]</c> (shared across heads) for backend portability; all entries outside the image-query / region-column rectangle are 0, leaving base conditioning and image self-attention untouched.</summary>
public static class RegionalAttentionBias
{
    /// <summary>Smallest effective region weight; floors <c>log()</c> so a fully-excluded cell gets a strong but finite negative bias.</summary>
    public const float MinWeight = 1e-4f;

    /// <summary>Builds the <c>[1, 1, seqLen, seqLen]</c> additive bias. <paramref name="regionColumnRanges"/>, <paramref name="regionGridMasks"/> (each <c>[numImg]</c> row-major in <c>[0,1]</c>), and <paramref name="regionWeights"/> are parallel per-region arrays. Image tokens occupy sequence indices <c>[imageStart, imageStart+numImg)</c>.</summary>
    public static Tensor Build(
        int seqLen,
        int imageStart,
        int numImg,
        IReadOnlyList<(int Start, int End)> regionColumnRanges,
        IReadOnlyList<float[]> regionGridMasks,
        ReadOnlySpan<float> regionWeights)
    {
        int regionCount = regionColumnRanges.Count;
        if (regionGridMasks.Count != regionCount || regionWeights.Length != regionCount)
        {
            throw new ArgumentException("regionColumnRanges, regionGridMasks, and regionWeights must have equal length.");
        }
        Tensor bias = new Tensor(new TensorShape(1, 1, seqLen, seqLen), DType.F32);
        Span<float> b = bias.AsSpan<float>();
        b.Clear();
        for (int r = 0; r < regionCount; r++)
        {
            (int start, int end) = regionColumnRanges[r];
            float[] mask = regionGridMasks[r];
            float weight = regionWeights[r];
            if (mask.Length != numImg)
            {
                throw new ArgumentException($"regionGridMasks[{r}] length {mask.Length} must equal numImg {numImg}.");
            }
            for (int k = 0; k < numImg; k++)
            {
                float effective = weight * mask[k];
                float biasVal = MathF.Log(MathF.Max(MinWeight, effective));
                if (biasVal == 0f)
                {
                    continue;
                }
                long rowOff = (long)(imageStart + k) * seqLen;
                for (int col = start; col < end; col++)
                {
                    b[(int)(rowOff + col)] = biasVal;
                }
            }
        }
        return bias;
    }
}
