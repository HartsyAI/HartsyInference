using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the guided top-K logits contract (<c>docs/BACKEND_OPS.md</c>): per row, the candidates are the
/// entries of <c>cond</c> at or above its K-th largest value (ties at that value stay candidates); each gets
/// <c>uncond + (cond − uncond) · guidance</c> and every other entry is −∞. The <see cref="IBackend"/> default and the oracle
/// the kernel is checked against.</summary>
public static unsafe class TopKGuidedReference
{
    public static void Apply(Tensor output, Tensor cond, Tensor uncond, int topK, float guidance)
    {
        (long rows, int width) = TopKGuidedContract.Validate(output, cond, uncond, topK, guidance);
        float* pOut = (float*)output.DataPointer;
        float* pCond = (float*)cond.DataPointer;
        float* pUncond = (float*)uncond.DataPointer;
        float[] sorted = new float[width];
        for (long row = 0; row < rows; row++)
        {
            long at = row * width;
            new ReadOnlySpan<float>(pCond + at, width).CopyTo(sorted);
            Array.Sort(sorted);
            float threshold = sorted[width - topK];
            for (int i = 0; i < width; i++)
            {
                float c = pCond[at + i];
                float u = pUncond[at + i];
                pOut[at + i] = c < threshold ? float.NegativeInfinity : u + (c - u) * guidance;
            }
        }
    }
}
