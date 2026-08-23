using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs;

/// <summary>Codebook primitives shared by the residual vector quantizers (DAC, SNAC, BiCodec).</summary>
internal static unsafe class VqOps
{
    /// <summary>L2-normalizes each row of a 2D codebook tensor. Cached once at load time so the cosine-similarity inner loop is a pure dot product.</summary>
    public static Tensor L2NormalizeRows(Tensor src, int rows, int dim)
    {
        Tensor result = new(src.Shape, DType.F32);
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)result.DataPointer;
        for (int r = 0; r < rows; r++)
        {
            double sumSq = 0d;
            int rowBase = r * dim;
            for (int d = 0; d < dim; d++) sumSq += (double)sp[rowBase + d] * sp[rowBase + d];
            float invNorm = (float)(1.0 / Math.Sqrt(sumSq + 1e-12));
            for (int d = 0; d < dim; d++) dp[rowBase + d] = sp[rowBase + d] * invNorm;
        }
        return result;
    }

    /// <summary>Picks the nearest codeword per frame by cosine similarity — the query is L2-normalized and
    /// <paramref name="codebookNorm"/> already is, so the largest dot product wins.</summary>
    /// <param name="projected">Channels-first <c>[batch, codebookDim, t]</c> post-<c>in_proj</c> latent.</param>
    /// <param name="codes">Destination <c>[batch, t]</c> plane; pass a pointer already offset to this codebook's slice.</param>
    public static void NearestCodebookIndices(float* projected, float* codebookNorm, int* codes,
        int batch, int t, int codebookDim, int codebookSize)
    {
        Span<float> normalizedQuery = stackalloc float[codebookDim];
        for (int b = 0; b < batch; b++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                double sumSq = 0d;
                for (int d = 0; d < codebookDim; d++)
                {
                    float v = projected[(b * codebookDim + d) * t + ti];
                    normalizedQuery[d] = v;
                    sumSq += (double)v * v;
                }
                float invNorm = (float)(1.0 / Math.Sqrt(sumSq + 1e-12));
                for (int d = 0; d < codebookDim; d++) normalizedQuery[d] *= invNorm;

                int bestIdx = 0;
                float bestDot = float.MinValue;
                for (int k = 0; k < codebookSize; k++)
                {
                    float dot = 0f;
                    int rowBase = k * codebookDim;
                    for (int d = 0; d < codebookDim; d++)
                        dot += normalizedQuery[d] * codebookNorm[rowBase + d];
                    if (dot > bestDot)
                    {
                        bestDot = dot;
                        bestIdx = k;
                    }
                }
                codes[b * t + ti] = bestIdx;
            }
        }
    }

    /// <summary>Gathers the codeword rows named by <paramref name="codes"/> into a fresh channels-first
    /// <c>[batch, codebookDim, t]</c> tensor ready for <c>out_proj</c>.</summary>
    /// <param name="codes">Source <c>[batch, t]</c> plane; pass a pointer already offset to this codebook's slice.</param>
    public static Tensor GatherCodebookVectors(Tensor codebook, int* codes, int batch, int t, int codebookDim)
    {
        Tensor quantized = new(new TensorShape(batch, codebookDim, t), DType.F32);
        float* qp = (float*)quantized.DataPointer;
        float* cb = (float*)codebook.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                int idx = codes[b * t + ti];
                int rowBase = idx * codebookDim;
                for (int d = 0; d < codebookDim; d++)
                    qp[(b * codebookDim + d) * t + ti] = cb[rowBase + d];
            }
        }
        return quantized;
    }
}
