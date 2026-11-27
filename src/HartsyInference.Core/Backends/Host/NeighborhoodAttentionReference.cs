using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the neighborhood attention contract (<c>docs/BACKEND_OPS.md</c>): for every query, a softmax
/// over the scaled dot products with the keys of its window, applied to the matching values. A plain loop with one score buffer
/// per call: the <see cref="IBackend"/> default and the oracle every kernel is checked against, not a performance path.</summary>
public static unsafe class NeighborhoodAttentionReference
{
    public static void Apply(Tensor output, Tensor q, Tensor k, Tensor v, in NeighborhoodWindow window, float scale)
    {
        NeighborhoodGeometry g = NeighborhoodAttentionContract.Validate(output, q, k, v, window, scale);
        int kt = g.WindowDepth, kh = g.WindowHeight, kw = g.WindowWidth, headDim = g.HeadDim;
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer;
        float* vp = (float*)v.DataPointer, op = (float*)output.DataPointer;
        long headStride = headDim, wStride = (long)g.Heads * headDim;
        long hStride = g.Width * wStride, tStride = (long)g.Height * hStride, bStride = (long)g.Depth * tStride;
        float[] scores = new float[kt * kh * kw];

        for (int b = 0; b < g.Batch; b++)
        for (int it = 0; it < g.Depth; it++)
        {
            int t0 = NeighborhoodAttentionContract.WindowStart(it, g.Depth, kt);
            for (int ih = 0; ih < g.Height; ih++)
            {
                int h0 = NeighborhoodAttentionContract.WindowStart(ih, g.Height, kh);
                for (int iw = 0; iw < g.Width; iw++)
                {
                    int w0 = NeighborhoodAttentionContract.WindowStart(iw, g.Width, kw);
                    long qBase = b * bStride + it * tStride + ih * hStride + iw * wStride;
                    for (int head = 0; head < g.Heads; head++)
                    {
                        long qOff = qBase + head * headStride;
                        float max = float.NegativeInfinity;
                        int n = 0;
                        for (int wt = 0; wt < kt; wt++)
                        for (int wh = 0; wh < kh; wh++)
                        for (int ww = 0; ww < kw; ww++)
                        {
                            long kOff = b * bStride + (t0 + wt) * tStride + (h0 + wh) * hStride + (w0 + ww) * wStride + head * headStride;
                            float dot = 0f;
                            for (int d = 0; d < headDim; d++) dot += qp[qOff + d] * kp[kOff + d];
                            dot *= scale;
                            scores[n++] = dot;
                            if (dot > max) max = dot;
                        }

                        float sum = 0f;
                        for (int i = 0; i < n; i++)
                        {
                            float e = MathF.Exp(scores[i] - max);
                            scores[i] = e;
                            sum += e;
                        }
                        float inv = sum > 0f ? 1f / sum : 0f;

                        for (int d = 0; d < headDim; d++) op[qOff + d] = 0f;
                        n = 0;
                        for (int wt = 0; wt < kt; wt++)
                        for (int wh = 0; wh < kh; wh++)
                        for (int ww = 0; ww < kw; ww++)
                        {
                            float weight = scores[n++] * inv;
                            long vOff = b * bStride + (t0 + wt) * tStride + (h0 + wh) * hStride + (w0 + ww) * wStride + head * headStride;
                            for (int d = 0; d < headDim; d++) op[qOff + d] += weight * vp[vOff + d];
                        }
                    }
                }
            }
        }
    }
}
