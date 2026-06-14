using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Axial Rotary Position Embedding for Flux. Computes RoPE rotation matrices across 3 axes [16, 56, 56] with theta=10000, then applies 2x2 rotation to each Q/K pair. Precomputes cos/sin tables per resolution for caching.</summary>
public sealed unsafe class FluxRope
{
    private readonly int[] _axesDim;
    private readonly int _theta;
    private readonly int _headDim;

    // Cached cos/sin tables: [totalSeqLen, headDim/2] interleaved
    private float[]? _cosCache;
    private float[]? _sinCache;
    private int _cachedSeqLen;

    /// <summary>Creates a FluxRope with the given per-axis dimensions.</summary>
    /// <param name="axesDim">Dimensions per axis. Default: [16, 56, 56]. Must sum to headDim.</param>
    /// <param name="theta">Base frequency. Default: 10000.</param>
    public FluxRope(int[]? axesDim = null, int theta = 10000)
    {
        _axesDim = axesDim ?? [16, 56, 56];
        _theta = theta;
        _headDim = 0;
        for (int i = 0; i < _axesDim.Length; i++)
            _headDim += _axesDim[i];
    }

    /// <summary>Total head dimension (sum of all axes dimensions, typically 128).</summary>
    public int HeadDim => _headDim;

    /// <summary>Precomputes cos/sin tables for the given position IDs. Call before Forward.</summary>
    /// <param name="posIds">Position IDs [totalSeqLen, numAxes] as float32. Text tokens: all zeros. Image tokens: [0, row, col].</param>
    public void Precompute(Tensor posIds)
    {
        int totalSeqLen = (int)posIds.Shape[0];
        int numAxes = (int)posIds.Shape[1];
        int halfDim = _headDim / 2;

        _cosCache = new float[totalSeqLen * halfDim];
        _sinCache = new float[totalSeqLen * halfDim];
        _cachedSeqLen = totalSeqLen;

        float* posPtr = (float*)posIds.DataPointer;
        int freqOffset = 0;

        // Max pairs across all axes (56/2=28 for Flux)
        int maxPairs = 0;
        for (int a = 0; a < numAxes; a++)
            maxPairs = Math.Max(maxPairs, _axesDim[a] / 2);
        Span<double> omega = stackalloc double[maxPairs];

        for (int axis = 0; axis < numAxes; axis++)
        {
            int axisDim = _axesDim[axis];
            int numPairs = axisDim / 2;

            // Precompute omega for this axis: omega[k] = 1 / (theta ^ (2k / axisDim))
            for (int k = 0; k < numPairs; k++)
            {
                double scale = (double)(2 * k) / axisDim;
                omega[k] = 1.0 / Math.Pow(_theta, scale);
            }

            for (int s = 0; s < totalSeqLen; s++)
            {
                double pos = posPtr[s * numAxes + axis];

                for (int k = 0; k < numPairs; k++)
                {
                    double angle = pos * omega[k];
                    int idx = s * halfDim + freqOffset + k;
                    _cosCache[idx] = (float)Math.Cos(angle);
                    _sinCache[idx] = (float)Math.Sin(angle);
                }
            }

            freqOffset += numPairs;
        }
    }

    /// <summary>Applies RoPE rotation to Q and K tensors in-place. Q/K must be [B, numHeads, seqLen, headDim] laid out as contiguous floats. Precompute must be called first with matching seqLen.</summary>
    /// <param name="q">Query tensor [B, numHeads, seqLen, headDim]. Modified in-place.</param>
    /// <param name="k">Key tensor [B, numHeads, seqLen, headDim]. Modified in-place.</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="seqLen">Sequence length (must match precomputed length).</param>
    public void Forward(Tensor q, Tensor k, int batch, int numHeads, int seqLen)
    {
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException("FluxRope.Precompute must be called before Forward.");

        int halfDim = _headDim / 2;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;

        fixed (float* cosPtr = _cosCache, sinPtr = _sinCache)
        {
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int vecOffset = ((b * numHeads + h) * seqLen + s) * _headDim;
                        int freqIdx = s * halfDim;

                        ApplyRotation(qPtr + vecOffset, cosPtr + freqIdx, sinPtr + freqIdx, halfDim);
                        ApplyRotation(kPtr + vecOffset, cosPtr + freqIdx, sinPtr + freqIdx, halfDim);
                    }
                }
            }
        }
    }

    /// <summary>Applies 2x2 rotation to each adjacent pair of elements in a vector. x[2i]' = cos * x[2i] - sin * x[2i+1], x[2i+1]' = sin * x[2i] + cos * x[2i+1].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyRotation(float* vec, float* cos, float* sin, int halfDim)
    {
        for (int i = 0; i < halfDim; i++)
        {
            float x0 = vec[2 * i];
            float x1 = vec[2 * i + 1];
            vec[2 * i] = cos[i] * x0 - sin[i] * x1;
            vec[2 * i + 1] = sin[i] * x0 + cos[i] * x1;
        }
    }

    /// <summary>Builds position IDs for Flux: text tokens get [0,0,0], image tokens get [0,row,col]. Returns a Tensor of shape [txtSeqLen + imgSeqLen, 3].</summary>
    /// <param name="txtSeqLen">Number of text tokens.</param>
    /// <param name="hPacked">Packed image height (latent_h / 2).</param>
    /// <param name="wPacked">Packed image width (latent_w / 2).</param>
    public static Tensor BuildPositionIds(int txtSeqLen, int hPacked, int wPacked)
    {
        int imgSeqLen = hPacked * wPacked;
        int totalSeqLen = txtSeqLen + imgSeqLen;
        TensorShape shape = new TensorShape(totalSeqLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);

        float* ptr = (float*)posIds.DataPointer;

        // Text tokens: all zeros (no spatial position)
        for (int i = 0; i < txtSeqLen * 3; i++)
            ptr[i] = 0f;

        // Image tokens: [0, row, col]
        for (int row = 0; row < hPacked; row++)
        {
            for (int col = 0; col < wPacked; col++)
            {
                int idx = txtSeqLen + row * wPacked + col;
                ptr[idx * 3 + 0] = 0f;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }
        }

        return posIds;
    }
}
