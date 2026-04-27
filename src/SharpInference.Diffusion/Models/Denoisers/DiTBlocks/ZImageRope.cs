using System.Runtime.CompilerServices;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Multi-axis rotary position embedding for Z-Image (Lumina2/NextDiT). Splits the head dimension across 3 axes [frame, height, width] = [32, 48, 48] with θ=256. Caption tokens get position [0,0,0]; image tokens get [0, row, col]. Caption pad tokens and image pad tokens use the same axis-0=0 fallback so RoPE is well-defined for the entire concatenated sequence.</summary>
public sealed unsafe class ZImageRope
{
    private readonly int[] _axesDim;
    private readonly float _theta;
    private readonly int _headDim;

    private float[]? _cosCache;
    private float[]? _sinCache;
    private int _cachedSeqLen;

    /// <summary>Creates a Z-Image RoPE with the given axis dims and base.</summary>
    /// <param name="axesDim">Per-axis dimensions. Must sum to head_dim. Default [32, 48, 48] (sums to 128).</param>
    /// <param name="theta">Base frequency. Default 256 (Z-Image; much smaller than Flux's 10000).</param>
    public ZImageRope(int[]? axesDim = null, float theta = 256.0f)
    {
        _axesDim = axesDim ?? [32, 48, 48];
        _theta = theta;
        _headDim = 0;
        for (int i = 0; i < _axesDim.Length; i++)
            _headDim += _axesDim[i];
    }

    /// <summary>Total head dimension (sum of all axes, 128 for Z-Image).</summary>
    public int HeadDim => _headDim;

    /// <summary>Precomputes per-token cos/sin tables across all 3 axes for the given position IDs. Call before <see cref="Forward"/>.</summary>
    /// <param name="posIds">Position IDs [totalSeqLen, numAxes] as F32. For Z-Image, axes are [frame=0, row, col] for image tokens; all-zero for caption + pad tokens.</param>
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

        int maxPairs = 0;
        for (int a = 0; a < numAxes; a++)
            maxPairs = Math.Max(maxPairs, _axesDim[a] / 2);
        Span<double> omega = stackalloc double[maxPairs];

        for (int axis = 0; axis < numAxes; axis++)
        {
            int axisDim = _axesDim[axis];
            int numPairs = axisDim / 2;

            // omega[k] = 1 / (theta ^ (2k / axisDim))
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

    /// <summary>Applies RoPE rotation to Q/K in-place. Q/K shape [B, numHeads, seqLen, headDim]. <see cref="Precompute"/> must have been called with matching seqLen.</summary>
    public void Forward(Tensor q, Tensor k, int batch, int numHeads, int seqLen)
    {
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException("ZImageRope.Precompute must be called before Forward.");

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

    /// <summary>Builds position IDs for Z-Image's concatenated [refined_caption, refined_image] sequence. Caption tokens (real + pad): all zeros. Image tokens (real + pad): [0, row, col] for real, [0, 0, 0] for pad. Returns [capPad + imgPad, 3] F32 tensor.</summary>
    /// <param name="capPaddedLen">Padded caption sequence length (after pad-to-multiple-of-32).</param>
    /// <param name="hPacked">Image height in patch units (latent_h / patch_size).</param>
    /// <param name="wPacked">Image width in patch units (latent_w / patch_size).</param>
    /// <param name="imgPaddedLen">Padded image sequence length (real h*w padded up to multiple of 32 with x_pad_token).</param>
    public static Tensor BuildPositionIds(int capPaddedLen, int hPacked, int wPacked, int imgPaddedLen)
    {
        int imgRealLen = hPacked * wPacked;
        if (imgRealLen > imgPaddedLen)
            throw new ArgumentException($"imgRealLen={imgRealLen} cannot exceed imgPaddedLen={imgPaddedLen}", nameof(imgPaddedLen));

        int totalLen = capPaddedLen + imgPaddedLen;
        TensorShape shape = new TensorShape(totalLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);

        float* ptr = (float*)posIds.DataPointer;
        for (int i = 0; i < totalLen * 3; i++)
            ptr[i] = 0f;

        // Image tokens start at offset capPaddedLen. Real tokens get [0, row, col]; pad tokens stay at [0,0,0].
        for (int row = 0; row < hPacked; row++)
        {
            for (int col = 0; col < wPacked; col++)
            {
                int idx = capPaddedLen + row * wPacked + col;
                ptr[idx * 3 + 0] = 0f;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }
        }

        return posIds;
    }
}
