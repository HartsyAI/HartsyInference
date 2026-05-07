using System.Runtime.CompilerServices;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>2-axis rotary position embedding for Hunyuan Image 2.1 (<c>HunyuanImageRotaryPosEmbed</c>). Splits the per-head dimension into <c>(height, width)</c> sub-bands (default <c>[64, 64]</c>) and rotates each pair of features by <c>theta = position * (1 / base^(2k/axisDim))</c>. Applied to image tokens only — text tokens are kept unrotated and concatenated after rotation. Each token at packed-grid <c>(row, col)</c> uses position <c>[row, col]</c>; both axes contribute <c>axesDim/2</c> cos/sin pairs to the head vector. Diffusers uses <c>theta=256</c> by default and <c>get_1d_rotary_pos_embed</c> with <c>use_real=True</c> producing <c>(cos, sin)</c> tables that are concatenated along the feature axis before being applied as the standard <c>(real, imag)</c> 2x2 rotation; this implementation produces the same numerical output by composing the per-axis tables in the same axis order.</summary>
public sealed unsafe class HunyuanImageRope
{
    private readonly int[] _axesDim;
    private readonly float _theta;
    private readonly int _headDim;

    /// <summary>Creates a HunyuanImageRope.</summary>
    /// <param name="axesDim">Per-axis dim split <c>(height, width)</c>. Each must be even. Default <c>[64, 64]</c>.</param>
    /// <param name="theta">RoPE base frequency. Default 256.</param>
    public HunyuanImageRope(int[]? axesDim = null, float theta = 256.0f)
    {
        _axesDim = axesDim ?? [64, 64];
        if (_axesDim.Length != 2)
            throw new ArgumentException("HunyuanImageRope requires exactly 2 axes (height, width).", nameof(axesDim));
        _theta = theta;
        _headDim = 0;
        for (int i = 0; i < _axesDim.Length; i++)
            _headDim += _axesDim[i];
    }

    /// <summary>Sum of per-axis dimensions; equals the per-head dim of the wrapped attention block.</summary>
    public int HeadDim => _headDim;

    /// <summary>Per-axis dim split <c>(height, width)</c>. Each must be even — pairs of features get rotated together.</summary>
    public ReadOnlySpan<int> AxesDim => _axesDim;

    /// <summary>Rotates Q and K in-place for image tokens only. Tokens are laid out row-major over <paramref name="hPacked"/> × <paramref name="wPacked"/>; token at index <c>r * W + c</c> uses position <c>(r, c)</c>. Both Q and K must be <c>[B, numHeads, imgSeqLen, headDim]</c> and <c>imgSeqLen == hPacked * wPacked</c>.</summary>
    public void ApplyImage(Tensor q, Tensor k, int batch, int numHeads, int hPacked, int wPacked)
    {
        int imgSeqLen = hPacked * wPacked;
        int halfDim = _headDim / 2;
        float[] cosTable = new float[imgSeqLen * halfDim];
        float[] sinTable = new float[imgSeqLen * halfDim];

        for (int s = 0; s < imgSeqLen; s++)
        {
            int row = s / wPacked;
            int col = s - row * wPacked;
            FillTokenFreqs(cosTable, sinTable, s, height: row, width: col);
        }

        ApplyRotationBatched(q, k, cosTable, sinTable, batch, numHeads, imgSeqLen);
    }

    private void FillTokenFreqs(Span<float> cosTable, Span<float> sinTable, int seqIdx,
        double height, double width)
    {
        int halfDim = _headDim / 2;
        int destOffset = seqIdx * halfDim;
        int freqOffset = 0;

        Span<double> positions = stackalloc double[2] { height, width };
        for (int axis = 0; axis < 2; axis++)
        {
            int axisDim = _axesDim[axis];
            int numPairs = axisDim / 2;
            double pos = positions[axis];

            for (int kIdx = 0; kIdx < numPairs; kIdx++)
            {
                double inv = 1.0 / Math.Pow(_theta, (double)(2 * kIdx) / axisDim);
                double angle = pos * inv;
                cosTable[destOffset + freqOffset + kIdx] = (float)Math.Cos(angle);
                sinTable[destOffset + freqOffset + kIdx] = (float)Math.Sin(angle);
            }
            freqOffset += numPairs;
        }
    }

    private void ApplyRotationBatched(Tensor q, Tensor k, ReadOnlySpan<float> cosTable, ReadOnlySpan<float> sinTable,
        int batch, int numHeads, int seqLen)
    {
        int halfDim = _headDim / 2;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;

        fixed (float* cosPtr = cosTable, sinPtr = sinTable)
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
}
