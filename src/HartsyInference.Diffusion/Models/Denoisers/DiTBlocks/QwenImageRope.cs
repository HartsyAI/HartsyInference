using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis rotary positional embedding for Qwen-Image (<c>QwenEmbedRope</c>). Splits the head dimension into <c>(frame, height, width)</c> sub-bands (default <c>[16, 56, 56]</c>) and rotates each pair of features by <c>theta = position * (1 / base^(2k/axisDim))</c>. Image tokens are positioned at <c>[0, h, w]</c> for a single-frame layout (<c>scale_rope=False</c>: positions span <c>[0, H)</c> and <c>[0, W)</c>). Text tokens use linearly-increasing positions starting at <c>max(H, W)</c>, mirroring diffusers' <c>txt_freqs = pos_freqs[max_vid_index : max_vid_index + max_txt_seq_len]</c>. The two streams are rotated separately before joint-attention concatenation. Apply order matches Flux's complex polar interpretation: pairs are <c>(real, imag)</c> at indices <c>(2i, 2i+1)</c>.</summary>
public sealed unsafe class QwenImageRope
{
    private readonly int[] _axesDim;
    private readonly int _theta;
    private readonly int _headDim;

    public QwenImageRope(int[]? axesDim = null, int theta = 10000)
    {
        _axesDim = axesDim ?? [16, 56, 56];
        if (_axesDim.Length != 3)
            throw new ArgumentException("QwenImageRope requires exactly 3 axes (frame, height, width).", nameof(axesDim));
        _theta = theta;
        _headDim = 0;
        for (int i = 0; i < _axesDim.Length; i++)
            _headDim += _axesDim[i];
    }

    /// <summary>Sum of the per-axis dimensions; equals the per-head dim of the wrapped attention block.</summary>
    public int HeadDim => _headDim;

    /// <summary>Per-axis dim split: <c>(frame, height, width)</c>. Each must be even — pairs of features get rotated together.</summary>
    public ReadOnlySpan<int> AxesDim => _axesDim;

    /// <summary>Rotates Q and K in-place for image tokens. Tokens are laid out row-major over <paramref name="hPacked"/> × <paramref name="wPacked"/>; token <c>r * W + c</c> uses position <c>[0, r, c]</c>. Both Q and K must be <c>[B, numHeads, imgSeqLen, headDim]</c> and <c>imgSeqLen == hPacked * wPacked</c>.</summary>
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
            FillTokenFreqs(cosTable, sinTable, s, frame: 0, height: row, width: col);
        }

        ApplyRotationBatched(q, k, cosTable, sinTable, batch, numHeads, imgSeqLen);
    }

    /// <summary>Rotates Q and K in-place for text tokens. Each text position <paramref name="positionStart"/> + s uses <c>[positionStart + s, positionStart + s, positionStart + s]</c> (diffusers <c>pos_freqs</c> is the same scalar across the three axes for the text region). Both Q and K must be <c>[B, numHeads, txtSeqLen, headDim]</c>.</summary>
    public void ApplyText(Tensor q, Tensor k, int batch, int numHeads, int txtSeqLen, int positionStart)
    {
        int halfDim = _headDim / 2;
        float[] cosTable = new float[txtSeqLen * halfDim];
        float[] sinTable = new float[txtSeqLen * halfDim];

        for (int s = 0; s < txtSeqLen; s++)
        {
            int pos = positionStart + s;
            FillTokenFreqs(cosTable, sinTable, s, frame: pos, height: pos, width: pos);
        }

        ApplyRotationBatched(q, k, cosTable, sinTable, batch, numHeads, txtSeqLen);
    }

    /// <summary>Computes the position offset to use when calling <see cref="ApplyText"/>. Matches diffusers'
    /// <c>scale_rope=False</c> mode where text starts at <c>max_vid_index = max(height, width)</c> after the image grid.</summary>
    public static int ComputeTextPositionStart(int hPacked, int wPacked) => Math.Max(hPacked, wPacked);

    private void FillTokenFreqs(Span<float> cosTable, Span<float> sinTable, int seqIdx,
        double frame, double height, double width)
    {
        int halfDim = _headDim / 2;
        int destOffset = seqIdx * halfDim;
        int freqOffset = 0;

        Span<double> positions = stackalloc double[3] { frame, height, width };
        for (int axis = 0; axis < 3; axis++)
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
