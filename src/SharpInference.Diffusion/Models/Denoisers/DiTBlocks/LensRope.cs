using System.Runtime.CompilerServices;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis complex-polar rotary positional embedding for Microsoft Lens (<c>LensEmbedRope</c>). Splits the head dimension into <c>(frame, height, width)</c> sub-bands (default <c>[8, 28, 28]</c> summing to 64 = head_dim) and rotates each pair of features by <c>θ = position * (1 / base^(2k/axisDim))</c>. <b>Differs from QwenImageRope in two ways:</b> (1) <c>scale_rope=True</c> hardcoded — height and width positions are split into negative+positive halves centered around 0 (top/left rows get <c>arange(4096).flip(0) * -1 - 1</c>, bottom/right rows get <c>arange(4096)</c>), centering the rotation around the image's middle pixel rather than the top-left; (2) text RoPE positions start at <c>max(h//2, w//2)</c> (not <c>max(h, w)</c>) because of the height/width halving. Apply order matches QwenImageRope's complex polar pairs: features at indices <c>(2i, 2i+1)</c> form one complex number that gets multiplied by <c>e^(iθ)</c>.</summary>
public sealed unsafe class LensRope
{
    private readonly int[] _axesDim;
    private readonly int _theta;
    private readonly int _headDim;

    /// <summary>Creates a Lens RoPE with the given axis split. Lens always uses <c>scale_rope=True</c> — height/width positions are split into negative+positive halves around 0. Frame axis stays at <c>[0]</c> for image-only inference (frame=1).</summary>
    public LensRope(int[]? axesDim = null, int theta = 10000)
    {
        _axesDim = axesDim ?? [8, 28, 28];
        if (_axesDim.Length != 3)
            throw new ArgumentException("LensRope requires exactly 3 axes (frame, height, width).", nameof(axesDim));
        _theta = theta;
        _headDim = 0;
        for (int i = 0; i < _axesDim.Length; i++)
            _headDim += _axesDim[i];
    }

    /// <summary>Sum of per-axis dims; equals the per-head dim of the wrapped attention block (64 for Lens default).</summary>
    public int HeadDim => _headDim;

    /// <summary>Per-axis dim split: <c>(frame, height, width)</c>.</summary>
    public ReadOnlySpan<int> AxesDim => _axesDim;

    /// <summary>Rotates Q and K in-place for image tokens in <c>scale_rope=True</c> mode. With <paramref name="hPacked"/>=H, the top <c>H - H/2</c> rows use positions <c>[-H+H/2, ..., -1]</c> and the bottom <c>H/2</c> rows use positions <c>[0, ..., H/2 - 1]</c> (Microsoft's centered convention). Same split applies to width. Token <c>r*W + c</c> uses position <c>[0, row_pos, col_pos]</c>. Both Q and K must be <c>[B, numHeads, imgSeqLen, headDim]</c>.</summary>
    public void ApplyImage(Tensor q, Tensor k, int batch, int numHeads, int hPacked, int wPacked)
    {
        int imgSeqLen = hPacked * wPacked;
        int halfDim = _headDim / 2;
        float[] cosTable = new float[imgSeqLen * halfDim];
        float[] sinTable = new float[imgSeqLen * halfDim];

        // Centered position lookup tables: row 0..hPacked-1 → centered position.
        // First (hPacked - hPacked/2) rows get negative positions, last hPacked/2 rows get positive.
        int[] rowPositions = ComputeCenteredPositions(hPacked);
        int[] colPositions = ComputeCenteredPositions(wPacked);

        for (int s = 0; s < imgSeqLen; s++)
        {
            int row = s / wPacked;
            int col = s - row * wPacked;
            FillTokenFreqs(cosTable, sinTable, s, frame: 0, height: rowPositions[row], width: colPositions[col]);
        }

        ApplyRotationBatched(q, k, cosTable, sinTable, batch, numHeads, imgSeqLen);
    }

    /// <summary>Rotates Q and K in-place for text tokens. With <c>scale_rope=True</c> text positions start at <c>max(hPacked/2, wPacked/2)</c> to live past the image's positive-half range. All three axes (frame, h, w) get the same scalar position per text token (upstream uses <c>pos_freqs</c> directly on the text region with no axis split).</summary>
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

    /// <summary>Computes the position offset for text tokens in <c>scale_rope=True</c> mode. Matches upstream's <c>max_vid_index = max(height // 2, width // 2)</c>.</summary>
    public static int ComputeTextPositionStart(int hPacked, int wPacked) => Math.Max(hPacked / 2, wPacked / 2);

    private static int[] ComputeCenteredPositions(int len)
    {
        // Upstream LensEmbedRope._compute_video_freqs (scale_rope=True branch):
        //   freqs_height = cat([freqs_neg[1][-(H - H//2):], freqs_pos[1][:H//2]], dim=0)
        // where freqs_neg uses indices [-1, -2, ..., -(H - H//2)] (the last (H - H//2) of the reversed negative range)
        // and freqs_pos uses indices [0, 1, ..., H//2 - 1].
        // Net effect: rows 0..(H - H//2 - 1) get positions [-(H - H//2), ..., -1]
        //             rows (H - H//2)..H-1 get positions [0, ..., H//2 - 1]
        int[] positions = new int[len];
        int halfLen = len / 2;
        int negRows = len - halfLen;
        for (int i = 0; i < negRows; i++)
            positions[i] = -(negRows - i);
        for (int i = 0; i < halfLen; i++)
            positions[negRows + i] = i;
        return positions;
    }

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
