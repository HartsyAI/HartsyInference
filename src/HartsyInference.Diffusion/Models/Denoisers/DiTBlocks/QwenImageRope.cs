using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis rotary positional embedding for Qwen-Image (<c>QwenEmbedRope</c>). Splits the head dimension into <c>(frame, height, width)</c> sub-bands (default <c>[16, 56, 56]</c>) and rotates each pair of features by <c>theta = position * (1 / base^(2k/axisDim))</c>. Image tokens use CENTERED positions <c>[0, h-(H-H//2), w-(W-W//2)]</c> for a single-frame layout (<c>scale_rope=True</c>, which diffusers hardcodes). Text tokens use linearly-increasing positions starting at <c>max(H//2, W//2)</c>, mirroring diffusers' <c>txt_freqs = pos_freqs[max_vid_index : max_vid_index + max_txt_seq_len]</c>. The two streams are rotated separately before joint-attention concatenation. Apply order matches Flux's complex polar interpretation: pairs are <c>(real, imag)</c> at indices <c>(2i, 2i+1)</c>.</summary>
public sealed unsafe class QwenImageRope
{
    private readonly int[] _axesDim;
    private readonly int _theta;
    private readonly int _headDim;

    // Cached joint cos/sin tables in the [S, headDim] layout IBackend.WanRopeInterleaved reads (the angle for
    // pair i lives at index 2i; both slots filled). Position-only, so one host build serves every block of
    // every step — the per-block host ApplyJoint this replaces was a D2H+H2D round trip of joint Q AND K
    // (~50 MB each) per block. Keyed on the full sequence layout; a resolution/prompt-length change rebuilds.
    private (int ImgH, int ImgW, int Txt, int TxtStart, long RefSig) _jointTableKey =
        (-1, -1, -1, -1, -1);
    private Tensor? _jointCos;
    private Tensor? _jointSin;

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

        // scale_rope=True (diffusers QwenImageTransformer2DModel hardcodes it): spatial positions are CENTERED.
        // height = row - (H - H//2), width = col - (W - W//2); frame axis stays 0 (not centered). For even dims
        // this is row - H/2 / col - W/2. Centering matters for non-square images and the frame sub-band of cross-attn.
        int hCenter = hPacked - hPacked / 2;
        int wCenter = wPacked - wPacked / 2;
        for (int s = 0; s < imgSeqLen; s++)
        {
            int row = s / wPacked;
            int col = s - row * wPacked;
            FillTokenFreqs(cosTable, sinTable, s, frame: 0, height: row - hCenter, width: col - wCenter);
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

    /// <summary>Rotates Q and K in-place for the JOINT <c>[txt, img]</c> sequence in one pass — text occupies rows
    /// <c>0..txtSeqLen</c> (positions <c>positionStart + s</c> on all three axes), image occupies rows
    /// <c>txtSeqLen..txtSeqLen+H·W</c> (centered <c>[0, row-Hc, col-Wc]</c>). Bit-identical to calling
    /// <see cref="ApplyText"/> then <see cref="ApplyImage"/> on the separate streams before concatenation, since RoPE
    /// is per-row independent. Used by the GPU-resident block path, which concatenates <c>[txt, img]</c> on the GPU
    /// first (contiguous row-concat) and then ropes the joint tensor once — avoiding a per-stream CPU round-trip.
    /// Both Q and K must be <c>[B, numHeads, txtSeqLen + H·W (+ refH·refW), headDim]</c>.
    /// <para>Qwen-Image-Edit reference-latent tokens: when <paramref name="refPackedH"/>/<paramref name="refPackedW"/>
    /// are non-zero, an extra <c>refH·refW</c> token section follows the main image rows. Ref tokens use frame axis
    /// <paramref name="refFrameIndex"/> (ComfyUI <c>qwen_image/model.py</c> ref_latents "index" method: first ref = 1)
    /// and spatial positions centered on the REF grid — the reference image may have a different resolution than the
    /// output.</para></summary>
    public void ApplyJoint(Tensor q, Tensor k, int batch, int numHeads,
        int imgPackedH, int imgPackedW, int txtSeqLen, int txtPositionStart,
        ReadOnlySpan<(int H, int W)> refGrids = default)
    {
        int imgSeqLen = imgPackedH * imgPackedW;
        int refSeqLen = 0;
        foreach ((int rh, int rw) in refGrids) refSeqLen += rh * rw;
        int totalSeqLen = txtSeqLen + imgSeqLen + refSeqLen;
        int halfDim = _headDim / 2;
        float[] cosTable = new float[totalSeqLen * halfDim];
        float[] sinTable = new float[totalSeqLen * halfDim];

        FillJointFreqs(cosTable, sinTable, imgPackedH, imgPackedW, txtSeqLen, txtPositionStart, refGrids);
        ApplyRotationBatched(q, k, cosTable, sinTable, batch, numHeads, totalSeqLen);
    }

    /// <summary>Fills the joint-layout frequency tables: text rows (positions <c>txtPositionStart + s</c>),
    /// centered main-image rows (frame 0), then each reference grid in order with frame axis <c>i + 1</c>
    /// (ComfyUI ref_latents "index"/"index_timestep_zero" methods increment the frame per ref) and spatial
    /// positions centered on that reference's own grid.</summary>
    private void FillJointFreqs(Span<float> cosTable, Span<float> sinTable,
        int imgPackedH, int imgPackedW, int txtSeqLen, int txtPositionStart, ReadOnlySpan<(int H, int W)> refGrids)
    {
        for (int s = 0; s < txtSeqLen; s++)
        {
            int pos = txtPositionStart + s;
            FillTokenFreqs(cosTable, sinTable, s, frame: pos, height: pos, width: pos);
        }
        int imgSeqLen = imgPackedH * imgPackedW;
        int hCenter = imgPackedH - imgPackedH / 2;
        int wCenter = imgPackedW - imgPackedW / 2;
        for (int si = 0; si < imgSeqLen; si++)
        {
            int row = si / imgPackedW;
            int col = si - row * imgPackedW;
            FillTokenFreqs(cosTable, sinTable, txtSeqLen + si, frame: 0, height: row - hCenter, width: col - wCenter);
        }
        int rowBase = txtSeqLen + imgSeqLen;
        for (int r = 0; r < refGrids.Length; r++)
        {
            (int refH, int refW) = refGrids[r];
            int refHCenter = refH - refH / 2;
            int refWCenter = refW - refW / 2;
            int refSeq = refH * refW;
            for (int si = 0; si < refSeq; si++)
            {
                int row = si / refW;
                int col = si - row * refW;
                FillTokenFreqs(cosTable, sinTable, rowBase + si,
                    frame: r + 1, height: row - refHCenter, width: col - refWCenter);
            }
            rowBase += refSeq;
        }
    }

    /// <summary>Builds (or returns the cached) joint cos/sin tables for the device rope path, <c>[S, headDim]</c>
    /// F32 in the <see cref="Core.Backends.IBackend.WanRopeInterleaved"/> convention (pair i's angle at index
    /// <c>2i</c>). Same position math as <see cref="ApplyJoint"/> — the rotation itself matches the interleaved
    /// kernel exactly (<c>x0·cos−x1·sin, x0·sin+x1·cos</c> on pairs (2i, 2i+1)).</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildJointTables(int imgPackedH, int imgPackedW, int txtSeqLen,
        int txtPositionStart, ReadOnlySpan<(int H, int W)> refGrids = default)
    {
        long refSig = 0x51F0;
        int refSeqLen = 0;
        foreach ((int rh, int rw) in refGrids)
        {
            refSig = refSig * 1000003L ^ ((long)rh << 20 | (uint)rw);
            refSeqLen += rh * rw;
        }
        (int, int, int, int, long) key = (imgPackedH, imgPackedW, txtSeqLen, txtPositionStart, refSig);
        if (_jointCos is not null && _jointSin is not null && _jointTableKey == key)
            return (_jointCos, _jointSin);
        _jointCos?.Dispose();
        _jointSin?.Dispose();

        int imgSeqLen = imgPackedH * imgPackedW;
        int totalSeqLen = txtSeqLen + imgSeqLen + refSeqLen;
        int halfDim = _headDim / 2;
        float[] cosTable = new float[totalSeqLen * halfDim];
        float[] sinTable = new float[totalSeqLen * halfDim];
        FillJointFreqs(cosTable, sinTable, imgPackedH, imgPackedW, txtSeqLen, txtPositionStart, refGrids);

        Tensor cos = new Tensor(new TensorShape(totalSeqLen, _headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(totalSeqLen, _headDim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        for (int s = 0; s < totalSeqLen; s++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                float c = cosTable[s * halfDim + i];
                float sn = sinTable[s * halfDim + i];
                long off = (long)s * _headDim + 2 * i;
                cp[off] = c;
                cp[off + 1] = c;
                sp[off] = sn;
                sp[off + 1] = sn;
            }
        }
        _jointCos = cos;
        _jointSin = sin;
        _jointTableKey = key;
        return (cos, sin);
    }

    /// <summary>Computes the position offset to use when calling <see cref="ApplyText"/>. Matches diffusers'
    /// <c>scale_rope=True</c> mode where text starts at <c>max_vid_index = max(height//2, width//2)</c>
    /// (the centered image grid's max index), used as the single scalar across all three axes for text.</summary>
    public static int ComputeTextPositionStart(int hPacked, int wPacked) => Math.Max(hPacked / 2, wPacked / 2);

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
