using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis rotary positional embedding for OmniGen2 (<c>OmniGen2RotaryPosEmbed</c> from the upstream Python
/// repo). Splits the head dim into <c>(time, height, width)</c> sub-bands (default <c>[40, 40, 40]</c> summing to 120) and
/// rotates each pair of features by <c>theta = position * (1 / base^(2k/axisDim))</c>. Position assignments for the
/// text-to-image path differ from Flux/Qwen-Image:
/// <list type="bullet">
/// <item>Text token <c>i</c> uses <c>(i, i, i)</c> — same scalar across all 3 axes (matching diffusers' <c>repeat "l -&gt; l 3"</c>).</item>
/// <item>Image token at packed grid <c>(row, col)</c> uses <c>(text_seq_len, row, col)</c> — the time axis is offset
/// by the caption length (matching <c>pe_shift = cap_seq_len</c> in the upstream repo).</item>
/// </list>
/// In the joint <c>layers</c> stack the merged sequence carries different positions per token (text positions for
/// the first <c>txt_seq_len</c> tokens, image positions for the rest); the per-token cos/sin tables built by
/// <see cref="BuildJointTable"/> capture that exactly so a single rotation pass works for the joint stream.
/// Rotation pair layout matches Flux/Qwen-Image: <c>(real, imag) = vec[2i], vec[2i+1]</c>.
/// <para>Editing/reference-image positions (<c>l_effective_ref_img_len</c> path) are intentionally not modeled
/// here — this implementation is t2i-only.</para></summary>
public sealed unsafe class OmniGen2Rope
{
    private readonly int[] _axesDim;
    private readonly int _theta;
    private readonly int _headDim;

    /// <summary>Creates a 3-axis OmniGen2 RoPE.</summary>
    /// <param name="axesDim">Per-axis dimensions <c>(time, height, width)</c>; each must be even. Sum must equal head_dim.</param>
    /// <param name="theta">Base frequency. 10000 for OmniGen2.</param>
    public OmniGen2Rope(int[]? axesDim = null, int theta = 10000)
    {
        _axesDim = axesDim ?? [40, 40, 40];
        if (_axesDim.Length != 3)
            throw new ArgumentException("OmniGen2Rope requires exactly 3 axes (time, height, width).", nameof(axesDim));
        for (int i = 0; i < _axesDim.Length; i++)
            if ((_axesDim[i] & 1) != 0)
                throw new ArgumentException($"axesDim[{i}] = {_axesDim[i]} must be even (RoPE rotates pairs of features).");
        _theta = theta;
        _headDim = _axesDim[0] + _axesDim[1] + _axesDim[2];
    }

    /// <summary>Sum of the per-axis dimensions; equals the per-head dim of the wrapped attention block.</summary>
    public int HeadDim => _headDim;

    /// <summary>Builds an interleaved <c>(cos, sin)</c> table sized <c>[seqLen * headDim/2]</c> for a text-only
    /// sequence: token <c>s</c> uses position <c>(s, s, s)</c>. Used by <c>context_refiner</c> blocks.</summary>
    public (float[] cos, float[] sin) BuildTextTable(int txtSeqLen)
    {
        int halfDim = _headDim / 2;
        float[] cos = new float[txtSeqLen * halfDim];
        float[] sin = new float[txtSeqLen * halfDim];
        for (int s = 0; s < txtSeqLen; s++)
            FillTokenFreqs(cos, sin, s, time: s, height: s, width: s);
        return (cos, sin);
    }

    /// <summary>Builds an interleaved <c>(cos, sin)</c> table for an image-only sequence: token at packed grid
    /// <c>(r, c)</c> uses position <c>(timeOffset, r, c)</c>. Used by <c>noise_refiner</c> blocks.</summary>
    public (float[] cos, float[] sin) BuildImageTable(int hPacked, int wPacked, int timeOffset)
    {
        int imgSeqLen = hPacked * wPacked;
        int halfDim = _headDim / 2;
        float[] cos = new float[imgSeqLen * halfDim];
        float[] sin = new float[imgSeqLen * halfDim];
        for (int s = 0; s < imgSeqLen; s++)
        {
            int row = s / wPacked;
            int col = s - row * wPacked;
            FillTokenFreqs(cos, sin, s, time: timeOffset, height: row, width: col);
        }
        return (cos, sin);
    }

    /// <summary>Builds the joint-sequence table for the main <c>layers</c> stack: the first
    /// <paramref name="txtSeqLen"/> tokens use text positions <c>(s, s, s)</c>; the remaining
    /// <c>hPacked * wPacked</c> tokens use image positions <c>(txtSeqLen, row, col)</c>. The returned tables are
    /// sized for the full joint sequence so a single <see cref="Apply"/> pass rotates Q/K for the entire merged
    /// stream.</summary>
    public (float[] cos, float[] sin) BuildJointTable(int txtSeqLen, int hPacked, int wPacked)
    {
        int imgSeqLen = hPacked * wPacked;
        int totalSeqLen = txtSeqLen + imgSeqLen;
        int halfDim = _headDim / 2;
        float[] cos = new float[totalSeqLen * halfDim];
        float[] sin = new float[totalSeqLen * halfDim];

        for (int s = 0; s < txtSeqLen; s++)
            FillTokenFreqs(cos, sin, s, time: s, height: s, width: s);

        for (int s = 0; s < imgSeqLen; s++)
        {
            int row = s / wPacked;
            int col = s - row * wPacked;
            FillTokenFreqs(cos, sin, txtSeqLen + s, time: txtSeqLen, height: row, width: col);
        }

        return (cos, sin);
    }

    /// <summary>Builds the text table and rotates both Q and K in-place. GQA-aware: Q is rotated with its
    /// own head count, K with the (smaller) KV head count, both using the same RoPE table.</summary>
    public void ApplyText(Tensor q, Tensor k, int batch, int numQHeads, int numKvHeads, int seqLen)
    {
        (float[] cos, float[] sin) = BuildTextTable(seqLen);
        Apply(q, cos, sin, batch, numQHeads, seqLen);
        Apply(k, cos, sin, batch, numKvHeads, seqLen);
    }

    /// <summary>Builds the image table and rotates both Q and K in-place. GQA-aware: Q is rotated with its
    /// own head count, K with the (smaller) KV head count, both using the same RoPE table.</summary>
    public void ApplyImage(Tensor q, Tensor k, int batch, int numQHeads, int numKvHeads,
        int hPacked, int wPacked, int timeOffset)
    {
        int imgSeqLen = hPacked * wPacked;
        (float[] cos, float[] sin) = BuildImageTable(hPacked, wPacked, timeOffset);
        Apply(q, cos, sin, batch, numQHeads, imgSeqLen);
        Apply(k, cos, sin, batch, numKvHeads, imgSeqLen);
    }

    /// <summary>Builds the joint-sequence table and rotates both Q and K in-place over the merged
    /// <c>[txt, img]</c> sequence. Text positions <c>(s, s, s)</c> for the first <paramref name="txtSeqLen"/>
    /// tokens; image positions <c>(txtSeqLen, row, col)</c> for the remaining <c>hPacked * wPacked</c> tokens.
    /// GQA-aware.</summary>
    public void ApplyJoint(Tensor q, Tensor k, int batch, int numQHeads, int numKvHeads,
        int txtSeqLen, int hPacked, int wPacked)
    {
        int totalSeqLen = txtSeqLen + hPacked * wPacked;
        (float[] cos, float[] sin) = BuildJointTable(txtSeqLen, hPacked, wPacked);
        Apply(q, cos, sin, batch, numQHeads, totalSeqLen);
        Apply(k, cos, sin, batch, numKvHeads, totalSeqLen);
    }

    /// <summary>Rotates Q and K in-place using a precomputed <c>(cos, sin)</c> table sized to the sequence length.
    /// Q and K must be <c>[B, numHeads, seqLen, headDim]</c>. Use this once per block per stream — the table is
    /// independent of head count, so a single table can rotate both Q (with full Q heads) and K (with KV heads).</summary>
    public void Apply(Tensor qOrK, ReadOnlySpan<float> cosTable, ReadOnlySpan<float> sinTable,
        int batch, int numHeads, int seqLen)
    {
        int halfDim = _headDim / 2;
        float* ptr = (float*)qOrK.DataPointer;

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
                        ApplyRotation(ptr + vecOffset, cosPtr + freqIdx, sinPtr + freqIdx, halfDim);
                    }
                }
            }
        }
    }

    private void FillTokenFreqs(Span<float> cosTable, Span<float> sinTable, int seqIdx,
        double time, double height, double width)
    {
        int halfDim = _headDim / 2;
        int destOffset = seqIdx * halfDim;
        int freqOffset = 0;

        Span<double> positions = stackalloc double[3] { time, height, width };
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
