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
public sealed unsafe class OmniGen2Rope : IDisposable
{
    private readonly int[] _axesDim;
    private readonly int _theta;
    private readonly int _headDim;

    // Device rope-table caches (WanRopeInterleaved [S, headDim] layout). Positions are step-invariant and
    // size-invariant, so a table built once is reused across every block and every denoise step of a generation
    // (and across generations at the same geometry) — this is what turns the per-block host Q/K drain into a
    // single device kernel. One slot per stream type used concurrently in a forward (text / image / joint).
    private Tensor? _textCos, _textSin;
    private int _textKey = -1;
    private Tensor? _imageCos, _imageSin;
    private (int h, int w, int t) _imageKey = (-1, -1, -1);
    private Tensor? _jointCos, _jointSin;
    private (int txt, int h, int w) _jointKey = (-1, -1, -1);

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

    /// <summary>Builds an interleaved <c>(cos, sin)</c> table from explicit per-token position ids
    /// <c>(time, height, width)</c>. Generalizes <see cref="BuildTextTable"/>/<see cref="BuildImageTable"/>/
    /// <see cref="BuildJointTable"/> to arbitrary layouts — used by Boogu-Image's edit path, where reference and
    /// noise image tokens carry running time-axis offsets (<c>pe_shift</c>) that the fixed builders don't cover.
    /// The three id spans must have equal length (the sequence length).</summary>
    public (float[] cos, float[] sin) BuildTableFromPositions(
        ReadOnlySpan<int> timeIds, ReadOnlySpan<int> heightIds, ReadOnlySpan<int> widthIds)
    {
        int seqLen = timeIds.Length;
        if (heightIds.Length != seqLen || widthIds.Length != seqLen)
            throw new ArgumentException("OmniGen2Rope.BuildTableFromPositions requires equal-length id spans.");
        int halfDim = _headDim / 2;
        float[] cos = new float[seqLen * halfDim];
        float[] sin = new float[seqLen * halfDim];
        for (int s = 0; s < seqLen; s++)
            FillTokenFreqs(cos, sin, s, timeIds[s], heightIds[s], widthIds[s]);
        return (cos, sin);
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

    /// <summary>Expands a host <c>[S, halfDim]</c> pair-angle table (from <see cref="BuildTableFromPositions"/> /
    /// the fixed builders) into the <c>[S, headDim]</c> F32 layout <c>IBackend.WanRopeInterleaved</c> reads — pair
    /// <c>i</c>'s angle duplicated at indices <c>2i</c> and <c>2i+1</c>. That kernel's rotation
    /// (<c>re·c − im·s, im·c + re·s</c> on adjacent pairs) matches <see cref="ApplyRotation"/> exactly, so applying it
    /// pre-permute on <c>[1, S, H, D]</c> Q/K is bit-equivalent to the host <see cref="Apply"/> — minus the per-block
    /// D2H drain + re-upload of Q and K the host loop forced.</summary>
    public (Tensor cos, Tensor sin) ExpandToDeviceTables(float[] cosTable, float[] sinTable)
    {
        int halfDim = _headDim / 2;
        int seqLen = cosTable.Length / halfDim;
        Tensor cos = new Tensor(new TensorShape(seqLen, _headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(seqLen, _headDim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        for (int s = 0; s < seqLen; s++)
        {
            long rowOff = (long)s * _headDim;
            int srcOff = s * halfDim;
            for (int i = 0; i < halfDim; i++)
            {
                float cv = cosTable[srcOff + i];
                float sv = sinTable[srcOff + i];
                cp[rowOff + 2 * i] = cv;
                cp[rowOff + 2 * i + 1] = cv;
                sp[rowOff + 2 * i] = sv;
                sp[rowOff + 2 * i + 1] = sv;
            }
        }
        return (cos, sin);
    }

    /// <summary>Returns the cached device <c>(cos, sin)</c> tables for a text-only stream (positions <c>(s,s,s)</c>),
    /// building them on first use. <c>[txtSeqLen, headDim]</c> in the <see cref="Core.Backends.IBackend.WanRopeInterleaved"/>
    /// convention — bit-equivalent to the host <see cref="ApplyText"/> path minus the per-block Q/K D2H drain.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildTextTables(int txtSeqLen)
    {
        if (_textCos is not null && _textSin is not null && _textKey == txtSeqLen)
            return (_textCos, _textSin);
        _textCos?.Dispose();
        _textSin?.Dispose();
        (float[] cos, float[] sin) = BuildTextTable(txtSeqLen);
        (_textCos, _textSin) = ExpandToDeviceTables(cos, sin);
        _textKey = txtSeqLen;
        return (_textCos, _textSin);
    }

    /// <summary>Returns the cached device <c>(cos, sin)</c> tables for an image-only stream (positions
    /// <c>(timeOffset, row, col)</c>), building them on first use. <c>[hPacked·wPacked, headDim]</c>.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildImageTables(int hPacked, int wPacked, int timeOffset)
    {
        (int, int, int) key = (hPacked, wPacked, timeOffset);
        if (_imageCos is not null && _imageSin is not null && _imageKey == key)
            return (_imageCos, _imageSin);
        _imageCos?.Dispose();
        _imageSin?.Dispose();
        (float[] cos, float[] sin) = BuildImageTable(hPacked, wPacked, timeOffset);
        (_imageCos, _imageSin) = ExpandToDeviceTables(cos, sin);
        _imageKey = key;
        return (_imageCos, _imageSin);
    }

    /// <summary>Returns the cached device <c>(cos, sin)</c> tables for the joint <c>[text, image]</c> stream,
    /// building them on first use. <c>[txtSeqLen + hPacked·wPacked, headDim]</c>.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildJointTables(int txtSeqLen, int hPacked, int wPacked)
    {
        (int, int, int) key = (txtSeqLen, hPacked, wPacked);
        if (_jointCos is not null && _jointSin is not null && _jointKey == key)
            return (_jointCos, _jointSin);
        _jointCos?.Dispose();
        _jointSin?.Dispose();
        (float[] cos, float[] sin) = BuildJointTable(txtSeqLen, hPacked, wPacked);
        (_jointCos, _jointSin) = ExpandToDeviceTables(cos, sin);
        _jointKey = key;
        return (_jointCos, _jointSin);
    }

    /// <summary>Frees the cached device rope tables.</summary>
    public void Dispose()
    {
        _textCos?.Dispose();
        _textSin?.Dispose();
        _imageCos?.Dispose();
        _imageSin?.Dispose();
        _jointCos?.Dispose();
        _jointSin?.Dispose();
        _textCos = _textSin = _imageCos = _imageSin = _jointCos = _jointSin = null;
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
