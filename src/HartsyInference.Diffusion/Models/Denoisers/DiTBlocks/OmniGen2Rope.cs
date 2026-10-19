using System.Runtime.CompilerServices;
using HartsyInference.Core.Rope;
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
/// <remarks>Frequencies come from <see cref="RopeFrequencyPlan"/> and <see cref="MultiAxisRopeTables"/>. Position rules live
/// in <see cref="OmniGen2PosEmbed"/>. The device-layout tables are staged rows of a <see cref="RopeTableCache"/> (one slot per
/// role, one signature each, as before): host tensors the backend uploads on first use, since the builders carry no
/// backend.</remarks>
public sealed unsafe class OmniGen2Rope : IDisposable
{
    private const int TextSlot = 0, ImageSlot = 1, JointSlot = 2;

    private static readonly RopeTableSettings StagedTableSettings = new()
    {
        AngleOrder = RopeAngleOrder.MultiplyByInverse,
        Layout = RopeTableLayout.InterleavedPair,
        RetainedSignatures = 1,
    };

    private readonly int[] _axesDim;
    private readonly int _headDim;
    private readonly RopeFrequencyPlan _plan;

    // Device rope tables (WanRopeInterleaved [S, headDim] layout). Positions are step-invariant and size-invariant, so a
    // table built once is reused across every block and every denoise step of a generation (and across generations at the
    // same geometry) — this is what turns the per-block host Q/K drain into a single device kernel. One slot per stream
    // type used concurrently in a forward (text / image / joint).
    private readonly RopeTableCache _tables;
    private readonly RopeHeldTables<int> _textTables;
    private readonly RopeHeldTables<(int H, int W, int T)> _imageTables;
    private readonly RopeHeldTables<(int Txt, int H, int W)> _jointTables;

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
        _plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = _axesDim, Theta = theta });
        _headDim = _plan.HeadDim;
        _tables = new RopeTableCache(_plan, StagedTableSettings);
        _textTables = new RopeHeldTables<int>(_tables, TextSlot, OmniGen2PosEmbed.NumAxes);
        _imageTables = new RopeHeldTables<(int, int, int)>(_tables, ImageSlot, OmniGen2PosEmbed.NumAxes);
        _jointTables = new RopeHeldTables<(int, int, int)>(_tables, JointSlot, OmniGen2PosEmbed.NumAxes);
    }

    /// <summary>Sum of the per-axis dimensions; equals the per-head dim of the wrapped attention block.</summary>
    public int HeadDim => _headDim;

    /// <summary>Builds an interleaved <c>(cos, sin)</c> table sized <c>[seqLen * headDim/2]</c> for a text-only
    /// sequence: token <c>s</c> uses position <c>(s, s, s)</c>. Used by <c>context_refiner</c> blocks.</summary>
    public (float[] cos, float[] sin) BuildTextTable(int txtSeqLen) => Compact(OmniGen2PosEmbed.TextRows(txtSeqLen));

    /// <summary>Builds an interleaved <c>(cos, sin)</c> table for an image-only sequence: token at packed grid
    /// <c>(r, c)</c> uses position <c>(timeOffset, r, c)</c>. Used by <c>noise_refiner</c> blocks.</summary>
    public (float[] cos, float[] sin) BuildImageTable(int hPacked, int wPacked, int timeOffset) =>
        Compact(OmniGen2PosEmbed.ImageRows(hPacked, wPacked, timeOffset));

    /// <summary>Builds the joint-sequence table for the main <c>layers</c> stack: the first
    /// <paramref name="txtSeqLen"/> tokens use text positions <c>(s, s, s)</c>; the remaining
    /// <c>hPacked * wPacked</c> tokens use image positions <c>(txtSeqLen, row, col)</c>. The returned tables are
    /// sized for the full joint sequence so a single <see cref="Apply"/> pass rotates Q/K for the entire merged
    /// stream.</summary>
    public (float[] cos, float[] sin) BuildJointTable(int txtSeqLen, int hPacked, int wPacked) =>
        Compact(OmniGen2PosEmbed.JointRows(txtSeqLen, hPacked, wPacked));

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
        if (heightIds.Length != timeIds.Length || widthIds.Length != timeIds.Length)
            throw new ArgumentException("OmniGen2Rope.BuildTableFromPositions requires equal-length id spans.");
        return Compact(OmniGen2PosEmbed.FromIds(timeIds, heightIds, widthIds));
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
        int count = checked(seqLen * _headDim);
        Tensor cos = new Tensor(new TensorShape(seqLen, _headDim), DType.F32);
        Tensor? sin = null;
        bool succeeded = false;
        try
        {
            sin = new Tensor(new TensorShape(seqLen, _headDim), DType.F32);
            RopeTableExpander.Expand(cosTable, sinTable, seqLen, _headDim, RopeTableLayout.InterleavedPair, 0,
                RopeTablePadding.Identity, new Span<float>(cos.DataPointer, count), new Span<float>(sin.DataPointer, count));
            succeeded = true;
            return (cos, sin);
        }
        finally
        {
            if (!succeeded)
            {
                cos.Dispose();
                sin?.Dispose();
            }
        }
    }

    /// <summary>Returns the cached device <c>(cos, sin)</c> tables for a text-only stream (positions <c>(s,s,s)</c>),
    /// building them on first use. <c>[txtSeqLen, headDim]</c> in the <see cref="Core.Backends.IBackend.WanRopeInterleaved"/>
    /// convention — bit-equivalent to the host <see cref="ApplyText"/> path minus the per-block Q/K D2H drain.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildTextTables(int txtSeqLen) =>
        _textTables.GetOrBuildStaged(txtSeqLen, () => OmniGen2PosEmbed.TextRows(txtSeqLen));

    /// <summary>Returns the cached device <c>(cos, sin)</c> tables for an image-only stream (positions
    /// <c>(timeOffset, row, col)</c>), building them on first use. <c>[hPacked·wPacked, headDim]</c>.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildImageTables(int hPacked, int wPacked, int timeOffset) =>
        _imageTables.GetOrBuildStaged((hPacked, wPacked, timeOffset), () => OmniGen2PosEmbed.ImageRows(hPacked, wPacked, timeOffset));

    /// <summary>Returns the cached device <c>(cos, sin)</c> tables for the joint <c>[text, image]</c> stream,
    /// building them on first use. <c>[txtSeqLen + hPacked·wPacked, headDim]</c>.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildJointTables(int txtSeqLen, int hPacked, int wPacked) =>
        _jointTables.GetOrBuildStaged((txtSeqLen, hPacked, wPacked), () => OmniGen2PosEmbed.JointRows(txtSeqLen, hPacked, wPacked));

    /// <summary>Frees the cached device rope tables.</summary>
    public void Dispose()
    {
        _textTables.Dispose();
        _imageTables.Dispose();
        _jointTables.Dispose();
        _tables.Dispose();
    }

    /// <summary>Computes compact pair tables for flat <c>(time, height, width)</c> rows.</summary>
    private (float[] Cos, float[] Sin) Compact(float[] positions)
    {
        int tokens = positions.Length / OmniGen2PosEmbed.NumAxes;
        float[] cos = new float[checked(tokens * _plan.Pairs)];
        float[] sin = new float[cos.Length];
        MultiAxisRopeTables.ComputePairs(_plan, positions, tokens, RopeAngleOrder.MultiplyByInverse, cos, sin);
        return (cos, sin);
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
