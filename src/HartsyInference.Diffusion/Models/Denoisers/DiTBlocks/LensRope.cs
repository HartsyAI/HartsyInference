using System.Runtime.CompilerServices;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis complex-polar rotary positional embedding for Microsoft Lens (<c>LensEmbedRope</c>). Splits the head dimension into <c>(frame, height, width)</c> sub-bands (default <c>[8, 28, 28]</c> summing to 64 = head_dim) and rotates each pair of features by <c>θ = position * (1 / base^(2k/axisDim))</c>. <b>Differs from QwenImageRope in two ways:</b> (1) <c>scale_rope=True</c> hardcoded — height and width positions are split into negative+positive halves centered around 0 (top/left rows get <c>arange(4096).flip(0) * -1 - 1</c>, bottom/right rows get <c>arange(4096)</c>), centering the rotation around the image's middle pixel rather than the top-left; (2) text RoPE positions start at <c>max(h//2, w//2)</c> (not <c>max(h, w)</c>) because of the height/width halving. Apply order matches QwenImageRope's complex polar pairs: features at indices <c>(2i, 2i+1)</c> form one complex number that gets multiplied by <c>e^(iθ)</c>.</summary>
/// <remarks>The operation is complex multiplication, which is the same as the interleaved 2x2 rotation the apply ops run. Frequencies
/// come from <see cref="RopeFrequencyPlan"/>, rows from <see cref="MultiAxisRopeTables"/>, and position rules live in
/// <see cref="LensPosEmbed"/>. The single joint device table is the staged row set of a <see cref="RopeTableCache"/> slot: host
/// tensors the backend uploads on first use, since the builder carries no backend. A rebuild installs the new table before it
/// frees the old one.</remarks>
public sealed unsafe class LensRope : IDisposable
{
    private static readonly RopeTableSettings StagedTableSettings = new()
    {
        AngleOrder = RopeAngleOrder.MultiplyByInverse,
        Layout = RopeTableLayout.InterleavedPair,
        RetainedSignatures = 1,
    };

    private readonly int[] _axesDim;
    private readonly int _headDim;
    private readonly RopeFrequencyPlan _plan;

    // Cached joint cos/sin tables in the [S, headDim] layout IBackend.WanRopeInterleaved reads (the angle for
    // pair i duplicated at indices 2i and 2i+1). Position-only, so one host build serves every block of every
    // step — replacing the per-block host ApplyRotationBatched, which was a D2H+H2D round trip of Q AND K per
    // stream per block (the dominant Lens step cost). Keyed on the full sequence layout; a resolution or
    // prompt-length change rebuilds. Joint order is [img, txt] (Lens's concat order — opposite of Qwen-Image).
    private readonly RopeTableCache _tables;
    private readonly RopeHeldTables<(int ImgH, int ImgW, int Txt, int TxtStart)> _jointTables;

    /// <summary>Creates a Lens RoPE with the given axis split. Lens always uses <c>scale_rope=True</c> — height/width positions are split into negative+positive halves around 0. Frame axis stays at <c>[0]</c> for image-only inference (frame=1).</summary>
    public LensRope(int[]? axesDim = null, int theta = 10000)
    {
        _axesDim = axesDim ?? [8, 28, 28];
        if (_axesDim.Length != 3)
            throw new ArgumentException("LensRope requires exactly 3 axes (frame, height, width).", nameof(axesDim));
        _plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = _axesDim, Theta = theta });
        _headDim = _plan.HeadDim;
        _tables = new RopeTableCache(_plan, StagedTableSettings);
        _jointTables = new RopeHeldTables<(int, int, int, int)>(_tables, slot: 0, axes: LensPosEmbed.NumAxes);
    }

    /// <summary>Sum of per-axis dims; equals the per-head dim of the wrapped attention block (64 for Lens default).</summary>
    public int HeadDim => _headDim;

    /// <summary>Per-axis dim split: <c>(frame, height, width)</c>.</summary>
    public ReadOnlySpan<int> AxesDim => _axesDim;

    /// <summary>Rotates Q and K in-place for image tokens in <c>scale_rope=True</c> mode. With <paramref name="hPacked"/>=H, the top <c>H - H/2</c> rows use positions <c>[-H+H/2, ..., -1]</c> and the bottom <c>H/2</c> rows use positions <c>[0, ..., H/2 - 1]</c> (Microsoft's centered convention). Same split applies to width. Token <c>r*W + c</c> uses position <c>[0, row_pos, col_pos]</c>. Both Q and K must be <c>[B, numHeads, imgSeqLen, headDim]</c>.</summary>
    public void ApplyImage(Tensor q, Tensor k, int batch, int numHeads, int hPacked, int wPacked)
    {
        ApplyPositions(q, k, batch, numHeads, LensPosEmbed.ImageRows(hPacked, wPacked));
    }

    /// <summary>Rotates Q and K in-place for text tokens. With <c>scale_rope=True</c> text positions start at <c>max(hPacked/2, wPacked/2)</c> to live past the image's positive-half range. All three axes (frame, h, w) get the same scalar position per text token (upstream uses <c>pos_freqs</c> directly on the text region with no axis split).</summary>
    public void ApplyText(Tensor q, Tensor k, int batch, int numHeads, int txtSeqLen, int positionStart)
    {
        ApplyPositions(q, k, batch, numHeads, LensPosEmbed.TextRows(txtSeqLen, positionStart));
    }

    /// <summary>Computes the position offset for text tokens in <c>scale_rope=True</c> mode. Matches upstream's <c>max_vid_index = max(height // 2, width // 2)</c>.</summary>
    public static int ComputeTextPositionStart(int hPacked, int wPacked) => LensPosEmbed.TextPositionStart(hPacked, wPacked);

    /// <summary>Rotates Q and K in-place for the JOINT <c>[img, txt]</c> sequence in one host pass — bit-identical
    /// to calling <see cref="ApplyImage"/> then <see cref="ApplyText"/> on the separate streams before concatenation
    /// (RoPE is per-row independent). Batch&gt;1 fallback for the device rope path; both Q and K must be
    /// <c>[B, numHeads, imgSeqLen + txtSeqLen, headDim]</c>.</summary>
    public void ApplyJoint(Tensor q, Tensor k, int batch, int numHeads, int hPacked, int wPacked,
        int txtSeqLen, int txtPositionStart)
    {
        ApplyPositions(q, k, batch, numHeads, LensPosEmbed.JointRows(hPacked, wPacked, txtSeqLen, txtPositionStart));
    }

    /// <summary>Builds (or returns the cached) joint <c>[img, txt]</c> cos/sin tables for the device rope path,
    /// <c>[S, headDim]</c> F32 in the <see cref="Core.Backends.IBackend.WanRopeInterleaved"/> convention (pair i's
    /// angle duplicated at indices 2i and 2i+1). Same position math as <see cref="ApplyImage"/> +
    /// <see cref="ApplyText"/> — RoPE is per-row independent, so rotating the GPU-concatenated joint sequence once
    /// is bit-identical to the two per-stream host passes it replaces. Mirrors QwenImageRope.GetOrBuildJointTables.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildJointTables(int hPacked, int wPacked, int txtSeqLen, int txtPositionStart) =>
        _jointTables.GetOrBuildStaged((hPacked, wPacked, txtSeqLen, txtPositionStart),
            static key => LensPosEmbed.JointRows(key.ImgH, key.ImgW, key.Txt, key.TxtStart));

    /// <summary>Frees the cached joint tables.</summary>
    public void Dispose()
    {
        _jointTables.Dispose();
        _tables.Dispose();
    }

    /// <summary>Computes the compact pair tables for flat <c>(frame, height, width)</c> rows and rotates Q and K with them.</summary>
    private void ApplyPositions(Tensor q, Tensor k, int batch, int numHeads, float[] positions)
    {
        int tokens = positions.Length / LensPosEmbed.NumAxes;
        float[] cos = new float[tokens * _plan.Pairs];
        float[] sin = new float[cos.Length];
        MultiAxisRopeTables.ComputePairs(_plan, positions, tokens, RopeAngleOrder.MultiplyByInverse, cos, sin);
        ApplyRotationBatched(q, k, cos, sin, batch, numHeads, tokens);
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
