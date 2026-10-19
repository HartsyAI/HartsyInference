using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using JointKey = (int ImgH, int ImgW, int Txt, int TxtStart, long RefSig, int TokenStart, int TokenCount);

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis rotary positional embedding for Qwen-Image (<c>QwenEmbedRope</c>). Splits the head dimension into <c>(frame, height, width)</c> sub-bands (default <c>[16, 56, 56]</c>) and rotates each pair of features by <c>theta = position * (1 / base^(2k/axisDim))</c>. Image tokens use CENTERED positions <c>[0, h-(H-H//2), w-(W-W//2)]</c> for a single-frame layout (<c>scale_rope=True</c>, which diffusers hardcodes). Text tokens use linearly-increasing positions starting at <c>max(H//2, W//2)</c>, mirroring diffusers' <c>txt_freqs = pos_freqs[max_vid_index : max_vid_index + max_txt_seq_len]</c>. The two streams are rotated separately before joint-attention concatenation. Apply order matches Flux's complex polar interpretation: pairs are <c>(real, imag)</c> at indices <c>(2i, 2i+1)</c>.</summary>
/// <remarks>Frequencies and tables come from <see cref="RopeFrequencyPlan"/> and <see cref="MultiAxisRopeTables"/>. Position
/// rules live in <see cref="QwenImagePosEmbed"/>. Device tables go through a <see cref="RopeTableCache"/>, one lease per
/// backend, so each DiT shard keeps its own tables.</remarks>
public sealed unsafe class QwenImageRope : IDisposable
{
    private static readonly RopeTableSettings DeviceTableSettings = new()
    {
        AngleOrder = RopeAngleOrder.MultiplyByInverse,
        Layout = RopeTableLayout.InterleavedPair,
        Preload = true,
        RetainedSignatures = 2,
    };

    private readonly int[] _axesDim;
    private readonly int _headDim;
    // false → text tokens get NO positional embedding (identity rotation): Mage-Flow's apply_text_rotary_emb=false.
    // Achieved by putting text at position 0 on all axes (angle 0 → cos 1 / sin 0). Qwen-Image ropes text (=true).
    private readonly bool _ropeText;
    private readonly RopeFrequencyPlan _plan;
    private readonly RopeTableCache _deviceTables;

    // Joint device tables are keyed by the full sequence layout, one held lease per backend. DiT sharding runs blocks on a
    // second backend, and context-parallel ranks build different row sub-ranges concurrently; each backend keeps its own
    // tables, so one rank's build never frees tables another rank still reads.
    private readonly RopeHeldTables<JointKey> _jointTables;

    public QwenImageRope(int[]? axesDim = null, int theta = 10000, bool ropeText = true)
    {
        _axesDim = axesDim ?? [16, 56, 56];
        if (_axesDim.Length != 3)
            throw new ArgumentException("QwenImageRope requires exactly 3 axes (frame, height, width).", nameof(axesDim));
        _ropeText = ropeText;
        _plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = _axesDim, Theta = theta });
        _headDim = _plan.HeadDim;
        _deviceTables = new RopeTableCache(_plan, DeviceTableSettings);
        _jointTables = new RopeHeldTables<JointKey>(_deviceTables, slot: 0, axes: QwenImagePosEmbed.NumAxes);
    }

    /// <summary>Sum of the per-axis dimensions; equals the per-head dim of the wrapped attention block.</summary>
    public int HeadDim => _headDim;

    /// <summary>Per-axis dim split: <c>(frame, height, width)</c>. Each must be even — pairs of features get rotated together.</summary>
    public ReadOnlySpan<int> AxesDim => _axesDim;

    /// <summary>Rotates Q and K in-place for image tokens. Tokens are laid out row-major over <paramref name="hPacked"/> × <paramref name="wPacked"/>; token <c>r * W + c</c> uses position <c>[0, r, c]</c>. Both Q and K must be <c>[B, numHeads, imgSeqLen, headDim]</c> and <c>imgSeqLen == hPacked * wPacked</c>.</summary>
    public void ApplyImage(Tensor q, Tensor k, int batch, int numHeads, int hPacked, int wPacked)
    {
        // scale_rope=True (diffusers QwenImageTransformer2DModel hardcodes it): spatial positions are CENTERED.
        ApplyPositions(q, k, batch, numHeads, QwenImagePosEmbed.ImageRows(hPacked, wPacked));
    }

    /// <summary>Rotates Q and K in-place for text tokens. Each text position <paramref name="positionStart"/> + s uses <c>[positionStart + s, positionStart + s, positionStart + s]</c> (diffusers <c>pos_freqs</c> is the same scalar across the three axes for the text region). Both Q and K must be <c>[B, numHeads, txtSeqLen, headDim]</c>.</summary>
    public void ApplyText(Tensor q, Tensor k, int batch, int numHeads, int txtSeqLen, int positionStart)
    {
        ApplyPositions(q, k, batch, numHeads, QwenImagePosEmbed.TextRows(txtSeqLen, positionStart));
    }

    /// <summary>Rotates Q and K in-place for the JOINT <c>[txt, img]</c> sequence in one pass — text occupies rows
    /// <c>0..txtSeqLen</c> (positions <c>positionStart + s</c> on all three axes), image occupies rows
    /// <c>txtSeqLen..txtSeqLen+H·W</c> (centered <c>[0, row-Hc, col-Wc]</c>). Bit-identical to calling
    /// <see cref="ApplyText"/> then <see cref="ApplyImage"/> on the separate streams before concatenation, since RoPE
    /// is per-row independent. Used by the GPU-resident block path, which concatenates <c>[txt, img]</c> on the GPU
    /// first (contiguous row-concat) and then ropes the joint tensor once — avoiding a per-stream CPU round-trip.
    /// Both Q and K must be <c>[B, numHeads, txtSeqLen + H·W (+ refH·refW), headDim]</c>.
    /// <para>Qwen-Image-Edit reference-latent tokens: when <paramref name="refGrids"/> is non-empty, an extra
    /// <c>refH·refW</c> token section follows the main image rows. Ref tokens use frame axis <c>index + 1</c>
    /// (ComfyUI <c>qwen_image/model.py</c> ref_latents "index" method: first ref = 1) and spatial positions centered on the
    /// REF grid — the reference image may have a different resolution than the output.</para></summary>
    public void ApplyJoint(Tensor q, Tensor k, int batch, int numHeads, int imgPackedH, int imgPackedW, int txtSeqLen,
        int txtPositionStart, ReadOnlySpan<(int H, int W)> refGrids = default)
    {
        ApplyPositions(q, k, batch, numHeads,
            QwenImagePosEmbed.JointRows(imgPackedH, imgPackedW, txtSeqLen, txtPositionStart, _ropeText, refGrids,
                imgTokenStart: 0, imgTokenCount: -1));
    }

    /// <summary>Builds (or returns the cached) joint cos/sin tables for the device rope path, <c>[S, headDim]</c>
    /// F32 in the <see cref="IBackend.WanRopeInterleaved"/> convention (pair i's angle at index
    /// <c>2i</c>). Same position math as <see cref="ApplyJoint"/> — the rotation itself matches the interleaved
    /// kernel exactly (<c>x0·cos−x1·sin, x0·sin+x1·cos</c> on pairs (2i, 2i+1)).
    /// <para><paramref name="imgTokenStart"/>/<paramref name="imgTokenCount"/> restrict the image section to a
    /// context-parallel rank's global token sub-range (GLOBAL grid positions, fewer rows); incompatible with
    /// reference grids — the pipeline gates edit conditioning off under CP.</para>
    /// <para>The tables stay valid until this backend builds a different layout, or until <see cref="Dispose"/>.</para></summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildJointTables(IBackend backend, int imgPackedH, int imgPackedW, int txtSeqLen,
        int txtPositionStart, ReadOnlySpan<(int H, int W)> refGrids = default, int imgTokenStart = 0, int imgTokenCount = -1)
    {
        long refSig = 0x51F0;
        int refSeqLen = 0;
        foreach ((int rh, int rw) in refGrids)
        {
            refSig = refSig * 1000003L ^ ((long)rh << 20 | (uint)rw);
            refSeqLen += rh * rw;
        }
        if (imgTokenCount >= 0 && refSeqLen > 0)
            throw new ArgumentException("A rank-restricted image range cannot be combined with reference grids.", nameof(imgTokenCount));
        JointKey key = (imgPackedH, imgPackedW, txtSeqLen, txtPositionStart, refSig, imgTokenStart, imgTokenCount);
        if (_jointTables.TryGet(backend, key, out (Tensor Cos, Tensor Sin) held))
            return held;

        // A lambda cannot capture the span, so a miss copies the grids; ToArray on an empty span allocates nothing.
        return BuildJointTables(backend, key, refGrids.ToArray());
    }

    /// <summary>The miss path of <see cref="GetOrBuildJointTables"/>, kept apart so a hit allocates no closure.</summary>
    private (Tensor Cos, Tensor Sin) BuildJointTables(IBackend backend, JointKey key, (int H, int W)[] refs) =>
        _jointTables.GetOrBuild(backend, key, k => QwenImagePosEmbed.JointRows(k.ImgH, k.ImgW, k.Txt, k.TxtStart, _ropeText, refs,
            k.TokenStart, k.TokenCount));

    /// <summary>Computes the position offset to use when calling <see cref="ApplyText"/>. Matches diffusers'
    /// <c>scale_rope=True</c> mode where text starts at <c>max_vid_index = max(height//2, width//2)</c>
    /// (the centered image grid's max index), used as the single scalar across all three axes for text.</summary>
    public static int ComputeTextPositionStart(int hPacked, int wPacked) => QwenImagePosEmbed.TextPositionStart(hPacked, wPacked);

    /// <summary>Releases the held joint tables and the cache. Tables still under a lease are freed by their last release.</summary>
    public void Dispose()
    {
        _jointTables.Dispose();
        _deviceTables.Dispose();
    }

    /// <summary>Computes the compact per-pair host tables for <paramref name="positions"/> and rotates Q and K with them.</summary>
    private void ApplyPositions(Tensor q, Tensor k, int batch, int numHeads, float[] positions)
    {
        int tokens = positions.Length / QwenImagePosEmbed.NumAxes;
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
