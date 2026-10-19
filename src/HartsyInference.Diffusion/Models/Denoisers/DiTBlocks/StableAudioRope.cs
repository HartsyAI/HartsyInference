using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Stable Audio's partial split-half (NeoX) RoPE — verified against <c>stable_audio_tools</c>'s
/// <c>RotaryEmbedding</c>/<c>apply_rotary_pos_emb</c>: only the leading <c>rotDim</c> channels of each
/// <c>headDim</c>-sized head rotate (GPT-J-style partial rotary, <c>rotDim &lt; headDim</c>); the remaining
/// channels pass through unchanged. Within the rotated span, pairing is split-half (index <c>d</c> with
/// <c>d + rotDim/2</c>), NOT interleaved adjacent pairs. Self-attention only — Stable Audio's cross-attention
/// never receives <c>rotary_pos_emb</c>.
///
/// <para>Applied via <see cref="Core.Backends.IBackend.ApplyRopeSingle"/> (GPU-resident on CUDA) rather than a
/// host loop — that op expects the pre-multi-head-permute <c>[B, L, H, D]</c> layout and a cos/sin table
/// <b>duplicated</b> across the rotated half (<c>table[pos, i] == table[pos, i + half]</c> for
/// <c>i in [0, half)</c>), which is what <see cref="BuildCosTable"/> below produces. The rows come from
/// <see cref="RopeTableExpander"/> in <see cref="RopeTableLayout.RotateHalf"/> with zero padding, so the trailing channels
/// are zero, as before.</para></summary>
public static unsafe class StableAudioRope
{
    /// <summary>Builds <c>[1, seqLen, headDim]</c> cos/sin tables for <see cref="Core.Backends.IBackend.ApplyRopeSingle"/>
    /// from the checkpoint's <c>inv_freq</c> (loaded, not recomputed from theta — Stable Audio ships it verbatim as
    /// <c>transformer.rotary_pos_emb.inv_freq</c>). Only the leading <c>2 * invFreq.Length</c> channels are filled
    /// (duplicated across the two halves, per <c>ApplyRopeSingle</c>'s expected layout); the trailing
    /// <c>[rotDim, headDim)</c> channels are left zero and are never read (partial rotary passthrough). Position 0
    /// is the prepended global token (rotation is identity there since angle = 0·inv_freq = 0).</summary>
    public static Tensor BuildCosTable(ReadOnlySpan<float> invFreq, int seqLen, int headDim) =>
        BuildTable(invFreq, seqLen, headDim, sine: false);

    /// <summary>The sine counterpart of <see cref="BuildCosTable"/>.</summary>
    public static Tensor BuildSinTable(ReadOnlySpan<float> invFreq, int seqLen, int headDim) =>
        BuildTable(invFreq, seqLen, headDim, sine: true);

    private static Tensor BuildTable(ReadOnlySpan<float> invFreq, int seqLen, int headDim, bool sine)
    {
        int half = invFreq.Length;
        if (2 * half > headDim)
            throw new ArgumentOutOfRangeException(nameof(invFreq), half, "Stable Audio RoPE rotates 2 × invFreq.Length channels, which must fit in headDim.");
        Tensor table = new(new TensorShape(1, seqLen, headDim), DType.F32);
        if (half == 0)
            return table;   // No rotated channels. RopeTableExpander treats rotaryDim 0 as full width, so skip it.

        float[] frequencies = invFreq.ToArray();
        float[] positions = new float[seqLen];
        for (int pos = 0; pos < seqLen; pos++)
            positions[pos] = pos;
        RopeHostTableWriter writer = (p, tokens, cos, sin) => WriteHostPairs(frequencies, p, tokens, cos, sin);
        float[] cosPairs = new float[checked(seqLen * half)];
        float[] sinPairs = new float[cosPairs.Length];
        writer(positions, seqLen, cosPairs, sinPairs);

        try
        {
            int count = checked(seqLen * headDim);
            float[] scratch = new float[count];
            Span<float> tableRows = new((void*)table.DataPointer, count);
            Span<float> scratchRows = scratch;
            RopeTableExpander.Expand(cosPairs, sinPairs, seqLen, headDim, RopeTableLayout.RotateHalf, 2 * half, RopeTablePadding.Zero,
                sine ? scratchRows : tableRows, sine ? tableRows : scratchRows);
            return table;
        }
        catch
        {
            table.Dispose();
            throw;
        }
    }

    /// <summary>Checkpoint frequencies in F32: <c>angle = position × invFreq[i]</c>, then <c>cos</c> and <c>sin</c> in F32.
    /// Positions are the sequence index, exact as floats.</summary>
    private static void WriteHostPairs(float[] invFreq, ReadOnlySpan<float> positions, int tokens, Span<float> cos, Span<float> sin)
    {
        int half = invFreq.Length;
        for (int t = 0; t < tokens; t++)
        {
            int row = t * half;
            for (int i = 0; i < half; i++)
            {
                float angle = positions[t] * invFreq[i];
                cos[row + i] = MathF.Cos(angle);
                sin[row + i] = MathF.Sin(angle);
            }
        }
    }
}
