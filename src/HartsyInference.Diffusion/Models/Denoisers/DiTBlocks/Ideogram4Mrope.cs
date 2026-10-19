using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis sectioned MRoPE for Ideogram 4, ported verbatim from <c>Ideogram4MRoPE</c> in upstream <c>modeling_ideogram4.py</c>.
///
/// Distinct from <see cref="FluxRope"/> (interleaved-pair) and <see cref="ErnieImageRope"/> (adjacent-duplicate <c>[θ0,θ0,θ1,θ1,…]</c>). Ideogram builds a half-length frequency vector, selectively overwrites entries with the height/width axes per <c>mrope_section</c>, then forms <c>emb = cat(freqs, freqs)</c> (BLOCK repeat: <c>[f0..f127, f0..f127]</c>) and applies the standard HF <c>rotate_half</c> rotation <c>x·cos + rotate_half(x)·sin</c> with <c>rotate_half(x) = cat(-x[half:], x[:half])</c>.
///
/// <para><b>Section assignment.</b> With <c>head_dim/2 = 128</c> frequency slots and <c>mrope_section = (T, H, W)</c>: slot <c>k</c> defaults to the temporal axis; slots <c>1, 4, …, 1+3·(H−1)</c> take the height axis; slots <c>2, 5, …, 2+3·(W−1)</c> take the width axis. (The temporal section count is implicit — temporal keeps every slot the H/W loops don't claim.)</para></summary>
public sealed unsafe class Ideogram4Mrope
{
    private const int NumAxes = 3;

    private readonly int _headDim;
    private readonly int _halfDim;
    private readonly RopeFrequencyPlan _plan;

    /// <summary>Creates the MRoPE for the given head dim, base θ, and per-axis sections <c>(temporal, height, width)</c>.</summary>
    public Ideogram4Mrope(int headDim, int theta, (int Temporal, int Height, int Width) section)
    {
        if (headDim % 2 != 0)
            throw new ArgumentException($"head_dim {headDim} must be even.", nameof(headDim));
        _headDim = headDim;
        _halfDim = headDim / 2;
        _plan = RopeFrequencyPlan.CreateSectioned(theta, PairAxes(_halfDim, section.Height, section.Width), NumAxes);
    }

    /// <summary>Total per-head dimension covered by the rotation.</summary>
    public int HeadDim => _headDim;

    /// <summary>The axis each of the <paramref name="half"/> frequency slots reads: temporal by default, height for slots
    /// <c>1, 4, …, 1 + 3·(H−1)</c> and width for <c>2, 5, …, 2 + 3·(W−1)</c>, each run stopping at the last slot.</summary>
    private static int[] PairAxes(int half, int sectionH, int sectionW)
    {
        int[] axes = new int[half];
        for (int j = 0; j < sectionH; j++)
        {
            int k = 1 + 3 * j;
            if (k >= half) break;
            axes[k] = 1;
        }
        for (int j = 0; j < sectionW; j++)
        {
            int k = 2 + 3 * j;
            if (k >= half) break;
            axes[k] = 2;
        }
        return axes;
    }

    /// <summary>Builds the cos/sin tensors for one denoising step. <paramref name="positionIds"/> is <c>[B, L, 3]</c> F32 holding <c>(temporal, height, width)</c> positions. Outputs are <c>[B, L, head_dim]</c> each.</summary>
    public (Tensor Cos, Tensor Sin) BuildCosSin(Tensor positionIds)
    {
        int batch = (int)positionIds.Shape[0];
        int seqLen = (int)positionIds.Shape[1];
        if ((int)positionIds.Shape[2] != NumAxes)
            throw new ArgumentException($"positionIds last dim must be 3, got {positionIds.Shape[2]}.", nameof(positionIds));

        // emb = cat(freqs, freqs): each slot's angle fills both halves, the rotate-half row layout.
        TensorShape shape = new TensorShape(batch, seqLen, _headDim);
        Tensor cos = new Tensor(shape, DType.F32);
        Tensor? sin = null;
        bool built = false;
        try
        {
            sin = new Tensor(shape, DType.F32);
            int tokens = checked(batch * seqLen);
            int count = checked(tokens * _headDim);
            MultiAxisRopeTables.BuildRows(_plan, new ReadOnlySpan<float>(positionIds.DataPointer, checked(tokens * NumAxes)), tokens,
                RopeAngleOrder.MultiplyByInverse, RopeTableLayout.RotateHalf, 0, RopeTablePadding.Identity,
                new Span<float>(cos.DataPointer, count), new Span<float>(sin.DataPointer, count));
            built = true;
            return (cos, sin);
        }
        finally
        {
            if (!built)
            {
                cos.Dispose();
                sin?.Dispose();
            }
        }
    }

    /// <summary>Applies the rotation in-place to Q and K, both <c>[B, L, numHeads, head_dim]</c>. cos/sin are <c>[B, L, head_dim]</c> (broadcast over heads). <c>out[i] = x[i]·cos[i] + rotate_half(x)[i]·sin[i]</c> with <c>rotate_half(x) = cat(-x[half:], x[:half])</c>.</summary>
    public void ApplyRotary(Tensor q, Tensor k, Tensor cos, Tensor sin)
    {
        int batch = (int)q.Shape[0];
        int seqLen = (int)q.Shape[1];
        int numHeads = (int)q.Shape[2];
        int headDim = (int)q.Shape[3];
        if (headDim != _headDim)
            throw new ArgumentException($"Q head_dim {headDim} != rope head_dim {_headDim}.", nameof(q));

        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        int half = _halfDim;

        Span<float> snapshot = stackalloc float[half];

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                long freqBase = ((long)b * seqLen + s) * _headDim;
                float* c = cosPtr + freqBase;
                float* sn = sinPtr + freqBase;
                for (int h = 0; h < numHeads; h++)
                {
                    long vecOff = (((long)b * seqLen + s) * numHeads + h) * _headDim;
                    ApplyOne(qPtr + vecOff, c, sn, half, snapshot);
                    ApplyOne(kPtr + vecOff, c, sn, half, snapshot);
                }
            }
        }
    }

    private static void ApplyOne(float* vec, float* cos, float* sin, int half, Span<float> snapshot)
    {
        // Snapshot the lower half — the upper-half write reads original lower-half values.
        for (int i = 0; i < half; i++) snapshot[i] = vec[i];
        for (int i = 0; i < half; i++)
        {
            float lower = snapshot[i];
            float upper = vec[i + half];
            // rotate_half: lower position pairs with -upper; upper position pairs with +lower.
            vec[i] = lower * cos[i] - upper * sin[i];
            vec[i + half] = upper * cos[i + half] + lower * sin[i + half];
        }
    }
}
