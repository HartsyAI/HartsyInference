using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan-Video 3D rotary position embedding (<c>WanRotaryPosEmbed</c> + the processor's <c>apply_rotary_emb</c> in diffusers <c>transformer_wan.py</c>). Per-head (head_dim) interleaved-pair RoPE: the head dim splits into <c>t_dim / h_dim / w_dim</c> (<c>h=w=2·(head_dim//6)</c>, <c>t=head_dim−h−w</c>), each axis getting standard <c>get_1d_rotary_pos_embed</c> frequencies (<c>θ^(−2i/dim)</c>, repeat-interleaved). Distinct from <see cref="LtxRope"/> (full-dim) and <see cref="Multimodal3DRope"/> (block form).</summary>
/// <remarks>Frequencies come from a <see cref="RopeFrequencyPlan"/> per θ (one per head in sigma_theta mode) and the duplicated-pair
/// rows from <see cref="MultiAxisRopeTables"/>; this class owns only the <c>(frame, row, col)</c> coordinates.</remarks>
public sealed unsafe class WanRope
{
    private const int NumAxes = 3;

    private readonly int _headDim;
    private readonly RopeFrequencyPlan _plan;
    private readonly int _numHeads;                     // > 0 only in per-head (sigma_theta) mode
    private readonly RopeFrequencyPlan[]? _headPlans;   // one θ_h per head (sigma_theta mode)

    /// <param name="sigmaTheta">Matrix-Game 3.0 per-head θ spread (<c>config.json sigma_theta</c>). When non-zero (and
    /// <paramref name="numHeads"/> &gt; 0) each head <c>h</c> gets <c>θ_h = theta·(1 + sigmaTheta·ε_h)</c> with
    /// <c>ε_h = linspace(-1, 1, numHeads)</c>, and <see cref="BuildCosSin(ReadOnlySpan{int}, int, int)"/> emits per-head
    /// tables <c>[numHeads, S, headDim]</c>. Zero = a single shared θ (standard Wan behavior; byte-identical path).</param>
    public WanRope(int headDim, double theta, int maxSeqLen, double sigmaTheta = 0.0, int numHeads = 0)
    {
        int hDim = 2 * (headDim / 6);
        int tDim = headDim - 2 * hDim;
        _plan = Plan(tDim, hDim, hDim, theta);
        _headDim = _plan.HeadDim;
        _ = maxSeqLen;   // freqs are computed on demand; max length is informational
        if (sigmaTheta != 0.0 && numHeads > 0)
        {
            _numHeads = numHeads;
            _headPlans = new RopeFrequencyPlan[numHeads];
            for (int h = 0; h < numHeads; h++)
            {
                double eps = numHeads == 1 ? -1.0 : -1.0 + 2.0 * h / (numHeads - 1);   // linspace(-1, 1, numHeads)
                _headPlans[h] = Plan(tDim, hDim, hDim, theta * (1.0 + sigmaTheta * eps));
            }
        }
    }

    /// <summary>Explicit per-axis dim split (t, h, w) — Matrix-Game's ActionModule uses <c>[8, 28, 28]</c> at θ=256
    /// over a 64-wide head, instead of the main DiT's derived split.</summary>
    public WanRope(int tDim, int hDim, int wDim, double theta)
    {
        _plan = Plan(tDim, hDim, wDim, theta);
        _headDim = _plan.HeadDim;
    }

    /// <summary>Head dim this rope rotates.</summary>
    public int HeadDim => _headDim;

    private static RopeFrequencyPlan Plan(int tDim, int hDim, int wDim, double theta) =>
        RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [tDim, hDim, wDim], Theta = theta });

    /// <summary>Builds cos/sin <c>[S, headDim]</c> for a token grid <c>gridT × gridH × gridW</c> in <c>(f,h,w)</c> order. Each value is duplicated across its interleaved pair (so the pair shares one angle).</summary>
    /// <param name="offsetT">Per-axis position offsets added to every token's index. Wan-Animate-2's driving stream
    /// uses <c>(1, 0, genGridW)</c>: <c>+1</c> lines driving frame <c>j</c> up with generation frame <c>j+1</c> past
    /// the reference slot, and the horizontal shift parks the driving tokens in their own strip of RoPE space.</param>
    public (Tensor Cos, Tensor Sin) BuildCosSin(int gridT, int gridH, int gridW, int offsetT = 0, int offsetH = 0, int offsetW = 0)
    {
        int[] frames = new int[gridT];
        for (int i = 0; i < gridT; i++) frames[i] = i + offsetT;
        return BuildCosSin(frames, gridH, gridW, offsetH, offsetW);
    }

    /// <summary>Builds cos/sin <c>[S, headDim]</c> with an explicit temporal index per frame slot — Matrix-Game's
    /// memory-augmented sequence assigns retrieved memory slots their <b>original historical</b> t-indices
    /// (non-contiguous gaps are fine; the model is trained for them).</summary>
    public (Tensor Cos, Tensor Sin) BuildCosSin(ReadOnlySpan<int> frameIndices, int gridH, int gridW, int offsetH = 0, int offsetW = 0)
    {
        float[] positions = GridPositions(frameIndices, gridH, gridW, offsetH, offsetW);
        int seq = positions.Length / NumAxes;
        if (_headPlans is null)
        {
            return MultiAxisRopeTables.BuildTensors(_plan, positions, seq, RopeAngleOrder.MultiplyByInverse,
                RopeTableLayout.InterleavedPair, 0, RopeTablePadding.Identity);
        }

        // Per-head (sigma_theta) cos/sin [numHeads, S, headDim]: token order matches the shared path, but each head uses its
        // own θ_h frequencies. Consumed by ApplyRotary's rank-3 branch.
        Tensor cos = new Tensor(new TensorShape(_numHeads, seq, _headDim), DType.F32);
        Tensor? sin = null;
        bool built = false;
        try
        {
            sin = new Tensor(new TensorShape(_numHeads, seq, _headDim), DType.F32);
            int headLength = checked(seq * _headDim);
            for (int h = 0; h < _numHeads; h++)
            {
                MultiAxisRopeTables.BuildRows(_headPlans[h], positions, seq, RopeAngleOrder.MultiplyByInverse,
                    RopeTableLayout.InterleavedPair, 0, RopeTablePadding.Identity,
                    new Span<float>((float*)cos.DataPointer + (long)h * headLength, headLength),
                    new Span<float>((float*)sin.DataPointer + (long)h * headLength, headLength));
            }
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

    /// <summary>Token-major <c>(frame, row, col)</c> coordinates for the <c>(f, h, w)</c> grid, with the row and column offsets
    /// applied.</summary>
    private static float[] GridPositions(ReadOnlySpan<int> frameIndices, int gridH, int gridW, int offsetH, int offsetW)
    {
        int gridT = frameIndices.Length;
        float[] positions = new float[checked(gridT * gridH * gridW * NumAxes)];
        int at = 0;
        for (int fi = 0; fi < gridT; fi++)
            for (int hi = 0; hi < gridH; hi++)
                for (int wi = 0; wi < gridW; wi++)
                {
                    positions[at++] = frameIndices[fi];
                    positions[at++] = hi + offsetH;
                    positions[at++] = wi + offsetW;
                }
        return positions;
    }

    /// <summary>Applies the per-head interleaved-pair rotation in-place to <paramref name="x"/> <c>[S, heads, headDim]</c>.
    /// cos/sin are either <c>[S, headDim]</c> (shared across heads, standard Wan) or <c>[heads, S, headDim]</c>
    /// (per-head sigma_theta mode); the layout is detected from the cos rank.</summary>
    public void ApplyRotary(Tensor x, Tensor cos, Tensor sin, int heads)
    {
        int seq = (int)x.Shape[0];
        float* xp = (float*)x.DataPointer;
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        int pairs = _headDim / 2;
        bool perHead = cos.Shape.Rank == 3;
        for (int s = 0; s < seq; s++)
        {
            long sharedCosOff = (long)s * _headDim;
            for (int h = 0; h < heads; h++)
            {
                long cosOff = perHead ? ((long)h * seq + s) * _headDim : sharedCosOff;
                long xoff = ((long)s * heads + h) * _headDim;
                for (int i = 0; i < pairs; i++)
                {
                    int i0 = 2 * i;
                    float re = xp[xoff + i0], im = xp[xoff + i0 + 1];
                    float c = cp[cosOff + i0], sn = sp[cosOff + i0];
                    xp[xoff + i0] = re * c - im * sn;
                    xp[xoff + i0 + 1] = re * sn + im * c;
                }
            }
        }
    }
}
