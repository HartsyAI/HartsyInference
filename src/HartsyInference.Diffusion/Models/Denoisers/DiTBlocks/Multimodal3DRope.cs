using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Qwen2.5-VL-style **multimodal 3D RoPE** (M-RoPE), generic and reusable across the Qwen2.5-VL family (Lance, Qwen-Image, future Wan / Qwen-VL models). The head dim is split into per-axis sections <c>(t, h, w)</c> via <c>mrope_section</c> (e.g. <c>[16,24,24]</c> summing to head_dim/2). Each output channel draws its rotation frequency from one axis, assembled as <c>apply_multimodal_rotary_pos_emb</c> does (sections doubled across the two rotate-half halves, axis = <c>blockIndex % 3</c>), then a standard HF <c>rotate_half</c> rotation.
///
/// <para><b>Reusability:</b> this class knows nothing about MaPE or modality roles — those are encoded in the <c>positionIds</c> the caller passes (MaPE is just a per-role offset applied to the temporal axis when building positions). Any model that produces <c>(t,h,w)</c> position ids per token can use this directly.</para></summary>
public sealed unsafe class Multimodal3DRope
{
    private const int NumAxes = 3;

    private readonly int _headDim;
    private readonly int _halfDim;
    private readonly RopeFrequencyPlan _plan;

    /// <summary>Creates the M-RoPE for the given head dim, base θ, and per-axis sections <c>(t, h, w)</c> (must sum to head_dim/2).</summary>
    public Multimodal3DRope(int headDim, double theta, (int Temporal, int Height, int Width) section)
    {
        if (headDim % 2 != 0)
            throw new ArgumentException($"headDim {headDim} must be even.", nameof(headDim));
        _headDim = headDim;
        _halfDim = headDim / 2;
        if (section.Temporal + section.Height + section.Width != _halfDim)
            throw new ArgumentException($"mrope_section {section} must sum to head_dim/2 = {_halfDim}.", nameof(section));
        // apply_multimodal_rotary_pos_emb doubles the sections across the two rotate-half halves ([t,h,w,t,h,w]), so pair k of
        // the first half reads its section's axis and channel k + half repeats it — the rotate-half row layout.
        int[] pairAxes = new int[_halfDim];
        pairAxes.AsSpan(section.Temporal, section.Height).Fill(1);
        pairAxes.AsSpan(section.Temporal + section.Height, section.Width).Fill(2);
        _plan = RopeFrequencyPlan.CreateSectioned(theta, pairAxes, NumAxes);
    }

    /// <summary>Total per-head dimension.</summary>
    public int HeadDim => _headDim;

    /// <summary>Builds cos/sin tensors <c>[seq, headDim]</c> from per-token positions <c>positionIds</c> shaped <c>[seq, 3]</c> F32 holding <c>(t, h, w)</c>.</summary>
    public (Tensor Cos, Tensor Sin) BuildCosSin(Tensor positionIds)
    {
        int seq = (int)positionIds.Shape[0];
        if ((int)positionIds.Shape[1] != NumAxes)
            throw new ArgumentException($"positionIds last dim must be 3, got {positionIds.Shape[1]}.", nameof(positionIds));
        return MultiAxisRopeTables.BuildTensors(_plan, new ReadOnlySpan<float>(positionIds.DataPointer, checked(seq * NumAxes)), seq,
            RopeAngleOrder.MultiplyByInverse, RopeTableLayout.RotateHalf, 0, RopeTablePadding.Identity);
    }

    /// <summary>Applies the rotation in-place to <paramref name="x"/> shaped <c>[seq, numHeads, headDim]</c> (used for both Q and K; call once each). cos/sin are <c>[seq, headDim]</c>, broadcast over heads. <c>out = x·cos + rotate_half(x)·sin</c>, <c>rotate_half(x) = cat(-x[half:], x[:half])</c>.</summary>
    public void ApplyRotary(Tensor x, Tensor cos, Tensor sin)
    {
        int seq = (int)x.Shape[0];
        int numHeads = (int)x.Shape[1];
        int headDim = (int)x.Shape[2];
        if (headDim != _headDim)
            throw new ArgumentException($"x head_dim {headDim} != rope head_dim {_headDim}.", nameof(x));

        float* xPtr = (float*)x.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        int half = _halfDim;
        Span<float> snapshot = stackalloc float[half];

        for (int s = 0; s < seq; s++)
        {
            float* c = cosPtr + (long)s * _headDim;
            float* sn = sinPtr + (long)s * _headDim;
            for (int h = 0; h < numHeads; h++)
            {
                float* vec = xPtr + (((long)s * numHeads + h) * _headDim);
                for (int i = 0; i < half; i++) snapshot[i] = vec[i];
                for (int i = 0; i < half; i++)
                {
                    float lower = snapshot[i];
                    float upper = vec[i + half];
                    vec[i] = lower * c[i] - upper * sn[i];
                    vec[i + half] = upper * c[i + half] + lower * sn[i + half];
                }
            }
        }
    }
}
