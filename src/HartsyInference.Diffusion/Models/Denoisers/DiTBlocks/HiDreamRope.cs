using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis Rotary Position Embedding for HiDream-I1. The rotation math matches BFL/Flux's pair-rotation RoPE exactly (same "stack [cos, -sin, sin, cos] then einsum" convention used by <c>apply_rope</c> in <c>transformer_hidream_image.py</c>), so this is a thin wrapper around <see cref="FluxRope"/> with the HiDream axes [64, 32, 32] and a different position-id layout.
/// <para>Position ids per token are <c>(layer_axis, row, col)</c>: image tokens always use layer_axis = 0 and their patch (row, col); text tokens use the same 3-tuple but all zeros, so the text path collapses to identity rotations.</para></summary>
public sealed class HiDreamRope
{
    private readonly FluxRope _inner;

    /// <summary>Creates a HiDream RoPE with the per-axis dimensions from <see cref="HiDreamConfig.AxesDimsRope"/>. Default: [64, 32, 32], which sums to head_dim=128.</summary>
    public HiDreamRope(int[]? axesDim = null, int theta = 10000)
    {
        _inner = new FluxRope(axesDim ?? [64, 32, 32], theta);
    }

    /// <summary>Total head dimension (sum of axes — 128 for HiDream-I1).</summary>
    public int HeadDim => _inner.HeadDim;

    /// <summary>Builds a [totalSeqLen, 3] position-id tensor laid out as (image tokens first, then text tokens). Image tokens carry their (0, row, col); text tokens carry (0, 0, 0). This matches the diffusers reference where <c>img_ids</c> spans [0, max_seq) for image tokens and <c>txt_ids</c> is all zeros.</summary>
    /// <param name="imgSeqLen">Number of image tokens (= patH * patW).</param>
    /// <param name="patH">Patchified height (latent_h / patch_size).</param>
    /// <param name="patW">Patchified width (latent_w / patch_size).</param>
    /// <param name="txtSeqLen">Total text token count (sum of all encoder hidden states concatenated for the joint stream).</param>
    public static unsafe Tensor BuildPositionIds(int imgSeqLen, int patH, int patW, int txtSeqLen)
    {
        int totalSeqLen = imgSeqLen + txtSeqLen;
        TensorShape shape = new TensorShape(totalSeqLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);
        float* ptr = (float*)posIds.DataPointer;

        // Image tokens: [0, row, col]
        for (int row = 0; row < patH; row++)
        {
            for (int col = 0; col < patW; col++)
            {
                int idx = row * patW + col;
                ptr[idx * 3 + 0] = 0f;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }
        }

        // Text tokens: all-zero positional ids — RoPE collapses to identity for these.
        for (int i = imgSeqLen * 3; i < totalSeqLen * 3; i++)
            ptr[i] = 0f;

        return posIds;
    }

    /// <summary>Precomputes cos/sin tables from the position ids. Call once per resolution.</summary>
    public void Precompute(Tensor posIds) => _inner.Precompute(posIds);

    /// <summary>Applies RoPE rotation to Q and K in place. Q/K must be [B, numHeads, seqLen, headDim].</summary>
    public void Forward(Tensor q, Tensor k, int batch, int numHeads, int seqLen) =>
        _inner.Forward(q, k, batch, numHeads, seqLen);
}
