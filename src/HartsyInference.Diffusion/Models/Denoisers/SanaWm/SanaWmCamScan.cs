using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Numerator-only camera delta-rule scan over frames, as upstream <c>_torch_cam_scan_single_chunk</c> and
/// <c>_torch_cam_scan_reference</c>. Per frame <c>f</c> with <c>Σ = Sᵀ</c> (<c>[Dk, Dv]</c>):
/// <c>Σ_f = d_f·(I − G_f)·Σ_{f−1} + U_fᵀ</c>, <c>G_f = Σ_s β k kᵀ</c>, <c>U_f = Σ_s β k vᵀ</c>, and the token output is
/// <c>q·Σ_f</c>. The forward pass is inclusive; the reverse pass reads the state before frame <c>f</c>'s update, so it
/// sums frames strictly after <c>f</c>. The output is the forward result plus the reverse result: exactly
/// <see cref="IBackend.FrameDeltaScan"/>.</summary>
internal static class SanaWmCamScan
{
    /// <summary>Returns the bidirectional scan of one batch item's <paramref name="query"/>, <paramref name="key"/> and
    /// <paramref name="value"/>, F32 token-major <c>[N, H·D]</c>, as a new <c>[N, H·D]</c> tensor. <paramref name="beta"/> is
    /// <c>[H, T, S]</c> and <paramref name="decay"/> is <c>[H, T]</c>. In a chunk-causal stream the forward pass starts from
    /// <paramref name="initialState"/> (<c>[H, D, D]</c>, the state after the earlier chunks) and <paramref name="finalState"/>
    /// receives the state after this chunk.</summary>
    internal static Tensor Run(IBackend backend, Tensor query, Tensor key, Tensor value, Tensor beta, Tensor decay, int frames,
        int spatial, int heads, int headDim, Tensor? initialState = null, Tensor? finalState = null)
    {
        Tensor output = new(new TensorShape((long)frames * spatial, (long)heads * headDim), DType.F32);
        try
        {
            backend.FrameDeltaScan(output, query, key, value, beta, decay, frames, spatial, heads, headDim, headDim, initialState,
                finalState);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }
}
