using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Frame-wise bidirectional delta-rule scan (SANA-WM Gated-DeltaNet); the host reference is the default.
public partial interface IBackend
{
    /// <summary>Bidirectional delta-rule scan over frames, every head independent. Tokens are frame-major,
    /// <c>N = frames·spatial</c>, and each frame's tokens update the state together. Per head and frame <c>t</c>, with
    /// <c>G_t = Σ_s β k kᵀ</c> (<c>[Dk, Dk]</c>) and <c>U_t = Σ_s β k vᵀ</c> (<c>[Dk, Dv]</c>) over the frame's tokens, the
    /// state advances as <c>S ← d_t·(S − G_t·S) + U_t</c>. The forward pass is inclusive (<c>F_t</c> is the state after frame
    /// <c>t</c>); the reverse pass is exclusive (<c>R_t</c> holds frames after <c>t</c> only). Each token's output row is
    /// <c>q · (F_t + R_t)</c>. The forward pass starts from <paramref name="initialState"/> and the reverse pass from zero, so
    /// one call per chunk with the forward state carried between calls is a chunk-causal scan: every earlier frame through
    /// the state, later frames of the same chunk through the reverse pass.</summary>
    /// <param name="output">F32 token-major <c>[N, heads·valueDim]</c>, overwritten.</param>
    /// <param name="query">F32 token-major <c>[N, heads·keyDim]</c>.</param>
    /// <param name="key">F32 token-major <c>[N, heads·keyDim]</c>.</param>
    /// <param name="value">F32 token-major <c>[N, heads·valueDim]</c>, or null for a value of 1 per token and head
    /// (<paramref name="valueDim"/> must then be 1): the scan of a linear-attention denominator.</param>
    /// <param name="beta">F32 write strength <c>[heads, frames, spatial]</c>.</param>
    /// <param name="decay">F32 per-frame decay <c>[heads, frames]</c>.</param>
    /// <param name="initialState">F32 <c>[heads, keyDim, valueDim]</c>, the forward state before frame 0; null starts from zero.</param>
    /// <param name="finalState">F32 <c>[heads, keyDim, valueDim]</c>, overwritten with the forward state after the last frame;
    /// null discards it. It must not alias <paramref name="initialState"/>.</param>
    void FrameDeltaScan(Tensor output, Tensor query, Tensor key, Tensor? value, Tensor beta, Tensor decay, int frames,
        int spatial, int heads, int keyDim, int valueDim, Tensor? initialState = null, Tensor? finalState = null) =>
        FrameDeltaScanReference.Run(output, query, key, value, beta, decay, frames, spatial, heads, keyDim, valueDim, initialState,
            finalState);
}
