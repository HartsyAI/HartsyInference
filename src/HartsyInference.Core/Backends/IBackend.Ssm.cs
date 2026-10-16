using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Gated-DeltaNet (Qwen3.5) prefill over a whole token sequence; the host reference is the default.
public partial interface IBackend
{
    /// <summary>Gated-DeltaNet prefill, part 1, over <c>seq</c> tokens: the causal depthwise conv over
    /// <c>[history ; qkvMixed]</c> plus SiLU, the <c>[q | k | v]</c> split, and the per-head L2 norm of q (then scaled by
    /// <paramref name="qScale"/>) and k. <paramref name="history"/> (<c>[kernel−1, convDim]</c>, the previous call's last
    /// pre-conv rows) is advanced to this call's last rows, so prefill can run in chunks.</summary>
    /// <param name="q">F32 <c>[seq, kHeads·skDim]</c>.</param>
    /// <param name="k">F32 <c>[seq, kHeads·skDim]</c>.</param>
    /// <param name="v">F32 <c>[seq, vHeads·svDim]</c>.</param>
    /// <param name="qkvMixed">F32 <c>[seq, convDim]</c> fused projection, <c>convDim = 2·kHeads·skDim + vHeads·svDim</c>.</param>
    /// <param name="history">F32 <c>[kernel−1, convDim]</c> conv carry, updated in place.</param>
    /// <param name="convW">F32 <c>[convDim, kernel]</c> depthwise taps.</param>
    void SsmConvSplitPrefill(Tensor q, Tensor k, Tensor v, Tensor qkvMixed, Tensor history, Tensor convW,
        int kernel, int kHeads, int skDim, int vHeads, int svDim, float eps, float qScale) =>
        GatedDeltaNetReference.ConvSplit(q, k, v, qkvMixed, history, convW, kernel, kHeads, skDim, vHeads, svDim, eps, qScale);

    /// <summary>Gated-DeltaNet prefill, part 2: the delta-rule recurrence over every token in order, carrying
    /// <paramref name="state"/>, then the gated RMSNorm <c>rmsnorm(o)·normW·silu(z)</c> per head. Per token and value
    /// head: <c>S ·= exp(softplus(alpha + dtBias)·ssmA)</c>, <c>S += k ⊗ ((v − S·k)·sigmoid(betaRaw))</c>, <c>o = S·q</c>.</summary>
    /// <param name="output">F32 <c>[seq, hv·sv]</c>.</param>
    /// <param name="state">F32 <c>[hv, sv, sk]</c> recurrent state, updated in place.</param>
    /// <param name="q">F32 <c>[seq, kHeads·sk]</c> from <see cref="SsmConvSplitPrefill"/>.</param>
    /// <param name="k">F32 <c>[seq, kHeads·sk]</c> from <see cref="SsmConvSplitPrefill"/>.</param>
    /// <param name="v">F32 <c>[seq, hv·sv]</c>.</param>
    /// <param name="z">F32 <c>[seq, hv·sv]</c> output-gate projection (pre-SiLU).</param>
    /// <param name="alphaRaw">F32 <c>[seq, hv]</c> decay projection.</param>
    /// <param name="betaRaw">F32 <c>[seq, hv]</c> write-strength projection (pre-sigmoid).</param>
    /// <param name="repeat">Value heads per key head (key head = value head / repeat).</param>
    void SsmDeltaPrefill(Tensor output, Tensor state, Tensor q, Tensor k, Tensor v, Tensor z, Tensor alphaRaw,
        Tensor betaRaw, Tensor dtBias, Tensor ssmA, Tensor normW, int hv, int sv, int sk, int repeat, float eps) =>
        GatedDeltaNetReference.DeltaScan(output, state, q, k, v, z, alphaRaw, betaRaw, dtBias, ssmA, normW, hv, sv, sk, repeat, eps);
}
