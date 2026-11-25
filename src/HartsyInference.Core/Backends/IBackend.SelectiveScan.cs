using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Mamba-1 selective state-space scan; the host reference is the default, so every backend runs it until it has a kernel.
public partial interface IBackend
{
    /// <summary>Mamba-1 selective scan over <paramref name="rows"/> independent sequences of <paramref name="frames"/> steps.
    /// Per row, channel <c>i</c> and step <c>t</c>: <c>h[n] = exp(delta·a[i,n])·h[n] + delta·b[t,n]·u</c>,
    /// <c>y = Σ_n c[t,n]·h[n] + d[i]·u</c>, then <c>y ·= silu(z)</c> when <paramref name="z"/> is given. The caller prepares
    /// <c>delta</c> (softplus already applied) and <c>a</c> (<c>−exp(A_log)</c>).</summary>
    /// <param name="output">F32 <c>[rows, frames, channels]</c>; may alias <paramref name="u"/>.</param>
    /// <param name="u">F32 <c>[rows, frames, channels]</c> convolved inputs.</param>
    /// <param name="delta">F32 <c>[rows, frames, channels]</c> positive step sizes.</param>
    /// <param name="a">F32 <c>[channels, states]</c> decay rates, negative.</param>
    /// <param name="b">F32 <c>[rows, frames, states]</c> input projections.</param>
    /// <param name="c">F32 <c>[rows, frames, states]</c> output projections.</param>
    /// <param name="d">F32 <c>[channels]</c> skip gains.</param>
    /// <param name="z">F32 <c>[rows, frames, channels]</c> gate (pre-SiLU), or null for no gate.</param>
    /// <param name="state">F32 <c>[rows, channels, states]</c> carried state, read as the initial state and rewritten with the
    /// state after the last step; null starts from zero and keeps nothing.</param>
    void SelectiveScan(Tensor output, Tensor u, Tensor delta, Tensor a, Tensor b, Tensor c, Tensor d, Tensor? z, Tensor? state,
        int rows, int frames, int channels, int states) =>
        SelectiveScanReference.Apply(output, u, delta, a, b, c, d, z, state, rows, frames, channels, states);
}
