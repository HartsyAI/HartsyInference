using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>The context-parallel K/V rendezvous as an attention rewrite: this rank's local K/V for the full sequence's, the first
/// <paramref name="prefixRows"/> rows (a joint stream's replicated text) taken from its own replica. Allocates and synchronizes,
/// so a CP step is never captured; <see cref="CpKvExchange.Exchange"/> owns the locals, also when it throws.</summary>
internal sealed class CpExchangeRewrite(CpKvExchange exchange, int rank, int prefixRows) : IKvRewrite
{
    public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v) =>
        exchange.Exchange(rank, k, v, replicatedPrefixRows: prefixRows);
}
