using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Per-layer key/value cache abstraction so <see cref="GenericTransformer"/> is agnostic to the cache
/// backing store. Two implementations exist: the device-resident <see cref="KvCache"/> (grown on-device with
/// <c>Concat</c>) and the host <c>StreamingKvCache</c> (pre-allocated, CPU memcpy). The transformer codes
/// against this interface so audio models keep their existing host cache while the LLM pipeline uses the
/// resident one.
///
/// <para>Ownership contract: <see cref="GetK"/>/<see cref="GetV"/> return a tensor <b>owned by the cache</b>
/// and valid until the next <see cref="Append"/> on that layer (or <see cref="Reset"/>/dispose). Callers must
/// NOT dispose it. The returned tensor is the exact populated prefix <c>[1, num_kv_heads, length, head_dim]</c>
/// (no padding), ready for the GQA repeat + SDPA.</para>
///
/// <para>Extends <see cref="IDisposable"/> so a decode cache can be created via a factory
/// (<see cref="KvCaches.ForDecode"/>) and scoped with <c>using</c> without the caller naming the concrete type.
/// All implementations (device <c>KvCache</c>, host <c>StreamingKvCache</c>, <c>FixedKvCache</c>) are disposable.</para></summary>
public interface IKvCache : IDisposable
{
    /// <summary>Number of layers this cache stores.</summary>
    int NumLayers { get; }

    /// <summary>Tokens currently committed (before the in-flight step's <see cref="AdvanceLength"/>).</summary>
    int CurrentLength { get; }

    /// <summary>Appends a new K/V block <c>[1, num_kv_heads, tNew, head_dim]</c> for <paramref name="layer"/>.
    /// <paramref name="backend"/> is used by device implementations for the on-device copy (ignored by host
    /// ones). Does not change <see cref="CurrentLength"/> — call <see cref="AdvanceLength"/> after all layers.
    /// (Named to avoid colliding with the host cache's existing <c>Append</c> used by non-LLM models.)</summary>
    void AppendStep(IBackend backend, int layer, Tensor newK, Tensor newV);

    /// <summary>The layer's cache-owned populated K prefix <c>[1, num_kv_heads, length, head_dim]</c>.</summary>
    Tensor KeyPrefix(int layer);

    /// <summary>The layer's cache-owned populated V prefix <c>[1, num_kv_heads, length, head_dim]</c>.</summary>
    Tensor ValuePrefix(int layer);

    /// <summary>Advances the shared position counter by <paramref name="by"/> (once per step, after all layers).</summary>
    void AdvanceLength(int by);

    /// <summary>Rolls the committed length back to <paramref name="newLength"/> (must be &lt;=
    /// <see cref="CurrentLength"/>), discarding K/V entries beyond it without disturbing anything before it.
    /// Used by speculative decoding to drop rejected draft tokens' K/V after a batched verify — the rejected
    /// entries were already physically written (the verification forward pass has to run before acceptance
    /// is known), so this is a real rollback, not just "don't write it in the first place." Default throws:
    /// only implementations speculative decoding actually drives (<see cref="FixedKvCache"/>,
    /// <see cref="PagedKvCache"/>) need to support it.</summary>
    void Truncate(int newLength) => throw new NotSupportedException($"{GetType().Name} does not support Truncate (speculative decoding rollback).");

    /// <summary>Drops stored K/V and resets the position counter for reuse.</summary>
    void Reset();
}
