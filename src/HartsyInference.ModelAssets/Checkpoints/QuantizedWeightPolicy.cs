using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>Reconciles a checkpoint's quantized weights with what the backend that will run them can actually hold.</summary>
/// <remarks><para>A backend's packed-weight kernels cover some dtypes and not others. Handing it one it has no kernel
/// for does not fail at load — it fails in the middle of the first GEMM, several minutes and one confusing stack trace
/// later. This pass moves that decision to load time: anything the backend cannot consume packed is widened on the
/// host, which costs memory and is correct, and the log says which weights and why.</para>
/// <para>The LLM loader has carried this rule privately since GGUF text models were wired; no diffusion recipe applied
/// it, which is why a Q2_K or IQ4_NL diffusion GGUF — formats the community publishes routinely — died inside
/// <c>Linear</c> rather than loading slowly. This gives every other model the same behaviour, asked of the backend
/// rather than hardcoded. <c>GgufLanguageModel.PrepareWeights</c> still keeps its own copy of the set, because it
/// widens weights before any backend is chosen and takes a CPU/GPU flag instead; folding it in means giving that load
/// path the backend, which is a change to the text-generation surface rather than to this one.</para></remarks>
public static class QuantizedWeightPolicy
{
    /// <summary>Widens, in place, every weight whose dtype <paramref name="backend"/> cannot hold packed.</summary>
    /// <param name="weights">The converted weight dictionary; entries this replaces are disposed by the returned handle.</param>
    /// <param name="backend">The backend that will consume the weights.</param>
    /// <param name="wideDType">What a widened weight becomes. F16 halves the cost of F32 and is what every dequant kernel targets anyway.</param>
    /// <returns>A handle owning the tensors this created. Disposing it frees them, so it belongs to whatever owns the weight dictionary.</returns>
    public static PreparedWeights PrepareForBackend(IDictionary<string, Tensor> weights, IBackend backend,
        DType wideDType = default)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return PrepareFor(weights, backend.SupportsResidentQuant, backend.Capabilities.Name, wideDType);
    }

    /// <summary>The same widening against an explicit capability predicate, for a consumer that is not a backend.</summary>
    /// <remarks>Offline tooling reads a checkpoint to re-quantize or inspect it with no device attached, and wants the
    /// same rule stated against whatever it can decode. Naming the consumer keeps the log line meaningful.</remarks>
    public static PreparedWeights PrepareFor(IDictionary<string, Tensor> weights,
        Func<DType, bool> supportsResidentQuant, string consumerName, DType wideDType = default)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(supportsResidentQuant);
        if (wideDType == default) wideDType = DType.F16;

        List<Tensor> created = new();
        List<string>? widenedKeys = null;
        Dictionary<string, int> countByDtype = new(StringComparer.Ordinal);
        try
        {
            foreach (string key in weights.Keys.ToList())
            {
                Tensor weight = weights[key];
                if (!weight.DType.IsQuantized || supportsResidentQuant(weight.DType))
                    continue;
                Tensor wide = GgufDequantizer.Dequantize(weight, wideDType);
                created.Add(wide);
                weights[key] = wide;
                (widenedKeys ??= new List<string>()).Add(key);
                countByDtype.TryGetValue(weight.DType.Name, out int seen);
                countByDtype[weight.DType.Name] = seen + 1;
            }
        }
        catch
        {
            foreach (Tensor tensor in created) tensor.Dispose();
            throw;
        }

        if (widenedKeys is not null)
        {
            string breakdown = string.Join(", ", countByDtype.Select(entry => $"{entry.Value}×{entry.Key}"));
            Logs.Info($"Quant policy: {consumerName} has no packed-weight kernel for {breakdown}; "
                + $"widened {widenedKeys.Count} weights to {wideDType.Name} on the host.");
        }
        return new PreparedWeights(created);
    }

    /// <summary>Owns the widened copies <see cref="PrepareForBackend"/> put into a weight dictionary.</summary>
    /// <remarks>The originals borrow the checkpoint's memory map and are freed with it; these do not, so something has
    /// to hold them. Keeping them in one handle means a recipe disposes a single object rather than tracking which of
    /// its weights happen to be owned.</remarks>
    public sealed class PreparedWeights(IReadOnlyList<Tensor> owned) : IDisposable
    {
        private int _disposed;

        /// <summary>How many weights were widened; zero when the backend could hold everything packed.</summary>
        public int WidenedCount => owned.Count;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }
}
