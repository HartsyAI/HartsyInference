using HartsyInference.Core.Moe;
using HartsyInference.Core.Moe.Residency;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Moe;

namespace HartsyInference.Cuda;

/// <summary>Builds a <see cref="MoeExpertOffload"/> on a CUDA backend: a <see cref="CudaExpertCache"/> for the resident experts and the
/// packed CPU kernels for the rest.</summary>
public static class CudaMoeOffload
{
    /// <summary>Decay of an expert's recent-use score per routed access. With 8 experts per token over 48 layers a step is
    /// 1/384 of a token, so a score halves over about 50 tokens: recent enough to follow a prompt, long enough not to chase noise.</summary>
    public const double ScoreDecayPerAccess = 0.99996;

    /// <summary>How much colder than the current victim a candidate must be to replace it, so eviction does not flip between
    /// near-equal experts.</summary>
    public const double VictimHysteresis = 0.25;

    /// <summary>Creates the offload. Every expert tensor is kept out of the backend's auto-promotion: an expert's residency belongs to
    /// the cache, and a promoted copy would outlive the eviction meant to free it, or a streamed copy would stay behind.</summary>
    /// <param name="backend">The device the cache lives on.</param>
    /// <param name="budgetBytes">Device bytes the cache may hold.</param>
    /// <param name="hidden">Model width H.</param>
    /// <param name="intermediate">Expert inner width I.</param>
    /// <param name="expertTensors">Every routed expert projection of the model.</param>
    public static MoeExpertOffload Create(CudaBackend backend, long budgetBytes, int hidden, int intermediate, IEnumerable<Tensor> expertTensors)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(expertTensors);
        CudaExpertCache cache = new(backend, budgetBytes);
        try
        {
            if (backend.StreamingCache is CudaStreamingWeightCache streaming)
                streaming.ExcludeFromAutoPromotion(expertTensors);
            MoeExpertOffload? offload = null;
            // The runner reads each projection in its tensor's own format (Q4_K_M mixes Q4_K and Q6_K) through the offload's registry.
            PackedExpertHostRunner runner = new(hidden, intermediate, key => offload!.Resolve(key));
            offload = new MoeExpertOffload(cache, runner, new DecayedLfuHysteresisPolicy(ScoreDecayPerAccess, VictimHysteresis));
            return offload;
        }
        catch
        {
            cache.Dispose();
            throw;
        }
    }
}
