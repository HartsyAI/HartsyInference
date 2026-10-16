using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Engine.Placement;

/// <summary>Sets up expert offload for a loaded mixture-of-experts model, the same way for every service that offers it: the
/// dense weights are preloaded, a device expert cache of the planned budget holds the routed experts it can, seeded evenly across
/// the layers, and the rest run on the CPU.</summary>
public static class ExpertOffloadSetup
{
    /// <summary>Attaches an expert offload of <paramref name="budgetBytes"/> to <paramref name="model"/> on <paramref name="backend"/>.
    /// Dequantized weight casts stop being cached on the backend first: a prefill's casts would otherwise grow into the memory the
    /// plan gave the expert cache. The caller owns the returned offload and disposes it with the model.</summary>
    /// <param name="backend">The device the model runs on; must be CUDA.</param>
    /// <param name="model">The loaded model.</param>
    /// <param name="denseGroups">Every weight except the routed experts, grouped for <see cref="IBackend.PreloadWeightGroups"/>.</param>
    /// <param name="budgetBytes">Device bytes the expert cache may hold.</param>
    /// <param name="logTag">Prefix for the log line that reports the cache.</param>
    /// <exception cref="HartsyInferenceException">The backend is not CUDA.</exception>
    public static MoeExpertOffload Attach(IBackend backend, IExpertOffloadTarget model, IEnumerable<IReadOnlyList<Tensor>> denseGroups,
        long budgetBytes, string logTag)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(denseGroups);
        if (backend is not CudaBackend cuda)
        {
            throw new HartsyInferenceException($"Expert offload needs a CUDA backend; this model runs on {backend.Capabilities.Name}.");
        }
        backend.CacheWeightCasts = false;
        backend.PreloadWeightGroups(denseGroups);
        List<Tensor> experts = [.. model.EnumerateExpertWeights()];
        MoeExpertOffload offload = CudaMoeOffload.Create(cuda, budgetBytes, model.ExpertHiddenSize, model.ExpertIntermediateSize, experts);
        try
        {
            IReadOnlyList<(int Layer, int ExpertCount)> layers = model.AttachExpertOffload(offload);
            int numExperts = layers.Count == 0 ? 0 : layers.Max(static l => l.ExpertCount);
            long expertBytes = experts.Sum(static t => t.DType.ComputeByteCount(t.ElementCount));
            int totalExperts = layers.Sum(static l => l.ExpertCount);
            long perExpert = Math.Max(1, expertBytes / Math.Max(1, totalExperts));
            int perLayer = (int)Math.Min(numExperts, budgetBytes / perExpert / Math.Max(1, layers.Count));
            offload.Seed(layers, perLayer);
            Logs.Info($"{logTag} Expert offload: {ByteFormat.GbF1(budgetBytes)} cache, seeded {perLayer} of {numExperts} experts in "
                + $"each of {layers.Count} layers.");
            return offload;
        }
        catch
        {
            offload.Dispose();
            throw;
        }
    }
}
