using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.Engine.Services;

/// <summary>A loaded image-to-mesh pipeline of the TRELLIS family with the matter it uses for images that carry no alpha, and the gate that serialises
/// generations on it. The matter is always attached and looks for its checkpoint on first use, so the service's up-front check and the pipeline's
/// agree and a checkpoint installed after the first load is still picked up. The stages share one set of resident weights, so a disposal waits for the
/// generation in progress.</summary>
/// <typeparam name="TPipeline">The pipeline type; this runtime owns it.</typeparam>
internal abstract class MatterGatedRuntime<TPipeline> : IDisposable where TPipeline : IDisposable
{
    private readonly Trellis2LazyMatter _matter;
    private int _disposed;

    protected MatterGatedRuntime(TPipeline pipeline, Trellis2LazyMatter matter)
    {
        Pipeline = pipeline;
        _matter = matter;
    }

    /// <summary>The pipeline.</summary>
    public TPipeline Pipeline { get; }

    /// <summary>Serialises generations on this runtime.</summary>
    public SemaphoreSlim Gate { get; } = new(1, 1);

    /// <summary>True once <see cref="Dispose"/> has started.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <summary>Whether a matting checkpoint is installed under <paramref name="modelsRoot"/> right now; the same lookup the runtime's matter makes.</summary>
    public static bool MatterAvailable(string modelsRoot) => Trellis2ModelFiles.FindMatterFile(modelsRoot) is not null;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        // A generation in progress holds the gate; the weights must not go under it.
        Gate.Wait();
        try
        {
            Pipeline.Dispose();
            _matter.Dispose();
        }
        finally
        {
            Gate.Release();
        }
    }

    /// <summary>Builds the matter, then the pipeline and the runtime around it; the matter is released if either step throws.</summary>
    protected static TRuntime Open<TRuntime>(
        string modelsRoot, Func<IForegroundMatter, TPipeline> loadPipeline, Func<TPipeline, Trellis2LazyMatter, TRuntime> create)
    {
        Trellis2LazyMatter matter = new(modelsRoot);
        try
        {
            return create(loadPipeline(matter), matter);
        }
        catch
        {
            matter.Dispose();
            throw;
        }
    }
}
