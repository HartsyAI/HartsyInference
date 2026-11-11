using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.Engine.Services;

/// <summary>A loaded TRELLIS.2 pipeline with the matter it uses for images that carry no alpha. The matter is always attached and looks for
/// its checkpoint on first use, so the service's up-front check and the pipeline's agree and a checkpoint installed after the first load
/// is still picked up.</summary>
internal sealed class Trellis2Runtime : IDisposable
{
    private readonly Trellis2LazyMatter _matter;
    private int _disposed;

    private Trellis2Runtime(Trellis2ImageTo3DPipeline pipeline, Trellis2LazyMatter matter)
    {
        Pipeline = pipeline;
        _matter = matter;
    }

    /// <summary>The pipeline.</summary>
    public Trellis2ImageTo3DPipeline Pipeline { get; }

    /// <summary>Serialises generations on this runtime; the stages share one set of resident weights.</summary>
    public SemaphoreSlim Gate { get; } = new(1, 1);

    /// <summary>Whether a matting checkpoint is installed under <paramref name="modelsRoot"/> right now; the same lookup the runtime's matter makes.</summary>
    public static bool MatterAvailable(string modelsRoot) => Trellis2ModelFiles.FindMatterFile(modelsRoot) is not null;

    /// <summary>Loads the pipeline from <paramref name="localPath"/>; the matting checkpoint is looked up under <paramref name="modelsRoot"/> when an image needs it.</summary>
    public static Trellis2Runtime Load(IBackend backend, string localPath, string modelsRoot)
    {
        Trellis2LazyMatter matter = new(modelsRoot);
        try
        {
            return new Trellis2Runtime(Trellis2ImageTo3DPipeline.LoadFromPath(backend, localPath, matter), matter);
        }
        catch
        {
            matter.Dispose();
            throw;
        }
    }

    /// <summary>True once <see cref="Dispose"/> has started.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

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
}
