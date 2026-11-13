using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.Engine.Services;

/// <summary>A loaded image-to-mesh pipeline of the TRELLIS family and the gate that serialises generations on it. The pipeline mattes an image that carries no alpha with the engine's one shared matter, which the engine owns and releases, so the checkpoint is never resident twice and a free of this model leaves it alone. The stages share one set of resident weights, so a disposal waits for the generation in progress.</summary>
/// <typeparam name="TPipeline">The pipeline type; this runtime owns it.</typeparam>
internal abstract class MatterGatedRuntime<TPipeline> : IDisposable where TPipeline : IDisposable
{
    private int _disposed;

    protected MatterGatedRuntime(TPipeline pipeline) => Pipeline = pipeline;

    /// <summary>The pipeline.</summary>
    public TPipeline Pipeline { get; }

    /// <summary>Serialises generations on this runtime.</summary>
    public SemaphoreSlim Gate { get; } = new(1, 1);

    /// <summary>True once <see cref="Dispose"/> has started.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <summary>Whether a matting checkpoint is installed under <paramref name="modelsRoot"/> right now; the same lookup the shared matter makes.</summary>
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
        }
        finally
        {
            Gate.Release();
        }
    }
}
