using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded Demucs stem separator plus the loader whose mmapped weights it references.</summary>
internal sealed class DemucsRunner(DemucsPipeline pipeline, IDisposable loader) : IDisposable
{
    /// <summary>Output sample rate in Hz.</summary>
    internal int SampleRate => pipeline.SampleRate;

    /// <summary>The stem names this checkpoint produces, in output order.</summary>
    internal IReadOnlyList<string> Sources => pipeline.Sources;

    /// <summary>Separates a stereo mix into its stems.</summary>
    internal (float[] Left, float[] Right)[] Separate(IBackend backend, float[] left, float[] right) =>
        pipeline.Separate(backend, left, right);

    /// <inheritdoc/>
    public void Dispose()
    {
        pipeline.Dispose();
        loader.Dispose();
    }
}
