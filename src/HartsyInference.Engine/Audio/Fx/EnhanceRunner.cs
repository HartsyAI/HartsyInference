using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded Resemble-Enhance stack (denoiser + LCFM enhancer + UnivNet vocoder) plus its weight loaders.</summary>
internal sealed class EnhanceRunner(ResembleEnhancePipeline pipeline, IDisposable[] loaders) : IDisposable
{
    /// <summary>Output sample rate in Hz — the pipeline is fixed at 44.1 kHz.</summary>
    internal int SampleRate => 44_100;

    /// <summary>Denoises and enhances a mono 44.1 kHz clip.</summary>
    internal float[] Enhance(IBackend backend, float[] mono44k, float lambd, float tau, int seed,
        int? nfe = null, string? solver = null) =>
        pipeline.Enhance(backend, mono44k, lambd, tau, seed, nfe, solver);

    /// <inheritdoc/>
    public void Dispose()
    {
        pipeline.Dispose();
        foreach (IDisposable loader in loaders)
        {
            loader?.Dispose();
        }
    }
}
