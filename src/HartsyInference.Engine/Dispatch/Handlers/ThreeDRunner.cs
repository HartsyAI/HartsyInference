using HartsyInference.Engine;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Dispatch.Handlers;

/// <summary>A loaded image-to-3D pipeline (TripoSR or Hunyuan3D), exposed through a uniform generate delegate so the
/// handler does not branch on the concrete pipeline type at run time.</summary>
public sealed class ThreeDRunner : IModalityRunner
{
    private readonly IDisposable _pipeline;

    /// <summary>Creates a runner over <paramref name="pipeline"/> with its <paramref name="generate"/> entry point.</summary>
    public ThreeDRunner(string modelId, IDisposable pipeline,
        Func<ImageTo3DRequest, Action<GenerationProgress>?, ThreeDResult> generate)
    {
        ModelId = modelId;
        _pipeline = pipeline;
        Generate = generate;
    }

    /// <inheritdoc/>
    public Modality Modality => Modality.ThreeD;

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>Runs the loaded pipeline for one image-to-mesh request.</summary>
    public Func<ImageTo3DRequest, Action<GenerationProgress>?, ThreeDResult> Generate { get; }

    /// <inheritdoc/>
    public void Dispose() => _pipeline.Dispose();
}
