using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.Engine.Services;

/// <summary>A loaded TRELLIS.2 pipeline with the matter it uses for images that carry no alpha.</summary>
internal sealed class Trellis2Runtime : MatterGatedRuntime<Trellis2ImageTo3DPipeline>
{
    private Trellis2Runtime(Trellis2ImageTo3DPipeline pipeline, Trellis2LazyMatter matter) : base(pipeline, matter)
    {
    }

    /// <summary>Loads the pipeline from <paramref name="localPath"/>; the matting checkpoint is looked up under <paramref name="modelsRoot"/> when an image needs it.</summary>
    public static Trellis2Runtime Load(IBackend backend, string localPath, string modelsRoot)
        => Open(modelsRoot, matter => Trellis2ImageTo3DPipeline.LoadFromPath(backend, localPath, matter), (pipeline, matter) => new Trellis2Runtime(pipeline, matter));
}
