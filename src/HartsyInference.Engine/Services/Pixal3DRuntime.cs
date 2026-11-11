using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.Engine.Services;

/// <summary>A loaded Pixal3D pipeline with the matter it uses for images that carry no alpha (the BiRefNet-matting file TRELLIS.2 also uses).</summary>
internal sealed class Pixal3DRuntime : MatterGatedRuntime<Pixal3DImageTo3DPipeline>
{
    private Pixal3DRuntime(Pixal3DImageTo3DPipeline pipeline, Trellis2LazyMatter matter) : base(pipeline, matter)
    {
    }

    /// <summary>Loads the pipeline from <paramref name="localPath"/>; the matting checkpoint is looked up under <paramref name="modelsRoot"/> when an image needs it.</summary>
    public static Pixal3DRuntime Load(IBackend backend, string localPath, string modelsRoot)
        => Open(modelsRoot, matter => Pixal3DImageTo3DPipeline.LoadFromPath(backend, localPath, matter), (pipeline, matter) => new Pixal3DRuntime(pipeline, matter));
}
