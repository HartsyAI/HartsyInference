using System.Reflection;
using HartsyInference.Engine.Services;
using HartsyInference.Vision.Rmbg;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The engine's free memory disposes the live vision service and keeps using it, so its caches must be empty afterwards: a cached model whose
/// weights mapping was closed failed the next request ("Cannot access a disposed object: MmapHandle" on the second RMBG cutout in a SwarmUI session).</summary>
public sealed class VisionServiceReleaseTests
{
    [Fact]
    public void FreeMemoryEmptiesTheVisionCachesSoTheNextRequestLoadsAfresh()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        VisionService vision = engine.VisionInternal;
        // Put a model in the RMBG cache the way RmbgAlpha does, without weights (none are needed to hold it).
        MethodInfo getOrLoad = typeof(VisionService).GetMethod("GetOrLoad", BindingFlags.Instance | BindingFlags.NonPublic)!.MakeGenericMethod(typeof(BriaRmbg));
        object cache = typeof(VisionService).GetField("_rmbgCache", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(vision)!;
        BriaRmbg first = (BriaRmbg)getOrLoad.Invoke(vision, [cache, "rmbg.safetensors", new Func<BriaRmbg>(() => new BriaRmbg())])!;
        Assert.Equal(1, vision.CachedModelCount);

        engine.FreeMemory();

        Assert.Equal(0, vision.CachedModelCount);
        BriaRmbg second = (BriaRmbg)getOrLoad.Invoke(vision, [cache, "rmbg.safetensors", new Func<BriaRmbg>(() => new BriaRmbg())])!;
        Assert.NotSame(first, second);
    }
}
