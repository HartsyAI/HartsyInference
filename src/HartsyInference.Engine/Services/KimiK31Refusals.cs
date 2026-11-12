using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Kimi;

namespace HartsyInference.Engine.Services;

/// <summary>The requests the Kimi K3.1 Text route refuses by name: image input, because the route loads no vision tower, and multi-GPU
/// placement, because the route does not build the sharded decoder (verified on CPU virtual ranks only).</summary>
internal static class KimiK31Refusals
{
    /// <summary>Refuses an image on a Kimi K3.1 slot. The checkpoint's vision tower is not loaded, so an image would be dropped silently otherwise.</summary>
    /// <exception cref="NotSupportedException">The slot holds a Kimi K3.1 model and the request carries an image.</exception>
    internal static void RefuseImage(IHostReferenceTextModel? model, bool hasImage)
    {
        if (hasImage && model is KimiK31TextModel)
            throw new NotSupportedException("Kimi K3.1 image input is not supported: the Text path loads the text decoder only and runs no vision tower (vision_tower.*).");
    }

    /// <summary>Refuses any Kimi K3.1 request that spreads the model over more than one device, rather than loading it on one.</summary>
    /// <param name="deviceKey">The device the request names, possibly a composite such as <c>cuda:0+cuda:1</c>.</param>
    /// <param name="shardDeviceCount">The stage devices the engine placement would shard over, for the slot's device.</param>
    /// <param name="tensorParallelDegree">The engine's tensor-parallel degree.</param>
    /// <exception cref="NotSupportedException">The request asks for more than one device.</exception>
    internal static void RefuseMultiGpuPlacement(string deviceKey, int shardDeviceCount, int tensorParallelDegree)
    {
        if (deviceKey.Contains('+', StringComparison.Ordinal) || shardDeviceCount >= 2 || tensorParallelDegree > 1)
        {
            throw new NotSupportedException($"Kimi K3.1 multi-GPU placement is not supported yet; the request for '{deviceKey}' "
                + "names more than one device. Load it on a single device.");
        }
    }
}
