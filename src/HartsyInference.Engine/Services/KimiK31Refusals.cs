using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Kimi;

namespace HartsyInference.Engine.Services;

/// <summary>The requests the Kimi K3.1 Text route refuses by name: image input the checkpoint's vision tower cannot take, and multi-GPU
/// placement, because the route does not build the sharded decoder (verified on CPU virtual ranks only).</summary>
internal static class KimiK31Refusals
{
    /// <summary>Loads a Kimi K3.1 vision tower on the first image request, and refuses by name when the checkpoint cannot take
    /// images, so the image is never dropped silently. A text request never loads the tower.</summary>
    /// <exception cref="NotSupportedException">The slot holds a Kimi K3.1 model that refuses images, and the request carries one.</exception>
    internal static void RefuseImage(IHostReferenceTextModel? model, bool hasImage)
    {
        if (hasImage && model is KimiK31TextModel kimi) kimi.Vision.Route();
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
