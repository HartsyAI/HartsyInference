namespace HartsyInference.Engine.Services;

/// <summary>The requests the Kimi K3.1 Text route refuses by name, because the route does not build the sharded decoder (verified on CPU virtual
/// ranks only). Image refusals are in <c>RunKimiVision</c>, before the vision tower loads.</summary>
internal static class KimiK31Refusals
{
    /// <summary>Refuses any Kimi K3.1 request that spreads the model over more than one device, rather than loading it on one.</summary>
    /// <param name="deviceKey">The device the request names, possibly a composite such as <c>cuda:0+cuda:1</c>.</param>
    /// <param name="shardDeviceCount">The stage devices the engine placement would shard over, for the slot's device.</param>
    /// <param name="tensorParallelDegree">The engine's tensor-parallel degree.</param>
    /// <exception cref="NotSupportedException">The request asks for more than one device.</exception>
    internal static void RefuseMultiGpuPlacement(string deviceKey, int shardDeviceCount, int tensorParallelDegree)
    {
        if (tensorParallelDegree > 1)
        {
            throw new NotSupportedException($"Kimi K3.1 multi-GPU placement is not supported yet; the engine's tensor-parallel degree is "
                + $"{tensorParallelDegree}. Set it to 1 to load the model on a single device.");
        }
        if (deviceKey.Contains('+', StringComparison.Ordinal) || shardDeviceCount >= 2)
        {
            throw new NotSupportedException($"Kimi K3.1 multi-GPU placement is not supported yet; the request for '{deviceKey}' "
                + "names more than one device. Load it on a single device.");
        }
    }
}
