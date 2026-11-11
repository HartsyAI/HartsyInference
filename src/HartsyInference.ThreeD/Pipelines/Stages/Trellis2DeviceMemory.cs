using HartsyInference.Core.Backends;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// Device-memory hygiene between TRELLIS.2 stages. Each flow stage frees the weights it preloaded, but the stream-ordered pool, the
/// cached activations and the fused-attention plans of the stage before stay reserved, and the shape decoder, which never preloads,
/// leaves the weights the backend promoted on its own in the device cache. A 1.3B flow that starts on top of that runs 6 to 35 times
/// slower on a 12 GB card, without failing or logging anything: a fine shape step took 217 s instead of 15 s, a texture step 230 s
/// instead of 10 s.
/// </summary>
internal static class Trellis2DeviceMemory
{
    /// <summary>Drains the stream, then returns cached activations, attention plans and pool reservations to the driver. Cheap when
    /// there is nothing to return, and a no-op on the CPU backend.</summary>
    internal static void ReleaseBetweenStages(IBackend backend)
    {
        backend.Sync();
        backend.FreeActivations();
        backend.ReleaseAttentionExecutionCache();
        backend.TrimMemoryPool();
    }

    /// <summary>As <see cref="ReleaseBetweenStages"/>, and also drops every cached device weight, the shape decoder's included. Only
    /// for a point where no model is in use: the stage that calls it has its own weights on the host and preloads them after.</summary>
    internal static void ReleaseAll(IBackend backend)
    {
        backend.Sync();
        backend.FreeAllDeviceMemory();
    }
}
