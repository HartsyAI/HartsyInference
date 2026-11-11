using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// Device-memory hygiene between TRELLIS.2 stages. Each flow stage frees the weights it preloaded, but the stream-ordered pool, the
/// cached activations and the fused-attention plans of the stage before stay reserved, and the shape decoder, which never preloads,
/// leaves the weights the backend promoted on its own in the device cache. A 1.3B flow that starts on top of that runs 6 to 35 times
/// slower on a 12 GB card, without failing or logging anything: a fine shape step took 217 s instead of 15 s, a texture step 230 s
/// instead of 10 s.
/// <para>Sweeping activations is not a read-back: a tensor an op produced and nobody read on the host (the DINOv3 conditioning is one)
/// exists only on the device, and a sweep would leave its host buffer, which is zeros, to be uploaded by the next flow. Every tensor
/// that is still needed is therefore offloaded to the host first.</para>
/// </summary>
internal static class Trellis2DeviceMemory
{
    /// <summary>Offloads <paramref name="live"/> to the host, drains the stream, then returns cached activations, attention plans and pool
    /// reservations to the driver. Cheap when there is nothing to return, and a no-op on the CPU backend.</summary>
    /// <param name="live">Every tensor a later stage still reads that an op may have produced on the device.</param>
    internal static void ReleaseBetweenStages(IBackend backend, params Tensor[] live)
    {
        Offload(backend, live);
        backend.Sync();
        backend.FreeActivations();
        backend.ReleaseAttentionExecutionCache();
        backend.TrimMemoryPool();
    }

    /// <summary>As <see cref="ReleaseBetweenStages"/>, and also drops every cached device weight, the shape decoder's included. Only
    /// for a point where no model is in use: the stage that calls it has its own weights on the host and preloads them after.</summary>
    internal static void ReleaseAll(IBackend backend, params Tensor[] live)
    {
        Offload(backend, live);
        backend.Sync();
        backend.FreeAllDeviceMemory();
    }

    private static void Offload(IBackend backend, Tensor[] live)
    {
        foreach (Tensor tensor in live)
            backend.OffloadActivation(tensor);
    }
}
