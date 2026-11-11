using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Models.Pixal3D;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// Device-memory hygiene before each Pixal3D flow. The stage before a 1.3B flow leaves the stream-ordered pool, cached activations, fused-attention
/// plans and the weights the backend promoted on its own (the shape decoder's) reserved, and a flow that starts on top of that runs many times slower
/// on a 12 GB card without failing or logging. The sweep returns all of it to the driver.
/// </summary>
/// <remarks>
/// The sweep does not read back a tensor that lives only on the device, so conditioning built by device kernels (the projected features, the patch
/// map) reads as zeros afterwards. <see cref="ReleaseAll"/> therefore materializes every live conditioning tensor on the host first.
/// </remarks>
public static class Pixal3DDeviceMemory
{
    /// <summary>Materializes the conditioning of <paramref name="live"/> on the host, then drains the stream and frees every cached device allocation,
    /// weights included. Call it only where no model is in use: each stage loads its own weights after it, and cached ones upload again on first use.
    /// A no-op on the CPU backend.</summary>
    public static void ReleaseAll(IBackend backend, params Pixal3DCondition[] live)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(live);
        foreach (Pixal3DCondition condition in live)
        {
            backend.OffloadActivation(condition.Global);
            backend.OffloadActivation(condition.Proj);
        }

        backend.Sync();
        backend.FreeAllDeviceMemory();
    }
}
