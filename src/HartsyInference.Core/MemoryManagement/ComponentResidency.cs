using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>Resident-or-lazy placement for a whole component's weights, driven by the resolved <see cref="VramPolicy"/>.
///
/// <para>A pipeline whose weights cannot be split into a block window (a single-shot DiT, a text encoder) has no
/// <see cref="VramPlanner"/> path. Before this, each such pipeline compared free VRAM against its own estimate, once,
/// at construction, and never read the policy, so <c>--vram-mode</c> could not reach it. A cached pipeline outlives the
/// request that built it, so the decision is taken per generation, against that generation's activation floor.</para>
/// </summary>
public static class ComponentResidency
{
    /// <summary>Decides whether a component's weights stay device-resident for this generation.</summary>
    /// <param name="freeBytes">Free device memory after the caller trimmed the pool. Pass the value the floor was computed from.</param>
    /// <param name="weightBytes">Bytes the component holds when resident.</param>
    /// <param name="floorBytes">Activation floor for this geometry. Auto resides only when weights and floor both fit.</param>
    /// <param name="allowResident">False when the build cannot be resident at all (e.g. larger than the card). A hard limit, not a policy choice.</param>
    public static PhasePlacement Decide(IBackend backend, string model, string phase, long freeBytes,
        long weightBytes, long floorBytes, bool allowResident)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(phase);

        if (!allowResident)
        {
            Logs.Info($"[VRAM] {model}/{phase}: lazy (the build cannot be resident on this device).");
            return PhasePlacement.Streamed;
        }

        LeverState streaming = VramPolicyRegistry.Resolve(backend).WeightStreaming;
        string detail = $"weights {ByteFormat.Mb(weightBytes)} + activation floor {ByteFormat.Mb(floorBytes)} vs {ByteFormat.Mb(freeBytes)} free";
        switch (streaming)
        {
            case LeverState.On:
                Logs.Info($"[VRAM] {model}/{phase}: lazy (the resolved policy streams weights). {detail}.");
                return PhasePlacement.Streamed;
            case LeverState.Off:
                Logs.Info($"[VRAM] {model}/{phase}: resident (the resolved policy forbids streaming). {detail}.");
                return PhasePlacement.Resident;
        }

        if (freeBytes <= 0 || freeBytes >= weightBytes + floorBytes)
        {
            // A backend that does not report live VRAM cannot be measured, so it stays resident as before.
            Logs.Info($"[VRAM] {model}/{phase}: resident — fits. {detail}.");
            return PhasePlacement.Resident;
        }
        Logs.Info($"[VRAM] {model}/{phase}: lazy — does not fit resident. {detail}.");
        return PhasePlacement.Streamed;
    }

    /// <summary>Releases a finished component's device weights before the next phase needs the memory.</summary>
    /// <remarks>This is the phase-boundary eviction ComfyUI performs before a VAE decode. It is required, not an
    /// optional lever: a lazily-uploaded component leaves its weights cached, and the next phase must not compete
    /// with them. <see cref="IBackend.FreeWeights"/> on an absent weight is a no-op, so this is safe on either path.
    /// <para>TODO: activations are not released here. The next phase's own decoder already frees them at its boundary
    /// (<c>VaeDecoder</c>), so a pipeline that needs it must call <see cref="IBackend.FreeActivations()"/> itself.</para></remarks>
    public static void ReleaseAfterPhase(IBackend backend, string model, string phase, IEnumerable<Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        backend.FreeWeights(weights);
        backend.TrimMemoryPool();
        Logs.Info($"[VRAM] {model}/{phase}: released its weights before the next phase.");
    }

    // TODO: MemorySupportReport.ReportUnimplemented still lists ActivationOffload, Caches, QuantizedCompute,
    // MultiGpuSpill and FreeAfterGeneration as read by nothing; those levers need their own consumers.
    // TODO: VramPlanner's block-window path and this whole-component path still decide separately. Unify once a lazy
    // component is expressed as a block window.
}
