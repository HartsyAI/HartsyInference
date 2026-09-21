using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Activation dtype for GPU-resident DiT block/attention hot paths, read per generation so a
/// request-scoped profile decides it. Standard-profile
/// default: F16 for every OPTED-IN model (half the HBM traffic of the bandwidth-bound
/// norm/modulate/gate/attention kernels) while the once-per-forward text/image/timestep paths and the tiny
/// per-channel modulation vectors stay F32. <c>numerics.ditF16=false</c> forces F32 everywhere (the pre-profile
/// baseline).
///
/// <para>Models opt in IN CODE by allocating their block activations with <see cref="Act"/> — the switch alone never
/// flips an un-audited model (F16 safety is per-arch: QK-normed attention bounds the scores; the SwiGLU/FFN
/// intermediate must stay under F16's 65504). Weights stay packed fp8 via the F16→e4m3 activation-quant GEMM path,
/// so VRAM is unchanged. First user: Krea2 (validated coherent; see <c>Krea2Block</c>/<c>Krea2Attention</c> for the
/// conversion pattern).</para></summary>
public static class DitDtype
{
    public static DType Act => EngineKnobs.DitF16.Value ? DType.F16 : DType.F32;

    /// <summary>Casts an F32 block-input stream to <see cref="Act"/> (F16 on the <c>numerics.ditF16</c> hot path, else
    /// a no-op passthrough that returns the source unchanged). Disposes the source when it casts. Device-resident.
    /// Shared entry point for the opted-in DiTs (OmniGen2 keeps a private copy for its ref-conditioning variant).</summary>
    public static Tensor CastStreamToAct(IBackend backend, Tensor f32Stream)
    {
        if (Act == DType.F32)
            return f32Stream;
        Tensor casted = new(f32Stream.Shape, Act);
        backend.CastToF16(casted, f32Stream);
        f32Stream.Dispose();
        return casted;
    }
}

/// <summary>Per-generation switch for CUDA-graph capture of a DiT denoise step: a model
/// whose step issues an identical op sequence every step captures it once and replays it with a single
/// <c>cuGraphLaunch</c> (host issue time → ~0). Models opt in IN CODE via the <c>IBackend.StepGraph*</c> API with
/// fixed boundary buffers (see <c>Krea2Transformer.ForwardPatched</c> for the reference pattern: per-step-varying
/// inputs refreshed via <c>CopyInto</c>, in-place Euler via <c>CfgEulerStep</c>, self-disables on capture failure).
///
/// <para>Tri-state (the EnvSwitch convention): <see cref="Enabled"/> is the experimental opt-in gate (default
/// OFF — models where the graph was wall-neutral, e.g. GPU-bound Z-Image); <see cref="EnabledDefaultOn"/> is
/// default ON, for architectures where the per-generation graph is a validated win (host-issue-bound models —
/// Chroma). <c>numerics.ditGraph=false</c> kills both; <c>=1</c> forces both.</para></summary>
public static class DitStepGraph
{
    public static bool Enabled => EngineKnobs.DitGraph.Value;

    public static bool EnabledDefaultOn => EngineKnobs.DitGraphDefaultOn.Value;
}
