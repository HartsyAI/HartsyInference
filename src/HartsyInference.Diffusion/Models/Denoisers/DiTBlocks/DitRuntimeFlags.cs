using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Per-process activation dtype for GPU-resident DiT block/attention hot paths. Standard-profile
/// default: F16 for every OPTED-IN model (half the HBM traffic of the bandwidth-bound
/// norm/modulate/gate/attention kernels) while the once-per-forward text/image/timestep paths and the tiny
/// per-channel modulation vectors stay F32. <c>HARTSY_DIT_F16=0</c> forces F32 everywhere (the pre-profile
/// baseline).
///
/// <para>Models opt in IN CODE by allocating their block activations with <see cref="Act"/> — the switch alone never
/// flips an un-audited model (F16 safety is per-arch: QK-normed attention bounds the scores; the SwiGLU/FFN
/// intermediate must stay under F16's 65504). Weights stay packed fp8 via the F16→e4m3 activation-quant GEMM path,
/// so VRAM is unchanged. First user: Krea2 (validated coherent; see <c>Krea2Block</c>/<c>Krea2Attention</c> for the
/// conversion pattern).</para></summary>
public static class DitDtype
{
    public static readonly DType Act =
        EnvSwitch.IsEnabled("HARTSY_DIT_F16", defaultOn: true) ? DType.F16 : DType.F32;
}

/// <summary>Per-process switch for CUDA-graph capture of a DiT denoise step (<c>HARTSY_DIT_GRAPH=1</c>): a model
/// whose step issues an identical op sequence every step captures it once and replays it with a single
/// <c>cuGraphLaunch</c> (host issue time → ~0). Models opt in IN CODE via the <c>IBackend.StepGraph*</c> API with
/// fixed boundary buffers (see <c>Krea2Transformer.ForwardPatched</c> for the reference pattern: per-step-varying
/// inputs refreshed via <c>CopyInto</c>, in-place Euler via <c>CfgEulerStep</c>, self-disables on capture failure).
/// Default off.</summary>
public static class DitStepGraph
{
    public static readonly bool Enabled =
        Environment.GetEnvironmentVariable("HARTSY_DIT_GRAPH") == "1";
}
