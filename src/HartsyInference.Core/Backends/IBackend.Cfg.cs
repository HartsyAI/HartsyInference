using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// The guided Euler step the samplers and the autoregressive logit blends run.
public partial interface IBackend
{
    /// <summary>In-place guided Euler step: forms the guided prediction of <paramref name="spec"/> from <paramref name="cond"/> and
    /// <paramref name="uncond"/> (which may be one tensor), rescales it by the spec's norm, and adds it to <paramref name="z"/>
    /// times the spec's delta. A zero delta leaves <paramref name="z"/> untouched. F32; neither prediction may share storage with
    /// <paramref name="z"/>. Contract: <see cref="Contracts.GuidedEulerContract"/>; reference: <see cref="GuidedEulerReference"/>.</summary>
    void GuidedEulerStep(Tensor z, Tensor cond, Tensor uncond, in GuidedEulerSpec spec) => GuidedEulerReference.Apply(z, cond, uncond, spec);
}
