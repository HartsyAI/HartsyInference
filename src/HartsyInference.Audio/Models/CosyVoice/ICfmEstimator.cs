using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>The velocity network <c>v_θ(x, t | μ, spk, cond)</c> evaluated inside the OT-CFM Euler
/// solver. Given the current sample <paramref name="x"/> (<c>[1, melBins, T]</c>), the flow time
/// <paramref name="t"/> ∈ [0,1], the token-conditioning mel <paramref name="mu"/>, the (already
/// mel-projected) speaker vector, and the reference-mel <paramref name="cond"/>, it returns the
/// estimated velocity <c>dφ/dt</c> with the same shape as <paramref name="x"/>. Classifier-free
/// guidance is applied by the solver, which calls this twice (conditional + zeroed-conditioning).</summary>
public interface ICfmEstimator
{
    Tensor Estimate(IBackend backend, Tensor x, Tensor mu, float t, Tensor spk, Tensor cond);
}
