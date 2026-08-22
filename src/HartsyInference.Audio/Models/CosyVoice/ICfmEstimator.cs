using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>The velocity network <c>v_θ(x, t | μ, spk, cond)</c> evaluated inside the OT-CFM Euler solver: given the current sample <paramref name="x"/> (<c>[1, melBins, T]</c>), flow time <paramref name="t"/> ∈ [0,1], token-conditioning mel <paramref name="mu"/>, mel-projected speaker vector, and reference-mel <paramref name="cond"/>, returns the estimated velocity <c>dφ/dt</c> with the same shape as <paramref name="x"/>. Classifier-free guidance is applied by the solver, which calls this twice (conditional + zeroed-conditioning).</summary>
public interface ICfmEstimator
{
    /// <summary><paramref name="attnMask"/> null = the estimator's own default (full attention for the non-streaming path); a real mask opts a chunk-aware-trained estimator into that mode.</summary>
    Tensor Estimate(IBackend backend, Tensor x, Tensor mu, float t, Tensor spk, Tensor cond, Tensor? attnMask = null);
}
