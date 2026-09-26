using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>Unit-variance Gaussian noise for stochastic samplers, keyed by the interval it is drawn for.</summary>
public interface INoiseSource : IDisposable
{
    /// <summary>Returns a new host tensor of N(0,1) noise for the interval <c>sigma → sigmaNext</c>. The caller owns it.</summary>
    /// <param name="stepIndex">Schedule index of the step drawing the noise.</param>
    /// <param name="subDraw">Ordinal of this draw within the step, for samplers that draw more than once.</param>
    Tensor Sample(int stepIndex, int subDraw, float sigma, float sigmaNext);
}
