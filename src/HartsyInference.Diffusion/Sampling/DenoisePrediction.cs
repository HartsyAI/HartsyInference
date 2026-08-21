using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>One model evaluation's raw output, kept as the conditional/unconditional PAIR rather than pre-combined.
///
/// <para>The pair is load-bearing, not a convenience. <see cref="Core.Backends.IBackend.CfgEulerStep"/> fuses the CFG
/// combine and the Euler update into a single in-place device op, and every pipeline in the engine is built around it —
/// keeping the latent GPU-resident with zero host round-trips per step. Handing samplers a pre-combined tensor would
/// force that fusion apart and cost every default generation a separate combine pass, so <see cref="EulerSampler"/>
/// takes the pair and hands it straight to the fused op unchanged.</para>
///
/// <para><see cref="Uncond"/> may be reference-equal to <see cref="Cond"/> when guidance is off; the fused op documents
/// that as allowed. <see cref="Dispose"/> handles the aliasing.</para></summary>
public readonly struct DenoisePrediction : IDisposable
{
    /// <summary>The conditional (positive-prompt) model output.</summary>
    public Tensor Cond { get; }

    /// <summary>The unconditional (negative-prompt) model output, or the same instance as <see cref="Cond"/> when the
    /// caller is running guidance-free.</summary>
    public Tensor Uncond { get; }

    /// <summary>Creates a prediction pair. Pass the same tensor twice for a guidance-free step.</summary>
    public DenoisePrediction(Tensor cond, Tensor uncond)
    {
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(uncond);
        Cond = cond;
        Uncond = uncond;
    }

    /// <summary>Releases both tensors, tolerating the guidance-free aliasing where they are one instance.</summary>
    public void Dispose()
    {
        Cond.Dispose();
        if (!ReferenceEquals(Cond, Uncond))
        {
            Uncond.Dispose();
        }
    }
}
