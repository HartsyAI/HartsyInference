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

    /// <summary>CFG scale this pair should be combined at: <c>v = w·cond + (1−w)·uncond</c>.
    ///
    /// <para>It lives on the PAIR rather than on the predictor because guidance is not always constant across a run.
    /// Qwen-Image's limited-interval CFG drops the unconditional branch outside a sigma band — those steps must combine
    /// at exactly 1.0, and reusing the run's nominal scale would not be bit-identical even though <c>cond == uncond</c>
    /// makes it mathematically irrelevant (<c>g·c + (1−g)·c</c> does not cancel exactly in F32). Ideogram 4 carries a
    /// full per-step guidance schedule for the same reason.</para></summary>
    public float Guidance { get; }

    /// <summary>Whether disposing this pair releases the tensors. False when the predictor handed back buffers it does
    /// not own — Chroma's <c>ForwardPaired</c> returns the transformer's FIXED step-graph buffers under
    /// <c>callerOwns: false</c>, and disposing those would free memory the captured graph still points at.</summary>
    public bool OwnsTensors { get; }

    /// <summary>Creates a prediction pair. Pass the same tensor twice for a guidance-free step.</summary>
    /// <param name="guidance">CFG scale for this step; 1.0 when the pair is already combined or guidance-free.</param>
    /// <param name="ownsTensors">False to borrow rather than own — see <see cref="OwnsTensors"/>.</param>
    public DenoisePrediction(Tensor cond, Tensor uncond, float guidance = 1.0f, bool ownsTensors = true)
    {
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(uncond);
        Cond = cond;
        Uncond = uncond;
        Guidance = guidance;
        OwnsTensors = ownsTensors;
    }

    /// <summary>Releases both tensors when this pair owns them, tolerating the guidance-free aliasing where they are
    /// one instance.</summary>
    public void Dispose()
    {
        if (!OwnsTensors)
        {
            return;
        }
        Cond.Dispose();
        if (!ReferenceEquals(Cond, Uncond))
        {
            Uncond.Dispose();
        }
    }
}
