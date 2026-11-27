using HartsyInference.Core.Numerics;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>The few-step flow-matching schedule of a distilled refiner: a clean latent is re-noised to the first sigma and
/// walked back to 0 over an explicit sigma list, such as LTX-2's stage-2 <c>0.909375, 0.725, 0.421875, 0</c>. Each step
/// forms the model's clean estimate <c>x0 = x − σ·v</c> and moves the latent to the line between itself and that estimate
/// at the next sigma.</summary>
/// <remarks>The arithmetic follows the LTX-2 distilled sampler as SANA-WM's refiner runs it: float32 throughout, the sigma
/// ratio taken in double from the float32 sigmas, and, with BF16 rounding, the estimate and the stepped latent each rounded
/// once, as a sampler whose tensors are stored in BF16 does.</remarks>
public sealed class FlowRefineSchedule
{
    private readonly float[] _sigmas;

    /// <summary>Builds the schedule from descending <paramref name="sigmas"/> that start in <c>(0, 1]</c> and end at 0.</summary>
    /// <exception cref="ArgumentException">Fewer than two entries, a last entry other than 0, a first entry outside
    /// <c>(0, 1]</c>, or entries that do not strictly descend.</exception>
    public FlowRefineSchedule(ReadOnlySpan<float> sigmas)
    {
        if (sigmas.Length < 2 || sigmas[^1] != 0f)
            throw new ArgumentException("A refine schedule lists at least two sigmas and ends at 0.", nameof(sigmas));
        if (!(sigmas[0] > 0f && sigmas[0] <= 1f))
            throw new ArgumentException($"A refine schedule starts within (0, 1]; got {sigmas[0]}.", nameof(sigmas));
        for (int i = 1; i < sigmas.Length; i++)
        {
            if (!(sigmas[i] < sigmas[i - 1]) || sigmas[i] < 0f)
                throw new ArgumentException($"Sigmas must descend strictly to 0; got {sigmas[i]} at {i}.", nameof(sigmas));
        }

        _sigmas = sigmas.ToArray();
    }

    /// <summary>LTX-2's three-step stage-2 schedule (<c>STAGE_2_DISTILLED_SIGMA_VALUES</c>). A new array per call.</summary>
    public static float[] Ltx2Stage2Sigmas => [0.909375f, 0.725f, 0.421875f, 0f];

    /// <summary>Denoising steps: one per sigma before the terminal 0.</summary>
    public int Steps => _sigmas.Length - 1;

    /// <summary>Sigma at <paramref name="step"/>; <see cref="Steps"/> names the terminal 0.</summary>
    public float Sigma(int step)
    {
        if (step < 0 || step > Steps)
            throw new ArgumentOutOfRangeException(nameof(step), step, $"The schedule has {Steps} steps.");
        return _sigmas[step];
    }

    /// <summary>The timestep the model sees at <paramref name="step"/>: the sigma times <paramref name="scale"/>, in float32.</summary>
    public float Timestep(int step, float scale = 1000f) => _sigmas[CheckStep(step)] * scale;

    /// <summary>The schedule's starting point: <c>(1 − σ₀)·clean + σ₀·noise</c> written to <paramref name="destination"/>.</summary>
    public void Renoise(Span<float> destination, ReadOnlySpan<float> clean, ReadOnlySpan<float> noise, bool roundToBf16)
    {
        if (clean.Length != destination.Length || noise.Length != destination.Length)
            throw new ArgumentException("The clean latent, the noise and the destination hold the same values.", nameof(destination));
        float sigma = _sigmas[0];
        float keep = (float)(1.0 - sigma);
        for (int i = 0; i < destination.Length; i++)
        {
            float cleanPart = keep * clean[i];
            float noisePart = sigma * noise[i];
            float value = cleanPart + noisePart;
            destination[i] = roundToBf16 ? Bf16(value) : value;
        }
    }

    /// <summary>One step in place: <paramref name="latents"/> at this step's sigma moves to the next sigma along the line to
    /// the clean estimate <c>x − σ·v</c>. The last step lands on the estimate itself.</summary>
    /// <param name="velocity">The predicted flow velocity, one value per latent value.</param>
    public void Step(Span<float> latents, ReadOnlySpan<float> velocity, int step, bool roundToBf16)
    {
        CheckStep(step);
        if (velocity.Length != latents.Length)
            throw new ArgumentException("One velocity value per latent value.", nameof(velocity));
        float sigma = _sigmas[step];
        double ratioExact = (double)_sigmas[step + 1] / sigma;
        float ratio = (float)ratioExact;
        float toward = (float)(1.0 - ratioExact);
        for (int i = 0; i < latents.Length; i++)
        {
            float scaled = velocity[i] * sigma;
            float estimate = latents[i] - scaled;
            if (roundToBf16)
                estimate = Bf16(estimate);
            float held = ratio * latents[i];
            float moved = toward * estimate;
            float next = held + moved;
            latents[i] = roundToBf16 ? Bf16(next) : next;
        }
    }

    private static float Bf16(float value) => BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(value));

    private int CheckStep(int step)
    {
        if (step < 0 || step >= Steps)
            throw new ArgumentOutOfRangeException(nameof(step), step, $"The schedule has {Steps} steps.");
        return step;
    }
}
