using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Discrete few-step flow-matching scheduler for DMD/Self-Forcing-distilled students (Matrix-Game 2.0's
/// <c>denoising_step_list</c> = [1000, 666, 333] @ 3 steps). Each entry is an integer index into the canonical
/// 1000-step training schedule; with <c>warp_denoising_step</c> the index is remapped through the shifted sigma grid
/// (<c>σ = shift·t/(1 + (shift−1)·t)</c>, shift 5.0) exactly as the student was trained. Per step the model's velocity
/// becomes an x₀ estimate (<c>x₀ = x_t − σ·v</c>) and, for non-final steps, the latent is re-noised to the next
/// level with <b>fresh</b> noise (<c>x_next = (1−σ_next)·x₀ + σ_next·ε</c>) — the distilled student expects this
/// re-noise, not an ODE step. Reusable for any DMD-distilled flow model (Wan distills, future Lightning ports).</summary>
public sealed unsafe class FlowMatchDmdScheduler
{
    private readonly int[] _stepList;
    private readonly float[] _sigmas;

    /// <summary>Creates the scheduler for a fixed distilled step list (descending noise indices in [1, numTrainTimesteps]).</summary>
    public FlowMatchDmdScheduler(int[] denoisingStepList, float timestepShift = 5.0f, bool warpDenoisingStep = true,
        int numTrainTimesteps = 1000)
    {
        if (denoisingStepList.Length == 0) throw new ArgumentException("step list must not be empty.", nameof(denoisingStepList));
        for (int i = 1; i < denoisingStepList.Length; i++)
            if (denoisingStepList[i] >= denoisingStepList[i - 1])
                throw new ArgumentException("step list must be strictly descending.", nameof(denoisingStepList));
        _stepList = (int[])denoisingStepList.Clone();
        _sigmas = new float[denoisingStepList.Length];
        for (int i = 0; i < denoisingStepList.Length; i++)
        {
            float t = (float)denoisingStepList[i] / numTrainTimesteps;
            _sigmas[i] = warpDenoisingStep ? timestepShift * t / (1f + (timestepShift - 1f) * t) : t;
        }
    }

    /// <summary>Length of the distilled <c>denoisingStepList</c> this instance was built with.</summary>
    public int NumSteps => _stepList.Length;

    /// <summary>The model-conditioning timestep for step <paramref name="stepIndex"/> (<c>σ · 1000</c> on the warped grid).</summary>
    public float Timestep(int stepIndex) => _sigmas[stepIndex] * 1000f;

    /// <summary>Sigma at step <paramref name="stepIndex"/>.</summary>
    public float Sigma(int stepIndex) => _sigmas[stepIndex];

    /// <summary>Recovers the clean estimate in place: <c>x₀ = x_t − σ·v</c>.</summary>
    public void ToX0(Tensor sample, Tensor velocity, int stepIndex)
    {
        float sigma = _sigmas[stepIndex];
        float* sp = (float*)sample.DataPointer;
        float* vp = (float*)velocity.DataPointer;
        long n = sample.Shape.ElementCount;
        for (long i = 0; i < n; i++) sp[i] -= sigma * vp[i];
    }

    /// <summary>Re-noises the clean estimate to the NEXT step's level with fresh noise, in place:
    /// <c>x = (1−σ_next)·x₀ + σ_next·ε</c>. Call only for non-final steps.</summary>
    public void AddNoise(Tensor x0, Tensor freshNoise, int nextStepIndex)
    {
        float sigma = _sigmas[nextStepIndex];
        float* xp = (float*)x0.DataPointer;
        float* np = (float*)freshNoise.DataPointer;
        long n = x0.Shape.ElementCount;
        for (long i = 0; i < n; i++) xp[i] = (1f - sigma) * xp[i] + sigma * np[i];
    }

    /// <summary>Matrix-Game 2.0 universal / GTA preset (3 steps).</summary>
    public static FlowMatchDmdScheduler MatrixGame2ThreeStep() => new([1000, 666, 333]);

    /// <summary>Matrix-Game 2.0 TempleRun preset (4 steps).</summary>
    public static FlowMatchDmdScheduler MatrixGame2FourStep() => new([1000, 750, 500, 250]);
}
