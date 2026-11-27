using System.Globalization;
using HartsyInference.Core.Numerics;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>The few-step flow-matching schedule of a self-forcing distilled student: an explicit list of training
/// timesteps ending at 0, such as <c>1000, 960, 889, 727, 0</c>. The timesteps are taken verbatim as sigmas
/// (<c>timestep / 1000</c>): the list already carries the teacher's shift, so none is applied here. Each chunk of an
/// autoregressive rollout runs the whole list, and a frame marked clean (a conditioning frame) runs at timestep 0 and is
/// left untouched.</summary>
/// <remarks>The arithmetic follows diffusers <c>FlowMatchEulerDiscreteScheduler.step</c> with per-token timesteps:
/// float32 sigmas, <c>x ← x + (σ − σ_next)·(−v)</c> per frame, in float32.</remarks>
public sealed class SelfForcingFlowSchedule
{
    /// <summary>Training timesteps per unit sigma.</summary>
    public const float TrainTimesteps = 1000f;

    private readonly float[] _sigmas;
    private readonly float[] _timesteps;

    /// <summary>Builds the schedule from descending <paramref name="timesteps"/> that end at 0.</summary>
    /// <exception cref="ArgumentException">Fewer than two entries, a last entry other than 0, an entry outside
    /// <c>[0, 1000]</c>, or entries that do not strictly descend.</exception>
    public SelfForcingFlowSchedule(ReadOnlySpan<int> timesteps)
    {
        if (timesteps.Length < 2 || timesteps[^1] != 0)
            throw new ArgumentException("A self-forcing schedule lists at least two timesteps and ends at 0.", nameof(timesteps));
        _sigmas = new float[timesteps.Length];
        _timesteps = new float[timesteps.Length];
        for (int i = 0; i < timesteps.Length; i++)
        {
            if (timesteps[i] < 0 || timesteps[i] > (int)TrainTimesteps || (i > 0 && timesteps[i] >= timesteps[i - 1]))
                throw new ArgumentException($"Timesteps must descend strictly within [0, {TrainTimesteps:0}]; got {timesteps[i]} at {i}.",
                    nameof(timesteps));
            // diffusers stores float32 sigmas and derives the timesteps from them.
            _sigmas[i] = (float)(timesteps[i] / 1000.0);
            _timesteps[i] = _sigmas[i] * TrainTimesteps;
        }
    }

    /// <summary>Denoising steps per chunk: one per timestep before the terminal 0.</summary>
    public int Steps => _sigmas.Length - 1;

    /// <summary>The timestep the model sees at <paramref name="step"/>.</summary>
    public float Timestep(int step) => _timesteps[CheckStep(step)];

    /// <summary>Sigma at <paramref name="step"/>; <see cref="Steps"/> names the terminal 0.</summary>
    public float Sigma(int step)
    {
        if (step < 0 || step > Steps)
            throw new ArgumentOutOfRangeException(nameof(step), step, $"The schedule has {Steps} steps.");
        return _sigmas[step];
    }

    /// <summary>Parses a comma-separated timestep list such as <c>1000,960,889,727,0</c>.</summary>
    /// <exception cref="FormatException">An entry is not an integer.</exception>
    public static SelfForcingFlowSchedule Parse(string list)
    {
        ArgumentNullException.ThrowIfNull(list);
        string[] parts = list.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        int[] timesteps = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!int.TryParse(parts[i], NumberStyles.None, CultureInfo.InvariantCulture, out timesteps[i]))
                throw new FormatException($"'{parts[i]}' is not a timestep; expected integers such as 1000,960,889,727,0.");
        }

        return new SelfForcingFlowSchedule(timesteps);
    }

    /// <summary>Writes the per-frame timesteps of <paramref name="step"/>: the step's timestep for a frame being denoised, 0 for
    /// a clean frame.</summary>
    public void FrameTimesteps(int step, ReadOnlySpan<bool> cleanFrames, Span<float> destination)
    {
        float timestep = Timestep(step);
        if (destination.Length != cleanFrames.Length)
            throw new ArgumentException("One timestep per frame is written.", nameof(destination));
        for (int f = 0; f < cleanFrames.Length; f++)
            destination[f] = cleanFrames[f] ? 0f : timestep;
    }

    /// <summary>One Euler step on a chunk, in place: every frame being denoised moves by <c>(σ − σ_next)·(−v)</c>; clean frames
    /// keep their values.</summary>
    /// <param name="latents">Host <c>[C, T, P]</c> values, channel-major, <c>P</c> positions per frame.</param>
    /// <param name="velocity">The predicted flow velocity, shaped like <paramref name="latents"/>.</param>
    /// <param name="cleanFrames">One flag per frame; a set flag leaves the frame untouched.</param>
    /// <param name="roundToBf16">Rounds each updated value to BF16, as a sampler whose latents are stored in BF16 does.</param>
    public void Step(Span<float> latents, ReadOnlySpan<float> velocity, int step, int channels, ReadOnlySpan<bool> cleanFrames,
        bool roundToBf16)
    {
        CheckStep(step);
        int frames = cleanFrames.Length;
        if (channels < 1 || frames < 1 || latents.Length % (channels * frames) != 0 || velocity.Length != latents.Length)
            throw new ArgumentException("Latents and velocity must both be [channels, frames, positions].", nameof(latents));
        int plane = latents.Length / (channels * frames);
        float dt = _sigmas[step] - _sigmas[step + 1];
        for (int c = 0; c < channels; c++)
        {
            for (int f = 0; f < frames; f++)
            {
                if (cleanFrames[f])
                    continue;
                int start = (c * frames + f) * plane;
                for (int i = start; i < start + plane; i++)
                {
                    float next = latents[i] + dt * -velocity[i];
                    latents[i] = roundToBf16 ? BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(next)) : next;
                }
            }
        }
    }

    private int CheckStep(int step)
    {
        if (step < 0 || step >= Steps)
            throw new ArgumentOutOfRangeException(nameof(step), step, $"The schedule has {Steps} steps.");
        return step;
    }
}
