namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>The sigma lists of the LongCat-Video-Avatar pipeline (<c>get_timesteps_sigmas</c>, <c>pipeline_longcat_video_avatar.py</c> @ 58eb336).
/// These are the sampler's own sigmas, without the terminal zero the base schedule appends.</summary>
public static class LongCatAvatarSchedule
{
    /// <summary>Which of the reference's three sigma paths to build.</summary>
    public enum Kind
    {
        /// <summary><c>torch.linspace(1, 0.001, steps)</c>: the 50-step CFG path (the avatar pipeline's default, shift applied by the sampler).</summary>
        Linear,

        /// <summary>The 50-sample distilled path (avatar 1.0 and 1.6): distilled sigmas at multiples of 20, sampled at <c>steps</c> points.</summary>
        Distill50,

        /// <summary>The 8-sample distilled path (avatar 1.5): exactly eight sigmas, always used with eight steps.</summary>
        Distill8,
    }

    /// <summary>Timesteps of the reference's sample space.</summary>
    public const int NumTimesteps = 1000;

    /// <summary>Builds the sigmas. <see cref="Kind.Distill8"/> accepts only <paramref name="steps"/> = 8, since the reference's schedule has eight
    /// entries and the demo always uses eight; a different count is refused instead of being silently ignored.</summary>
    public static float[] Sigmas(int steps, Kind kind)
    {
        if (steps < 1)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "The step count must be at least 1.");
        return kind switch
        {
            Kind.Linear => Linspace(1f, 0.001f, steps),
            Kind.Distill50 => Distill50(steps),
            Kind.Distill8 => steps == 8
                ? Distill8()
                : throw new ArgumentOutOfRangeException(nameof(steps), steps, "The 8-sample distilled path runs with exactly 8 steps."),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null),
        };
    }

    /// <summary>Matches <c>torch.linspace</c>: the first half counts up from <paramref name="start"/>, the second half down from <paramref name="end"/>.</summary>
    private static float[] Linspace(float start, float end, int count)
    {
        float[] values = new float[count];
        if (count == 1)
        {
            values[0] = start;
            return values;
        }
        float step = (end - start) / (count - 1);
        int half = count / 2;
        for (int i = 0; i < count; i++)
            values[i] = i < half ? start + step * i : end - step * (count - i - 1);
        return values;
    }

    /// <summary>The reference: <c>distill_indices = (1..50)·20</c>, sigmas are <c>flip(distill_indices)[floor(linspace(0, 50, steps, endpoint=False))] / 1000</c>.</summary>
    private static float[] Distill50(int steps)
    {
        const int samples = 50, stride = NumTimesteps / samples;
        float[] sigmas = new float[steps];
        for (int i = 0; i < steps; i++)
        {
            // np.linspace(0, samples, steps, endpoint=False)[i] = i · samples / steps; floored.
            int index = (int)Math.Floor((double)i * samples / steps);
            // flip(distill_indices)[index] = (samples − index) · stride.
            sigmas[i] = (float)((samples - index) * stride) / NumTimesteps;
        }
        return sigmas;
    }

    /// <summary>The reference (avatar 1.5): <c>distill_indices = NumTimesteps − (1..8)·125</c>, then the reversed selection of the flipped
    /// <c>linspace(0, 1, 1000)</c>: sigma = (999 − d)/999 for d in [875, 750, …, 0], returned in descending order.</summary>
    private static float[] Distill8()
    {
        const int samples = 8, stride = NumTimesteps / samples;
        float[] sigmas = new float[samples];
        for (int j = 0; j < samples; j++)
        {
            // Descending order: j = 0 is the largest sigma.
            int d = NumTimesteps - (samples - j) * stride;
            sigmas[j] = (float)(NumTimesteps - 1 - d) / (NumTimesteps - 1);
        }
        return sigmas;
    }
}
