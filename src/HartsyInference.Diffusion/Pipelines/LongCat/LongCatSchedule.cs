namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>The LongCat-Video flow-match sigma schedule, as the pinned reference builds it: the base schedule (a linear ramp, or the
/// distilled indices), the static shift <c>s·σ / (1 + (s − 1)·σ)</c>, and for continuation the <c>enhance_hf</c> tail. Sigmas are returned
/// with the terminal 0 appended, timesteps are <c>σ·1000</c> per denoising step.</summary>
public sealed record LongCatSchedule(float[] Sigmas, float[] Timesteps)
{
    /// <summary>Number of denoising steps.</summary>
    public int Steps => Timesteps.Length;

    /// <summary>The base schedule with <paramref name="steps"/> steps, then shifted by <paramref name="shift"/>.</summary>
    /// <param name="distill">True for the step-distilled schedule (the 50 distilled indices, subsampled to <paramref name="steps"/>).</param>
    public static LongCatSchedule Build(int steps, bool distill, float shift)
    {
        if (steps < 1)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "At least one denoising step.");
        if (!float.IsFinite(shift) || shift <= 0f)
            throw new ArgumentOutOfRangeException(nameof(shift), shift, "The shift is a positive finite number.");

        float[] sigmas = distill ? DistilledSigmas(steps) : LinearSigmas(steps);
        for (int i = 0; i < sigmas.Length; i++)
            sigmas[i] = shift * sigmas[i] / (1f + (shift - 1f) * sigmas[i]);
        return FromSteps(sigmas);
    }

    /// <summary>The <c>enhance_hf</c> schedule of continuation: the steps above 500 kept, then ten uniform steps from 500 to 50.</summary>
    public LongCatSchedule EnhanceHf()
    {
        List<float> kept = [];
        foreach (float t in Timesteps)
            if (t > 500f)
                kept.Add(t);
        for (int i = 0; i < 10; i++)
            kept.Add(500f - i * 50f);   // numpy linspace(500, 0, 10, endpoint=False): 500, 450, …, 50
        float[] timesteps = kept.ToArray();
        float[] sigmas = new float[timesteps.Length + 1];
        for (int i = 0; i < timesteps.Length; i++)
            sigmas[i] = timesteps[i] / 1000f;
        return new LongCatSchedule(sigmas, timesteps);
    }

    /// <summary>Sigmas for the denoising steps, then the terminal 0. Mirrors <c>FlowMatchEulerDiscreteScheduler.set_timesteps</c> with
    /// <c>sigmas=</c>, where the timesteps are <c>σ·1000</c> and the terminal is appended.</summary>
    private static LongCatSchedule FromSteps(float[] shifted)
    {
        float[] timesteps = new float[shifted.Length];
        float[] sigmas = new float[shifted.Length + 1];
        for (int i = 0; i < shifted.Length; i++)
        {
            timesteps[i] = shifted[i] * 1000f;
            sigmas[i] = shifted[i];
        }
        sigmas[^1] = 0f;
        return new LongCatSchedule(sigmas, timesteps);
    }

    /// <summary>torch.linspace(1, 0.001, steps) with torch's symmetric evaluation (the second half counts down from the end).</summary>
    private static float[] LinearSigmas(int steps)
    {
        float[] values = new float[steps];
        const float start = 1f, end = 0.001f;
        float step = steps > 1 ? (end - start) / (steps - 1) : 0f;
        int half = steps / 2;
        for (int i = 0; i < steps; i++)
            values[i] = i < half ? start + step * i : end - step * (steps - i - 1);
        return values;
    }

    /// <summary>The distilled schedule: indices 20, 40, …, 1000 (flipped, then subsampled at <c>floor(linspace(0, 50, steps, endpoint=False))</c>),
    /// divided by 1000.</summary>
    private static float[] DistilledSigmas(int steps)
    {
        float[] values = new float[steps];
        for (int i = 0; i < steps; i++)
        {
            int index = (int)Math.Floor(i * (50.0 / steps));   // numpy linspace(0, 50, steps, endpoint=False)
            int distilled = 50 - index;                         // flipped distill_indices[index]: entry 50-1-index holds (50-index)*20
            values[i] = distilled * 20f / 1000f;
        }
        return values;
    }
}
