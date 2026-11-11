namespace HartsyInference.ThreeD.Pipelines;

/// <summary>The Hunyuan3D-2 flow-match sigma schedule and the timestep the DiT sees at each step.</summary>
internal static class Hunyuan3DFlowSchedule
{
    /// <summary>Sigmas for <paramref name="steps"/> evaluations plus the trailing 1.0 the Euler step reads.</summary>
    /// <remarks>Ascending from 0, so the last step has zero width.</remarks>
    internal static float[] Sigmas(int steps)
    {
        if (steps <= 0)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "Step count must be positive.");
        float[] sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++)
            sigmas[i] = steps > 1 ? (float)(i / (double)(steps - 1)) : 0f;
        sigmas[steps] = 1f;
        return sigmas;
    }

    /// <summary>The value for <see cref="Models.Hunyuan3D.Hunyuan3DDit.Forward"/>: the sigma, since the DiT applies <c>time_factor</c>.</summary>
    internal static float DitTimestep(float sigma) => sigma;
}
