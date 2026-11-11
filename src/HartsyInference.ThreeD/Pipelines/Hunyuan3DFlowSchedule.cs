using HartsyInference.ThreeD.Models.Hunyuan3D;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>The Hunyuan3D-2 flow-match sigma schedules and the timestep the DiT sees at each step.</summary>
internal static class Hunyuan3DFlowSchedule
{
    private const int TrainTimesteps = 1000;

    /// <summary>Sigmas for <paramref name="steps"/> evaluations plus the trailing 1.0 the Euler step reads, for the checkpoint's scheduler.</summary>
    internal static float[] Sigmas(int steps, Hunyuan3DScheduler scheduler = Hunyuan3DScheduler.FlowMatchEuler, int pcmTimesteps = 0) =>
        scheduler == Hunyuan3DScheduler.ConsistencyFlowMatchEuler ? ConsistencySigmas(steps, pcmTimesteps) : Sigmas(steps);

    /// <summary>Evenly spaced sigmas from 0 to 1 and a trailing 1.0.</summary>
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

    /// <summary>Sigmas picked from the <paramref name="pcmTimesteps"/>-entry table of <c>ConsistencyFlowMatchEulerDiscreteScheduler</c> (index <c>floor(linspace(0, pcm, steps, endpoint=False))</c>), plus the trailing 1.0.</summary>
    internal static float[] ConsistencySigmas(int steps, int pcmTimesteps)
    {
        if (steps <= 0 || pcmTimesteps <= 0 || pcmTimesteps > TrainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, $"Steps {steps} and pcm_timesteps {pcmTimesteps} must be positive, pcm at most {TrainTimesteps}.");
        int stepRatio = TrainTimesteps / pcmTimesteps;
        float[] sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            int table = (int)Math.Floor(i * (double)pcmTimesteps / steps);
            int trainIndex = table == 0 ? 0 : (int)Math.Round(table * (double)stepRatio, MidpointRounding.ToEven) - 1;
            sigmas[i] = (float)(trainIndex / (double)(TrainTimesteps - 1));
        }

        sigmas[steps] = 1f;
        return sigmas;
    }

    /// <summary>The value for <see cref="Models.Hunyuan3D.Hunyuan3DDit.Forward"/>: the sigma, since the DiT applies <c>time_factor</c>.</summary>
    internal static float DitTimestep(float sigma) => sigma;
}
