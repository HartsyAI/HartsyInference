namespace HartsyInference.Diffusion.Pipelines.MovieRoll;

/// <summary>MovieRoll's flow-match sigma ladder and rolling step-index table. The base chain runs <c>N</c> uniform steps of a shifted sigma
/// ladder; the distilled chain runs a custom ladder with one step per window slot. Both share the rolling table: window slot <c>w</c> starts
/// <c>w · StepsPerWindow</c> steps behind slot 0, so each queued chunk sits at its own point of the ladder.</summary>
public sealed class MovieRollSchedule
{
    private const double TrainTimesteps = 1000.0;

    private readonly int[][] _rows;

    private MovieRollSchedule(double[] sigmas, int numSteps, int windowSize, int stepsPerWindow)
    {
        Sigmas = sigmas.Select(s => (float)s).ToArray();
        Timesteps = sigmas.Take(numSteps).Select(s => (float)(s * TrainTimesteps)).ToArray();
        NumSteps = numSteps;
        WindowSize = windowSize;
        StepsPerWindow = stepsPerWindow;
        _rows = new int[windowSize][];
        for (int w = 0; w < windowSize; w++)
        {
            _rows[w] = new int[numSteps];
            for (int i = 0; i < numSteps; i++)
                _rows[w][i] = Mod(i - w * stepsPerWindow, numSteps);   // torch.roll(arange(N), w·spw)
        }
    }

    /// <summary>Sigmas, length <see cref="NumSteps"/> + 1, the last being 0 (the clean sample).</summary>
    public float[] Sigmas { get; }

    /// <summary>The shifted timesteps, length <see cref="NumSteps"/>; the timestep fed to the DiT for a step is <c>Timesteps[index]</c>.</summary>
    public float[] Timesteps { get; }

    /// <summary>Number of ladder entries (the base chain's step count, or the distilled ladder's length).</summary>
    public int NumSteps { get; }

    /// <summary>Chunks in a rolling window.</summary>
    public int WindowSize { get; }

    /// <summary>Global steps per round; the base chain has <c>NumSteps / WindowSize</c>, the distilled chain 1.</summary>
    public int StepsPerWindow { get; }

    /// <summary>The base chain: <paramref name="numInferenceSteps"/> steps of a sigma ladder linear from 1 to 0 and warped by the flow shift.
    /// The step count must divide evenly into the window.</summary>
    public static MovieRollSchedule Base(int numInferenceSteps, int windowSize, double shift)
    {
        if (numInferenceSteps <= 0 || windowSize <= 0 || numInferenceSteps % windowSize != 0)
            throw new ArgumentException($"The base chain needs a positive step count that a window of {windowSize} divides; got {numInferenceSteps}.");
        double[] sigmas = new double[numInferenceSteps + 1];
        for (int i = 0; i <= numInferenceSteps; i++) sigmas[i] = Shift(1.0 - (double)i / numInferenceSteps, shift);
        return new MovieRollSchedule(sigmas, numInferenceSteps, windowSize, numInferenceSteps / windowSize);
    }

    /// <summary>The distilled chain: the custom ladder <paramref name="denoisingStepList"/> (strictly descending, one entry per window slot). Its
    /// timesteps come from the shifted sigmas, not from the list.</summary>
    public static MovieRollSchedule Distilled(int[] denoisingStepList, int windowSize, double shift)
    {
        ArgumentNullException.ThrowIfNull(denoisingStepList);
        if (denoisingStepList.Length == 0 || denoisingStepList.Length != windowSize)
            throw new ArgumentException($"The distilled ladder needs one step per window slot: {windowSize} slots, {denoisingStepList.Length} steps.");
        for (int i = 0; i + 1 < denoisingStepList.Length; i++)
            if (denoisingStepList[i] <= denoisingStepList[i + 1])
                throw new ArgumentException("The distilled ladder must be strictly descending.", nameof(denoisingStepList));
        double[] sigmas = new double[denoisingStepList.Length + 1];
        for (int i = 0; i < denoisingStepList.Length; i++)
        {
            if (denoisingStepList[i] <= 0 || denoisingStepList[i] > TrainTimesteps)
                throw new ArgumentOutOfRangeException(nameof(denoisingStepList), denoisingStepList[i], "Ladder steps lie in (0, 1000].");
            sigmas[i] = Shift(denoisingStepList[i] / TrainTimesteps, shift);
        }
        sigmas[denoisingStepList.Length] = 0.0;
        return new MovieRollSchedule(sigmas, denoisingStepList.Length, windowSize, 1);
    }

    /// <summary>The ladder index of each window slot at a global step, slot 0 first. Slots are ordered as the queue: the oldest chunk is slot 0.</summary>
    public int[] StepIndices(int globalStep)
    {
        if (globalStep < 0) throw new ArgumentOutOfRangeException(nameof(globalStep));
        int stepIdx = globalStep % NumSteps;
        int lagged = Math.Max(globalStep - NumSteps + 1, 0);
        int offset = (int)(((long)lagged + StepsPerWindow - 1) / StepsPerWindow % WindowSize);
        int[] indices = new int[WindowSize];
        for (int k = 0; k < WindowSize; k++) indices[k] = _rows[(offset + k) % WindowSize][stepIdx];
        return indices;
    }

    private static double Shift(double t, double shift) => shift * t / (1 + (shift - 1) * t);

    private static int Mod(int value, int modulus) => ((value % modulus) + modulus) % modulus;
}
