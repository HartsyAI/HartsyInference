namespace HartsyInference.LLM.Generation.Speculative;

/// <summary>Steps per second of the engine at each batch size, as a cost table: entry <c>b - 1</c> is <c>SPS(b)</c>. The paper profiles it once at engine
/// initialization. This type holds the table; measuring it is left to the deployment that will serve the model.</summary>
internal sealed class SpsProfile
{
    private readonly double[] _stepsPerSecond;

    /// <param name="stepsPerSecond">SPS at batch sizes 1, 2 and so on. Every entry must be finite and positive.</param>
    public SpsProfile(IReadOnlyList<double> stepsPerSecond)
    {
        ArgumentNullException.ThrowIfNull(stepsPerSecond);
        if (stepsPerSecond.Count == 0) throw new ArgumentException("A profile needs at least batch size 1.", nameof(stepsPerSecond));
        _stepsPerSecond = new double[stepsPerSecond.Count];
        for (int b = 0; b < _stepsPerSecond.Length; b++)
        {
            double sps = stepsPerSecond[b];
            if (!double.IsFinite(sps) || sps <= 0) throw new ArgumentException($"SPS at batch size {b + 1} must be finite and positive.", nameof(stepsPerSecond));
            _stepsPerSecond[b] = sps;
        }
    }

    /// <summary>The largest batch size the table covers.</summary>
    public int MaxBatch => _stepsPerSecond.Length;

    /// <summary>Steps per second at <paramref name="batch"/> tokens.</summary>
    public double StepsPerSecond(int batch)
    {
        if ((uint)(batch - 1) >= (uint)_stepsPerSecond.Length)
            throw new ArgumentOutOfRangeException(nameof(batch), batch, $"The profile covers batch sizes 1 to {MaxBatch}.");
        return _stepsPerSecond[batch - 1];
    }
}
