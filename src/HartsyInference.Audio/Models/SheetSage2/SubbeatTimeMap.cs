namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>What one window's decoded timestamps say about where a subbeat falls in that window.
///
/// <para>Only some events carry a timestamp, so the rest are placed by interpolating between the ones that do,
/// and extrapolating at a fixed tempo past the outermost of them. The tempo is the median of what the anchors
/// themselves say, which survives a single wild timestamp where a mean would not.</para></summary>
public sealed class SubbeatTimeMap
{
    /// <summary>Seconds one subbeat lasts when the window says nothing usable: four subbeats to the beat at 120
    /// BPM. The released implementation states it outright in every fallback and nothing in the vocabulary
    /// implies it, so it is reproduced as the constant it is.</summary>
    public const double DefaultSubbeatSeconds = 0.125;

    private readonly double[] _steps;
    private readonly double[] _times;
    private readonly double _targetSeconds;

    /// <summary>Reads the anchors out of one window's decoded events.</summary>
    /// <param name="events">That window's events, in decode order; the ones carrying a timestamp anchor the map.</param>
    /// <param name="targetSeconds">Longest time the map will report. The released pipeline passes a whole window
    /// even for the short last one, so a map may well name a time past the end of the clip — what keeps events
    /// inside it is <see cref="WindowStitcher.WindowEvents"/>, not this.</param>
    public SubbeatTimeMap(IReadOnlyList<ScoreEvent> events, double targetSeconds = SlidingWindowPlan.WindowSeconds)
    {
        ArgumentNullException.ThrowIfNull(events);
        _targetSeconds = targetSeconds;
        // A repeated position keeps its LAST timestamp, as the reference's dict comprehension does.
        Dictionary<int, double> anchors = [];
        foreach (ScoreEvent item in events)
        {
            if (item.Timestamp is double seconds) anchors[item.Subbeat] = seconds;
        }
        int[] positions = [.. anchors.Keys];
        Array.Sort(positions);
        _steps = new double[positions.Length];
        _times = new double[positions.Length];
        for (int i = 0; i < positions.Length; i++)
        {
            _steps[i] = positions[i];
            _times[i] = anchors[positions[i]];
        }
        SubbeatSeconds = Tempo();
    }

    /// <summary>Seconds one subbeat lasts, used to extrapolate past the outermost anchor.</summary>
    public double SubbeatSeconds { get; }

    /// <summary>How many timestamps anchor the map.</summary>
    public int AnchorCount => _steps.Length;

    /// <summary>Where a position falls in the window, in seconds.</summary>
    /// <param name="subbeat">Position, counted from the window's own start.</param>
    public double Seconds(double subbeat)
    {
        if (_steps.Length == 0)
        {
            // Not one timestamp in the whole window: the reference reads it at a fixed tempo rather than refusing it.
            return Math.Min(_targetSeconds, Math.Max(0.0, subbeat * DefaultSubbeatSeconds));
        }
        if (subbeat <= _steps[0]) return Clip(_times[0] + (subbeat - _steps[0]) * SubbeatSeconds);
        if (subbeat >= _steps[^1]) return Clip(_times[^1] + (subbeat - _steps[^1]) * SubbeatSeconds);
        int index = Array.BinarySearch(_steps, subbeat);
        if (index < 0) index = ~index - 1;
        double slope = (_times[index + 1] - _times[index]) / (_steps[index + 1] - _steps[index]);
        // Interpolation is deliberately not clipped, matching the reference: only the two extrapolations are.
        return slope * (subbeat - _steps[index]) + _times[index];
    }

    private double Clip(double seconds) => Math.Min(Math.Max(seconds, 0.0), _targetSeconds);

    /// <summary>The median of what consecutive anchors say a subbeat lasts.</summary>
    private double Tempo()
    {
        if (_steps.Length < 2) return DefaultSubbeatSeconds;
        double[] rates = new double[_steps.Length - 1];
        for (int i = 0; i < rates.Length; i++)
        {
            // The reference floors the gap at one subbeat; positions are unique and sorted, so it never bites.
            rates[i] = (_times[i + 1] - _times[i]) / Math.Max(_steps[i + 1] - _steps[i], 1.0);
        }
        Array.Sort(rates);
        int middle = rates.Length / 2;
        double median = (rates.Length & 1) == 1 ? rates[middle] : (rates[middle - 1] + rates[middle]) / 2.0;
        // Timestamps that do not advance with position give a flat or backwards tempo, which places nothing.
        return double.IsFinite(median) && median > 0 ? median : DefaultSubbeatSeconds;
    }
}
