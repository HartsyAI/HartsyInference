namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Context carried into the next Drama 3 chunk (fish-speech <c>Drama3History</c>): the first chunk stays pinned,
/// plus the latest <see cref="RecentTurns"/> chunks, so the voice stays consistent across a long text.</summary>
/// <param name="recentTurns">Chunks kept besides the pinned first one.</param>
public sealed class FishAudioDrama3History(int recentTurns = FishAudioDrama3History.DefaultRecentTurns)
{
    /// <summary>Recent chunks kept besides the pinned one, as the hosted server does.</summary>
    public const int DefaultRecentTurns = 1;

    private readonly Queue<FishAudioDrama3Turn> _recent = new();

    /// <summary>Recent chunks kept besides the pinned one.</summary>
    public int RecentTurns { get; } = recentTurns >= 0 ? recentTurns : throw new ArgumentOutOfRangeException(nameof(recentTurns));

    /// <summary>The first chunk pushed, or null before any.</summary>
    public FishAudioDrama3Turn? Pinned { get; private set; }

    /// <summary>Records a generated chunk: the first one is pinned, later ones replace the oldest recent turn.</summary>
    public void Push(FishAudioDrama3Turn turn)
    {
        ArgumentNullException.ThrowIfNull(turn);
        if (Pinned is null)
        {
            Pinned = turn;
            return;
        }
        if (RecentTurns == 0)
        {
            return;
        }
        if (_recent.Count == RecentTurns)
        {
            _recent.Dequeue();
        }
        _recent.Enqueue(turn);
    }

    /// <summary>Histories to try, longest first: the oldest recent turns are dropped one by one, then the pinned one,
    /// ending with no history.</summary>
    public IEnumerable<IReadOnlyList<FishAudioDrama3Turn>> Candidates()
    {
        FishAudioDrama3Turn[] recent = [.. _recent];
        for (int start = 0; start <= recent.Length; start++)
        {
            List<FishAudioDrama3Turn> turns = new(recent.Length - start + 1);
            if (Pinned is not null)
            {
                turns.Add(Pinned);
            }
            turns.AddRange(recent.AsSpan(start).ToArray());
            yield return turns;
        }
        if (Pinned is not null)
        {
            yield return [];
        }
    }
}
