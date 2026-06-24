using System;
using System.Collections.Generic;
using System.Linq;

namespace HartsyInference.Audio.Streaming;

/// <summary>LocalAgreement-2 stabilizer for pseudo-streaming ASR (the UFAL whisper_streaming policy). Each time
/// the active audio window is re-transcribed, the full hypothesis is fed in via <see cref="Insert"/>; a word is
/// only confirmed once two consecutive hypotheses agree on it. This trades a small latency (a word is held until
/// the next re-decode corroborates it) for stable output that does not rewrite already-emitted text.
///
/// <para>Operates at word granularity (no timestamps) so it needs nothing from the decoder beyond the decoded
/// text. The hypothesis re-decodes the whole window, so it reproduces the already-confirmed words as a prefix;
/// this buffer compares the unconfirmed tail of the previous hypothesis with the current one and commits their
/// longest common prefix.</para></summary>
public sealed class HypothesisBuffer
{
    private readonly List<string> _committed = new();
    private List<string> _pendingTail = new();

    /// <summary>All words confirmed so far in the current window epoch (cleared by <see cref="Reset"/>).</summary>
    public IReadOnlyList<string> Committed => _committed;

    /// <summary>The current unconfirmed tail (the live "partial" the previous hypothesis ended on).</summary>
    public IReadOnlyList<string> PendingTail => _pendingTail;

    /// <summary>Feeds the full current hypothesis (the entire transcription of the active window, as words) and
    /// returns the words newly confirmed by agreement with the previous hypothesis.</summary>
    public IReadOnlyList<string> Insert(IReadOnlyList<string> hypothesis)
    {
        // The hypothesis re-decodes the whole window, so its first _committed.Count words are the already-confirmed
        // ones; take the tail beyond them.
        List<string> tail = hypothesis.Count > _committed.Count
            ? hypothesis.Skip(_committed.Count).ToList()
            : new List<string>();

        int k = CommonPrefixLength(_pendingTail, tail);
        List<string> newlyConfirmed = tail.GetRange(0, k);
        _committed.AddRange(newlyConfirmed);
        _pendingTail = tail.GetRange(k, tail.Count - k);
        return newlyConfirmed;
    }

    /// <summary>Confirms everything still pending (call at end-of-stream or when the window is cut), returning the
    /// words moved from the pending tail to committed.</summary>
    public IReadOnlyList<string> Flush()
    {
        List<string> rest = _pendingTail;
        _committed.AddRange(rest);
        _pendingTail = new List<string>();
        return rest;
    }

    /// <summary>Clears all state for a fresh window epoch.</summary>
    public void Reset()
    {
        _committed.Clear();
        _pendingTail = new List<string>();
    }

    private static int CommonPrefixLength(IReadOnlyList<string> a, IReadOnlyList<string> b)
    {
        int n = Math.Min(a.Count, b.Count);
        int i = 0;
        while (i < n && string.Equals(a[i], b[i], StringComparison.Ordinal))
        {
            i++;
        }
        return i;
    }
}
