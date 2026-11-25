using HartsyInference.Core.Layers.Attention;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Whether a block of rows attends to another, in the frame-v5 protocol: the query-key rules of upstream <c>duo/models/realization/modules/masks.py</c> (@068e357), read from <c>visible()</c>.</summary>
public enum DuoAttentionMode
{
    /// <summary>Generation: a row sees the history and the rows of its own block, never a later block.</summary>
    Causal,

    /// <summary>Joint inference over every noisy block; clean answer copies are refused.</summary>
    Bidirectional,
}

/// <summary>One attended rectangle: query rows <c>[QStart, QEnd)</c> may read key rows <c>[KStart, KEnd)</c>.</summary>
public readonly record struct DuoMaskRect(int QStart, int QEnd, int KStart, int KEnd);

/// <summary>Builds the boolean attention mask as sparse rectangles from groups, one rectangle per run of adjacent visible key rows.</summary>
public static class DuoMask
{
    /// <summary>Whether query group <paramref name="query"/> may read key group <paramref name="key"/>.</summary>
    /// <remarks><paramref name="refiner"/> selects the text refiner's own graph: shared rows, and text within one block.</remarks>
    public static bool Visible(DuoGroup query, DuoGroup key, DuoAttentionMode mode, bool refiner)
    {
        if (query.Sample != key.Sample)
        {
            return false;
        }
        if (query.Kind == DuoGroupKind.Padding || key.Kind == DuoGroupKind.Padding)
        {
            return query.Kind == DuoGroupKind.Padding && key.Kind == DuoGroupKind.Padding;
        }
        if (mode == DuoAttentionMode.Bidirectional)
        {
            if (query.Kind == DuoGroupKind.Clean || key.Kind == DuoGroupKind.Clean)
            {
                throw new InvalidOperationException("A bidirectional input contains a clean answer copy.");
            }
            return true;
        }
        if (refiner)
        {
            return key.Kind == DuoGroupKind.Shared
                || (query.Kind == DuoGroupKind.Text && key.Kind == DuoGroupKind.Text && query.Block == key.Block);
        }
        if (query.Kind is DuoGroupKind.Reference or DuoGroupKind.Shared)
        {
            return key.Kind == query.Kind;
        }
        if (key.Kind is DuoGroupKind.Reference or DuoGroupKind.Shared)
        {
            return true;
        }
        return key.Kind switch
        {
            DuoGroupKind.Text => key.Block == query.Block,
            DuoGroupKind.Noisy => query.Kind == DuoGroupKind.Noisy && key.Block == query.Block,
            DuoGroupKind.Clean => key.Block < query.Block || (query.Kind == DuoGroupKind.Clean && key.Block == query.Block),
            _ => false,
        };
    }

    /// <summary>The rectangles for every query group. Keys are scanned in the order given; adjacent visible keys merge into one span.</summary>
    /// <exception cref="InvalidOperationException">A query group sees no key at all.</exception>
    public static IReadOnlyList<DuoMaskRect> Build(IReadOnlyList<DuoGroup> queries, IReadOnlyList<DuoGroup> keys,
        DuoAttentionMode mode, bool refiner = false)
    {
        ArgumentNullException.ThrowIfNull(queries);
        ArgumentNullException.ThrowIfNull(keys);
        List<DuoMaskRect> rects = [];
        foreach (DuoGroup query in queries)
        {
            int spans = 0;
            int spanStart = 0, spanEnd = 0;
            bool open = false;
            foreach (DuoGroup key in keys)
            {
                if (!Visible(query, key, mode, refiner))
                {
                    continue;
                }
                if (open && spanEnd == key.Start)
                {
                    spanEnd = key.End;
                    continue;
                }
                if (open)
                {
                    rects.Add(new DuoMaskRect(query.Start, query.End, spanStart, spanEnd));
                    spans++;
                }
                spanStart = key.Start;
                spanEnd = key.End;
                open = true;
            }
            if (open)
            {
                rects.Add(new DuoMaskRect(query.Start, query.End, spanStart, spanEnd));
                spans++;
            }
            if (spans == 0)
            {
                throw new InvalidOperationException(
                    $"Query group (sample {query.Sample}, block {query.Block}, {query.Kind}) sees no key; the layout is malformed.");
            }
        }
        return rects;
    }

    /// <summary>The rectangles as one attention segment per query group, in row order, each naming the key rows it reads. Attending per segment
    /// replaces the <c>[queryLength, keyLength]</c> bias.</summary>
    /// <exception cref="ArgumentException">The query groups do not tile <c>[0, queryLength)</c>, or a rectangle leaves the keys.</exception>
    public static KeyRangeSegment[] Segments(IReadOnlyList<DuoMaskRect> rects, int queryLength, int keyLength)
    {
        ArgumentNullException.ThrowIfNull(rects);
        List<DuoMaskRect> ordered = [.. rects];
        ordered.Sort(static (a, b) =>
        {
            int byStart = a.QStart.CompareTo(b.QStart);
            if (byStart != 0) return byStart;
            int byEnd = a.QEnd.CompareTo(b.QEnd);
            return byEnd != 0 ? byEnd : a.KStart.CompareTo(b.KStart);
        });
        List<KeyRangeSegment> segments = [];
        for (int i = 0; i < ordered.Count;)
        {
            DuoMaskRect first = ordered[i];
            List<AttentionKeyRange> keys = [];
            for (; i < ordered.Count && ordered[i].QStart == first.QStart && ordered[i].QEnd == first.QEnd; i++)
            {
                AttentionKeyRange range = new(ordered[i].KStart, ordered[i].KEnd - ordered[i].KStart);
                if (keys.Count > 0 && keys[^1].End == range.Start)
                {
                    keys[^1] = new AttentionKeyRange(keys[^1].Start, keys[^1].Count + range.Count);
                }
                else
                {
                    keys.Add(range);
                }
            }
            segments.Add(new KeyRangeSegment(first.QStart, first.QEnd - first.QStart, keys));
        }
        KeyRangeSegment[] plan = [.. segments];
        KeyRangeSegment.Validate(plan, queryLength, keyLength);
        return plan;
    }

    /// <summary>Expands rectangles into a dense boolean matrix (true = attend). Test and CPU reference use only.</summary>
    public static bool[,] Dense(IReadOnlyList<DuoMaskRect> rects, int queryLength, int keyLength)
    {
        bool[,] dense = new bool[queryLength, keyLength];
        foreach (DuoMaskRect rect in rects)
        {
            for (int q = rect.QStart; q < rect.QEnd; q++)
            {
                for (int k = rect.KStart; k < rect.KEnd; k++)
                {
                    dense[q, k] = true;
                }
            }
        }
        return dense;
    }
}
