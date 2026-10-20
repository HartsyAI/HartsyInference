namespace HartsyInference.Tests.Common.Attention;

/// <summary>Operand storage of a case.</summary>
public enum AttnDType { F32, F16, BF16 }

/// <summary>How the operands sit in memory: head-major <c>[B,H,S,D]</c>, or token-major <c>[S, H·D]</c> (B = 1).</summary>
public enum AttnLayout { HeadMajor, TokenMajor }

/// <summary>The mask a case carries. Causal and Window run through <c>FlashAttention</c>, whose window is one-sided.</summary>
public enum AttnMask { None, KeyOnly, QueryKey, PerHead, PerBatch, Causal, Window }

/// <summary>Sequence preset: tiny, a tile tail, 4k rows (sampled), or one decoded row against a long cache.</summary>
public enum AttnSeq { Tiny, TileTail, Long4k, Decode }

/// <summary>The worst precision a backend's legacy entry may compute a case at; it picks the tolerance.</summary>
public enum AttnAchieved { F32, Tf32, F16, Bf16Io }

/// <summary>One attention case. <see cref="Flash"/> cases go through <c>FlashAttention</c> over a cache of
/// <see cref="Skv"/> rows whose first <see cref="KvLen"/> are valid, query row 0 at <see cref="QOffset"/>.</summary>
public sealed record AttnCase(
    string Id, AttnDType DType, AttnLayout Layout, AttnMask Mask, AttnSeq Seq,
    int Batch, int Heads, int KvHeads, int HeadDim, int Sq, int Skv, int KvLen, int QOffset, int Window,
    bool AllowF16, bool Flash, bool Causal, string Origin)
{
    /// <summary>Query heads per key/value head.</summary>
    public int Group => Heads / KvHeads;

    /// <summary>The softmax scale.</summary>
    public float Scale => 1f / MathF.Sqrt(HeadDim);

    /// <summary>FNV-1a of the id: <see cref="string.GetHashCode()"/> is randomized per process.</summary>
    public int Seed => StableHash(Id);

    /// <summary>The additive mask's shape, or empty when the case has none.</summary>
    public long[] MaskShape => Mask switch
    {
        AttnMask.KeyOnly => Layout == AttnLayout.TokenMajor ? [1, Skv] : [1, 1, 1, Skv],
        AttnMask.QueryKey => Layout == AttnLayout.TokenMajor ? [Sq, Skv] : [1, 1, Sq, Skv],
        AttnMask.PerHead => [1, Heads, Sq, Skv],
        AttnMask.PerBatch => [Batch, 1, Sq, Skv],
        _ => [],
    };

    /// <summary>Query rows the oracle checks: all of them, or a contiguous block of 64 at the end for the 4k preset (rows
    /// are independent, so the oracle stays exact).</summary>
    public (int First, int Count) CheckedRows => Seq == AttnSeq.Long4k ? (Sq - 64, 64) : (0, Sq);

    /// <summary>FNV-1a hash of <paramref name="text"/>.</summary>
    public static int StableHash(string text)
    {
        uint hash = 2166136261u;
        foreach (char c in text) hash = (hash ^ c) * 16777619u;
        return (int)(hash & 0x7FFFFFFF);
    }
}

/// <summary>The conformance cases: a deterministic pairwise cover of dtype × layout × mask × GQA group × head dim ×
/// sequence × F16 permission, the shape classes the regression set exercises, and edges.</summary>
public static class AttnCatalog
{
    /// <summary>(query heads, key/value heads) per GQA group: three MHA heads catch stride bugs power-of-two counts hide.</summary>
    public static readonly (int Hq, int Hkv)[] Groups = [(3, 3), (4, 2), (8, 2), (16, 2)];

    /// <summary>Head dims: the odd ones (40, 80, 120) are what cuDNN and the tiled kernels handle differently.</summary>
    public static readonly int[] HeadDims = [40, 64, 80, 120, 128, 256];

    /// <summary>Factor levels: dtype, layout, mask, GQA group, head dim, sequence, F16 permission. Declared before
    /// <see cref="All"/>, whose initializer reads it.</summary>
    private static readonly int[] Levels = [3, 2, 7, Groups.Length, HeadDims.Length, 4, 2];

    /// <summary>Every case.</summary>
    public static IReadOnlyList<AttnCase> All { get; } = Build();

    private static readonly Dictionary<string, AttnCase> ById = All.ToDictionary(c => c.Id, StringComparer.Ordinal);

    /// <summary>The case named <paramref name="id"/>.</summary>
    public static AttnCase Get(string id) => ById[id];

    /// <summary>Theory rows as primitives, so xUnit lists every row and a filter can pick one.</summary>
    public static IEnumerable<object[]> Rows(IEnumerable<string> kinds, Func<AttnCase, bool>? keep = null)
    {
        foreach (string kind in kinds)
            foreach (AttnCase c in All)
                if (keep is null || keep(c)) yield return [kind, c.Id];
    }

    private static List<AttnCase> Build()
    {
        List<AttnCase> cases = [];
        foreach (int[] combo in Pairwise.Cover(Levels, Admissible, seed: 20261020))
            cases.Add(FromCombo(combo));
        cases.AddRange(Anchors());
        cases.AddRange(Edges());
        Dictionary<string, AttnCase> unique = new(StringComparer.Ordinal);
        foreach (AttnCase c in cases) unique.TryAdd(c.Id, c);
        return [.. unique.Values];
    }

    /// <summary>Whether a partial assignment (-1 = not chosen yet) can be completed into a case the legacy entries take.</summary>
    internal static bool Admissible(int[] p)
    {
        AttnDType? dtype = p[0] < 0 ? null : (AttnDType)p[0];
        AttnLayout? layout = p[1] < 0 ? null : (AttnLayout)p[1];
        AttnMask? mask = p[2] < 0 ? null : (AttnMask)p[2];
        int? d = p[4] < 0 ? null : HeadDims[p[4]];
        AttnSeq? seq = p[5] < 0 ? null : (AttnSeq)p[5];
        bool? allowF16 = p[6] < 0 ? null : p[6] == 1;
        bool flash = mask is AttnMask.Causal or AttnMask.Window || (seq == AttnSeq.Decode && mask == AttnMask.None);
        if (layout == AttnLayout.TokenMajor && mask is AttnMask.PerBatch or AttnMask.Causal or AttnMask.Window) return false;
        if (flash && dtype is not null and not AttnDType.F32) return false;
        if (seq == AttnSeq.Long4k && (d > 128 || mask is AttnMask.QueryKey or AttnMask.PerHead or AttnMask.PerBatch or AttnMask.Window))
            return false;
        if (seq == AttnSeq.Decode && mask is AttnMask.QueryKey or AttnMask.PerHead or AttnMask.PerBatch) return false;
        if (seq == AttnSeq.Decode && layout == AttnLayout.TokenMajor) return false;
        if (dtype is not null and not AttnDType.F32 && allowF16 == false) return false;
        return true;
    }

    private static AttnCase FromCombo(int[] p)
    {
        AttnDType dtype = (AttnDType)p[0];
        AttnLayout layout = (AttnLayout)p[1];
        AttnMask mask = (AttnMask)p[2];
        (int hq, int hkv) = Groups[p[3]];
        int d = HeadDims[p[4]];
        AttnSeq seq = (AttnSeq)p[5];
        bool allowF16 = p[6] == 1;
        bool flash = mask is AttnMask.Causal or AttnMask.Window || (seq == AttnSeq.Decode && mask == AttnMask.None);
        int batch = mask == AttnMask.PerBatch ? 2 : 1;
        if (seq == AttnSeq.Long4k) (hq, hkv) = (hq / hkv * 2, 2);
        (int sq, int skv, int kvLen) = (seq, flash) switch
        {
            (AttnSeq.Tiny, false) => (7, 11, 11),
            (AttnSeq.Tiny, true) => (9, 12, 9),
            (AttnSeq.TileTail, false) => d >= 128 ? (33, 65, 65) : (65, 129, 129),
            (AttnSeq.TileTail, true) => d >= 128 ? (33, 48, 33) : (65, 80, 65),
            (AttnSeq.Long4k, _) => (4096, 4096, 4096),
            _ => (1, 800, 750),
        };
        if (!flash) kvLen = skv;
        bool causal = (flash && mask != AttnMask.None) || mask == AttnMask.Causal;
        int qOffset = flash ? kvLen - sq : 0;
        int window = mask == AttnMask.Window ? 5 : 0;
        string id = $"pw-{dtype}-{(layout == AttnLayout.TokenMajor ? "tm" : "hm")}-{mask}-h{hq}x{hkv}-d{d}-{seq}{(allowF16 ? "-f16ok" : "")}"
            .ToLowerInvariant();
        return new(id, dtype, layout, mask, seq, batch, hq, hkv, d, sq, skv, kvLen, qOffset, window, allowF16, flash, causal, "pairwise");
    }

    private static IEnumerable<AttnCase> Anchors()
    {
        // The regression set's shape classes, scaled down where the dispatch does not depend on length.
        yield return new("anchor-krea2-gqa-tm", AttnDType.F16, AttnLayout.TokenMajor, AttnMask.None, AttnSeq.TileTail, 1, 12, 3, 128, 96, 96, 96, 0, 0, true, false, false, "anchor");
        yield return new("anchor-flux-d128", AttnDType.F16, AttnLayout.HeadMajor, AttnMask.None, AttnSeq.TileTail, 1, 4, 4, 128, 96, 96, 96, 0, 0, true, false, false, "anchor");
        yield return new("anchor-t5-perhead", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.PerHead, AttnSeq.Tiny, 1, 8, 8, 64, 24, 24, 24, 0, 0, false, false, false, "anchor");
        yield return new("anchor-sd15-d40", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.None, AttnSeq.TileTail, 1, 4, 4, 40, 70, 70, 70, 0, 0, true, false, false, "anchor");
        yield return new("anchor-sdxl-cfg-b2", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.None, AttnSeq.Tiny, 2, 4, 4, 64, 20, 9, 9, 0, 0, true, false, false, "anchor");
        yield return new("anchor-ideogram4-d256-tm", AttnDType.F16, AttnLayout.TokenMajor, AttnMask.None, AttnSeq.Tiny, 1, 3, 3, 256, 21, 21, 21, 0, 0, true, false, false, "anchor");
        yield return new("anchor-lumina2-d96", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.None, AttnSeq.Tiny, 1, 6, 6, 96, 30, 30, 30, 0, 0, true, false, false, "anchor");
        yield return new("anchor-llm-prefill", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.Causal, AttnSeq.TileTail, 1, 8, 2, 128, 70, 96, 70, 0, 0, false, true, true, "anchor");
        yield return new("anchor-llm-chunk", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.Causal, AttnSeq.TileTail, 1, 8, 2, 64, 40, 128, 110, 70, 0, false, true, true, "anchor");
        yield return new("anchor-llm-decode-gqa", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.Causal, AttnSeq.Decode, 1, 32, 8, 128, 1, 800, 750, 749, 0, false, true, true, "anchor");
        yield return new("anchor-gemma-window", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.Window, AttnSeq.TileTail, 1, 8, 4, 128, 65, 80, 65, 0, 16, false, true, true, "anchor");
    }

    private static IEnumerable<AttnCase> Edges()
    {
        foreach (int d in (int[])[16, 48, 72, 96, 136])
            yield return new($"edge-d{d}", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.None, AttnSeq.Tiny, 1, 3, 3, d, 9, 13, 13, 0, 0, false, false, false, "edge");
        yield return new("edge-one-by-one", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.None, AttnSeq.Tiny, 1, 2, 2, 64, 1, 1, 1, 0, 0, false, false, false, "edge");
        yield return new("edge-skv-above-sq", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.QueryKey, AttnSeq.Tiny, 1, 3, 3, 64, 5, 77, 77, 0, 0, false, false, false, "edge");
        yield return new("edge-window-noncausal", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.Window, AttnSeq.Tiny, 1, 4, 2, 64, 9, 12, 9, 0, 3, false, true, false, "edge");
        yield return new("edge-window-noncausal-decode", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.Window, AttnSeq.Decode, 1, 8, 2, 64, 1, 12, 9, 4, 3, false, true, false, "edge");
        yield return new("edge-tm-keyonly", AttnDType.F32, AttnLayout.TokenMajor, AttnMask.KeyOnly, AttnSeq.TileTail, 1, 4, 4, 64, 66, 66, 66, 0, 0, true, false, false, "edge");
        yield return new("edge-tm-perhead", AttnDType.F16, AttnLayout.TokenMajor, AttnMask.PerHead, AttnSeq.Tiny, 1, 4, 2, 64, 70, 70, 70, 0, 0, true, false, false, "edge");
        yield return new("edge-sentinel-row", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.QueryKey, AttnSeq.Tiny, 1, 3, 3, 64, 7, 11, 11, 0, 0, false, false, false, "edge");
    }
}

/// <summary>Deterministic greedy all-pairs cover: every value pair of every two factors that some admissible row can hold
/// appears in at least one row. Same seed, same rows.</summary>
public static class Pairwise
{
    /// <summary>Rows covering every admissible pair of <paramref name="levels"/>.</summary>
    public static IEnumerable<int[]> Cover(IReadOnlyList<int> levels, Func<int[], bool> admissible, int seed)
    {
        int n = levels.Count;
        HashSet<(int, int, int, int)> uncovered = [];
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++)
                for (int va = 0; va < levels[a]; va++)
                    for (int vb = 0; vb < levels[b]; vb++)
                    {
                        int[] probe = Enumerable.Repeat(-1, n).ToArray();
                        probe[a] = va;
                        probe[b] = vb;
                        if (admissible(probe)) uncovered.Add((a, va, b, vb));
                    }
        Random rng = new(seed);
        List<int[]> rows = [];
        while (uncovered.Count > 0)
        {
            (int fa, int vfa, int fb, int vfb) = uncovered.First();
            int[] row = Enumerable.Repeat(-1, n).ToArray();
            row[fa] = vfa;
            row[fb] = vfb;
            for (int f = 0; f < n; f++)
            {
                if (row[f] >= 0) continue;
                int best = -1, bestGain = -1;
                foreach (int v in Enumerable.Range(0, levels[f]).OrderBy(_ => rng.Next()))
                {
                    row[f] = v;
                    if (!admissible(row)) continue;
                    int gain = 0;
                    for (int g = 0; g < n; g++)
                        if (g != f && row[g] >= 0 && uncovered.Contains(g < f ? (g, row[g], f, v) : (f, v, g, row[g]))) gain++;
                    if (gain > bestGain) (best, bestGain) = (v, gain);
                }
                row[f] = best;
                if (best < 0) break;
            }
            if (row.Any(v => v < 0) || !admissible(row))
            {
                uncovered.Remove((fa, vfa, fb, vfb));
                continue;
            }
            for (int a = 0; a < n; a++)
                for (int b = a + 1; b < n; b++)
                    uncovered.Remove((a, row[a], b, row[b]));
            rows.Add(row);
        }
        return rows;
    }
}
