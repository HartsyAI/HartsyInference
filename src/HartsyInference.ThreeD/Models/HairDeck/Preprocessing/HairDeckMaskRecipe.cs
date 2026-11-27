using HartsyInference.Vision.Segmentation;

namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>The mask recipe of the released HairDeck preset "best" (hairdeck.prep.recipe.run with that configuration): picks the
/// SAM 3 instances of the person, face and hair prompts, adds clothes and hands the matting model agrees with, and fills and
/// softens the person matte. Works at the working resolution; the caller rebuilds hair and face at full resolution from the keys.</summary>
internal sealed class HairDeckMaskRecipe
{
    /// <summary>The SAM 3 prompts of the preset in their order.</summary>
    public static readonly string[] Prompts = ["person", "anime character", "hair", "face", "fur", "clothes", "hand"];

    private const double PersonThreshold = 0.4;
    private const double HairThreshold = 0.5;
    private const double FaceThreshold = 0.5;
    private const double FurThreshold = 0.5;
    private const double ExtraThreshold = 0.5;
    private const double BoxFallbackThreshold = 0.2;
    private const double MainOverlap = 0.3;
    private const double ClusterOverlap = 0.1;
    private const double ClusterAlpha = 0.5;
    private const double ExtraAlpha = 0.5;
    private const double ExtraMax = 0.3;
    private const double NearRadiusFraction = 0.01;
    private const double DensWidthFraction = 0.04;
    private const float DensFill = 0.7f;
    private const float DensThreshold = 0.7f;
    private const float AddThreshold = 0.7f;
    private const double AddRatio = 0.7;
    private const double AddMax = 1.0;
    private const double FeatherSigma = 0.8;
    private const float FeatherEpsilon = 1e-3f;

    private static readonly string[] PersonPrompts = ["person", "anime character"];
    private static readonly string[] HairPrompts = ["hair"];
    private static readonly string[] FacePrompts = ["face"];
    private static readonly string[] FurPrompts = ["fur"];
    private static readonly string[] ExtraPrompts = ["clothes", "hand"];

    private readonly int _height;
    private readonly int _width;
    private readonly int _count;
    private readonly float[] _alpha;
    private readonly int _longSide;
    private readonly Dictionary<string, TextGroundedLogitResult> _byPrompt = new(StringComparer.Ordinal);
    private readonly Dictionary<HairDeckInstanceKey, bool[]> _masks = new();

    private HairDeckMaskRecipe(IReadOnlyList<TextGroundedLogitResult> sam, int height, int width, float[] alpha)
    {
        _height = height;
        _width = width;
        _count = height * width;
        _alpha = alpha;
        _longSide = Math.Max(height, width);
        foreach (TextGroundedLogitResult result in sam)
        {
            _byPrompt[result.Prompt] = result;
        }
    }

    /// <summary>Runs the recipe for one image at the working resolution <paramref name="height"/> x <paramref name="width"/>.</summary>
    /// <param name="sam">SAM 3 queries of the preset prompts above the working score floor, highest first.</param>
    /// <param name="alpha">Whole-character matte of the matting model at the working resolution, in [0, 1].</param>
    public static HairDeckRecipeResult Run(IReadOnlyList<TextGroundedLogitResult> sam, int height, int width, float[] alpha)
    {
        ArgumentNullException.ThrowIfNull(sam);
        ArgumentNullException.ThrowIfNull(alpha);
        if (alpha.Length != (long)height * width)
        {
            throw new ArgumentException($"Matte holds {alpha.Length} values; a {width}x{height} working image needs {(long)height * width}.", nameof(alpha));
        }

        return new HairDeckMaskRecipe(sam, height, width, alpha).Execute();
    }

    private HairDeckRecipeResult Execute()
    {
        bool boxFallback = false;
        List<Candidate> faces = Candidates(FacePrompts, FaceThreshold);
        if (faces.Count == 0 && Candidates(HairPrompts, HairThreshold).Count == 0)
        {
            faces = TopByScore(Candidates(FacePrompts, BoxFallbackThreshold), 1);
            boxFallback = true;
        }

        List<Candidate> persons = Candidates(PersonPrompts, PersonThreshold);
        bool[] face = new bool[_count];
        List<HairDeckInstanceKey> faceKeys = [];
        if (faces.Count > 0)
        {
            Candidate chosen;
            if (persons.Count > 0)
            {
                bool[] largestPerson = LargestByArea(persons).Mask;
                List<Candidate> inside = [];
                foreach (Candidate candidate in faces)
                {
                    if (Overlap(candidate.Mask, largestPerson) > 0.5 * Math.Max(1, Count(candidate.Mask)))
                    {
                        inside.Add(candidate);
                    }
                }

                chosen = BestByScore(inside.Count > 0 ? inside : faces);
            }
            else
            {
                chosen = BestByScore(faces);
            }

            face = chosen.Mask;
            faceKeys.Add(chosen.Key);
        }

        bool[] person0 = new bool[_count];
        int unselected = 0;
        if (persons.Count > 0)
        {
            (person0, unselected) = ClusterAroundAnchor(persons, face);
        }

        List<Candidate> hairs = Candidates(HairPrompts, HairThreshold);
        if (hairs.Count == 0 && boxFallback)
        {
            hairs = TopByScore(Candidates(HairPrompts, BoxFallbackThreshold), 1);
        }

        if (hairs.Count > 1)
        {
            hairs = WithoutFur(hairs);
        }

        bool[] hair = Union(hairs);
        List<HairDeckInstanceKey> hairKeys = [];
        foreach (Candidate candidate in hairs)
        {
            hairKeys.Add(candidate.Key);
        }

        bool[] matte;
        if (persons.Count == 0)
        {
            if (!AnyAboveHalf(_alpha))
            {
                return new HairDeckRecipeResult(FilledMatte(1f), hairKeys, faceKeys, HairDeckPrepStatus.NoPersonMask, boxFallback, null, 0);
            }

            matte = new bool[_count];
            for (int i = 0; i < _count; i++)
            {
                matte[i] = _alpha[i] > 0.5f;
            }
        }
        else
        {
            matte = new bool[_count];
            for (int i = 0; i < _count; i++)
            {
                matte[i] = person0[i] || hair[i] || face[i];
            }
        }

        AddExtraParts(matte);
        FillDensePixels(matte);
        AddEnclosedGaps(matte);

        float[] soft = new float[_count];
        for (int i = 0; i < _count; i++)
        {
            soft[i] = matte[i] ? 1f : 0f;
        }

        float[] blurred = HairDeckMorphology.GaussianReflect(soft, _height, _width, FeatherSigma);
        for (int i = 0; i < _count; i++)
        {
            soft[i] = matte[i]
                ? Math.Max(blurred[i], 0.5f + FeatherEpsilon)
                : Math.Min(blurred[i], 0.5f - FeatherEpsilon);
        }

        int intersection = 0, unionCount = 0;
        for (int i = 0; i < _count; i++)
        {
            bool inside = soft[i] > 0.5f;
            bool matted = _alpha[i] > 0.5f;
            if (inside && matted)
            {
                intersection++;
            }

            if (inside || matted)
            {
                unionCount++;
            }
        }

        double agree = (double)intersection / Math.Max(1, unionCount);
        return new HairDeckRecipeResult(soft, hairKeys, faceKeys, HairDeckPrepStatus.Ok, boxFallback, agree, unselected);
    }

    /// <summary>The person union: the instances overlapping the anchor (the face, else the best person) plus, repeatedly, every
    /// instance mostly covered by that union, skipping those whose matte is weak. Also counts the person instances left out.</summary>
    private (bool[] Mask, int Unselected) ClusterAroundAnchor(List<Candidate> persons, bool[] face)
    {
        bool[] anchor = AnyTrue(face) ? face : BestByScore(persons).Mask;
        double anchorArea = Math.Max(1, Count(anchor));
        bool[] baseMask = new bool[_count];
        bool[] included = new bool[persons.Count];
        bool any = false;
        for (int i = 0; i < persons.Count; i++)
        {
            if (Overlap(persons[i].Mask, anchor) >= MainOverlap * anchorArea)
            {
                OrInto(baseMask, persons[i].Mask);
                included[i] = true;
                any = true;
            }
        }

        if (!any)
        {
            int best = IndexOfBest(persons);
            OrInto(baseMask, persons[best].Mask);
            included[best] = true;
        }

        bool[] used = new bool[persons.Count];
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int i = 0; i < persons.Count; i++)
            {
                if (used[i])
                {
                    continue;
                }

                bool[] candidate = persons[i].Mask;
                if (Overlap(candidate, baseMask) <= ClusterOverlap * Math.Max(1, Count(candidate)))
                {
                    continue;
                }

                if (MeanOver(candidate) < ClusterAlpha)
                {
                    continue;
                }

                OrInto(baseMask, candidate);
                used[i] = true;
                changed = true;
            }
        }

        int unselected = 0;
        for (int i = 0; i < persons.Count; i++)
        {
            if (!included[i] && !used[i])
            {
                unselected++;
            }
        }

        return (baseMask, unselected);
    }

    private static int IndexOfBest(List<Candidate> candidates)
    {
        int best = 0;
        for (int i = 1; i < candidates.Count; i++)
        {
            if (candidates[i].Score > candidates[best].Score)
            {
                best = i;
            }
        }

        return best;
    }

    /// <summary>Drops hair instances, other than the best one, that lie mostly on fur instances.</summary>
    private List<Candidate> WithoutFur(List<Candidate> hairs)
    {
        List<Candidate> fur = Candidates(FurPrompts, FurThreshold);
        if (fur.Count == 0)
        {
            return hairs;
        }

        bool[] furUnion = Union(fur);
        Candidate best = BestByScore(hairs);
        List<Candidate> kept = [];
        foreach (Candidate candidate in hairs)
        {
            if (ReferenceEquals(candidate, best) || Overlap(candidate.Mask, furUnion) < 0.5 * Math.Max(1, Count(candidate.Mask)))
            {
                kept.Add(candidate);
            }
        }

        return kept.Count < hairs.Count ? kept : hairs;
    }

    /// <summary>Adds clothes and hand instances that touch the matte, are small enough, and are mostly foreground to the matting model.</summary>
    private void AddExtraParts(bool[] matte)
    {
        if (!AnyTrue(matte))
        {
            return;
        }

        int radius = HairDeckMorphology.Radius(NearRadiusFraction * _longSide);
        bool[] near = HairDeckMorphology.Dilate(matte, _height, _width, radius);
        foreach (Candidate candidate in Candidates(ExtraPrompts, ExtraThreshold))
        {
            bool touches = false;
            for (int i = 0; i < _count && !touches; i++)
            {
                touches = candidate.Mask[i] && near[i];
            }

            if (touches && Count(candidate.Mask) <= ExtraMax * Count(matte) && MeanOver(candidate.Mask) >= ExtraAlpha)
            {
                OrInto(matte, candidate.Mask);
            }
        }
    }

    /// <summary>Adds matte pixels that the matting model calls foreground where the matte is dense around them.</summary>
    private void FillDensePixels(bool[] matte)
    {
        if (!AnyTrue(matte))
        {
            return;
        }

        int size = Math.Max(3, (int)Math.Round(DensWidthFraction * _longSide, MidpointRounding.ToEven) | 1);
        double[] density = HairDeckMorphology.BoxMean(matte, _height, _width, size);
        for (int i = 0; i < _count; i++)
        {
            if (_alpha[i] > DensThreshold && density[i] >= DensFill)
            {
                matte[i] = true;
            }
        }
    }

    /// <summary>Adds matting-model components that are missing from the matte and mostly enclosed by it (or the image border).</summary>
    private void AddEnclosedGaps(bool[] matte)
    {
        if (!AnyTrue(matte))
        {
            return;
        }

        bool[] candidate = new bool[_count];
        for (int i = 0; i < _count; i++)
        {
            candidate[i] = _alpha[i] > AddThreshold && !matte[i];
        }

        int[] labels = new int[_count];
        int components = HairDeckMorphology.Label(candidate, _height, _width, labels);
        if (components == 0)
        {
            return;
        }

        int paddedWidth = _width + 2, paddedHeight = _height + 2;
        int[] labelsPadded = new int[paddedWidth * paddedHeight];
        bool[] maskPadded = new bool[paddedWidth * paddedHeight];
        for (int y = 0; y < paddedHeight; y++)
        {
            for (int x = 0; x < paddedWidth; x++)
            {
                bool inside = y > 0 && y < paddedHeight - 1 && x > 0 && x < paddedWidth - 1;
                maskPadded[y * paddedWidth + x] = !inside || matte[(y - 1) * _width + (x - 1)];
                if (inside)
                {
                    labelsPadded[y * paddedWidth + x] = labels[(y - 1) * _width + (x - 1)];
                }
            }
        }

        int[] dilated = new int[labelsPadded.Length];
        for (int y = 0; y < paddedHeight; y++)
        {
            for (int x = 0; x < paddedWidth; x++)
            {
                int best = 0;
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < paddedHeight && nx >= 0 && nx < paddedWidth)
                        {
                            best = Math.Max(best, labelsPadded[ny * paddedWidth + nx]);
                        }
                    }
                }

                dilated[y * paddedWidth + x] = best;
            }
        }

        long[] total = new long[components + 1], support = new long[components + 1], area = new long[components + 1];
        for (int p = 0; p < labelsPadded.Length; p++)
        {
            bool ring = dilated[p] > 0 && labelsPadded[p] == 0;
            if (!ring)
            {
                continue;
            }

            total[dilated[p]]++;
            if (maskPadded[p])
            {
                support[dilated[p]]++;
            }
        }

        for (int i = 0; i < _count; i++)
        {
            if (labels[i] > 0)
            {
                area[labels[i]]++;
            }
        }

        long matteCount = Count(matte);
        bool[] accepted = new bool[components + 1];
        for (int k = 1; k <= components; k++)
        {
            accepted[k] = support[k] >= AddRatio * Math.Max(total[k], 1) && area[k] <= AddMax * matteCount;
        }

        for (int i = 0; i < _count; i++)
        {
            if (labels[i] > 0 && accepted[labels[i]])
            {
                matte[i] = true;
            }
        }
    }

    private List<Candidate> Candidates(string[] prompts, double threshold)
    {
        List<Candidate> result = [];
        foreach (string prompt in prompts)
        {
            if (!_byPrompt.TryGetValue(prompt, out TextGroundedLogitResult? found))
            {
                continue;
            }

            for (int j = 0; j < found.Instances.Count; j++)
            {
                TextGroundedLogit instance = found.Instances[j];
                if (instance.Score <= threshold)
                {
                    continue;
                }

                HairDeckInstanceKey key = new(prompt, j);
                if (!_masks.TryGetValue(key, out bool[]? mask))
                {
                    mask = HairDeckMorphology.LogitsToMask(instance.Logits, found.LogitSide, _height, _width);
                    _masks[key] = mask;
                }

                result.Add(new Candidate(instance.Score, key, mask));
            }
        }

        return result;
    }

    private static List<Candidate> TopByScore(List<Candidate> candidates, int count)
    {
        List<Candidate> sorted = [.. candidates];
        for (int i = 1; i < sorted.Count; i++)
        {
            Candidate current = sorted[i];
            int j = i - 1;
            while (j >= 0 && sorted[j].Score < current.Score)
            {
                sorted[j + 1] = sorted[j];
                j--;
            }

            sorted[j + 1] = current;
        }

        return sorted.Count > count ? sorted.GetRange(0, count) : sorted;
    }

    private static Candidate BestByScore(List<Candidate> candidates)
    {
        Candidate best = candidates[0];
        foreach (Candidate candidate in candidates)
        {
            if (candidate.Score > best.Score)
            {
                best = candidate;
            }
        }

        return best;
    }

    private static Candidate LargestByArea(List<Candidate> candidates)
    {
        Candidate best = candidates[0];
        int bestCount = Count(best.Mask);
        foreach (Candidate candidate in candidates)
        {
            int count = Count(candidate.Mask);
            if (count > bestCount)
            {
                best = candidate;
                bestCount = count;
            }
        }

        return best;
    }

    private bool[] Union(List<Candidate> candidates)
    {
        bool[] result = new bool[_count];
        foreach (Candidate candidate in candidates)
        {
            OrInto(result, candidate.Mask);
        }

        return result;
    }

    private float[] FilledMatte(float value)
    {
        float[] result = new float[_count];
        Array.Fill(result, value);
        return result;
    }

    private double MeanOver(bool[] mask)
    {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < _count; i++)
        {
            if (mask[i])
            {
                sum += _alpha[i];
                count++;
            }
        }

        return sum / count;
    }

    private bool AnyAboveHalf(float[] alpha)
    {
        foreach (float value in alpha)
        {
            if (value > 0.5f)
            {
                return true;
            }
        }

        return false;
    }

    private static int Count(bool[] mask)
    {
        int count = 0;
        foreach (bool value in mask)
        {
            if (value)
            {
                count++;
            }
        }

        return count;
    }

    private static bool AnyTrue(bool[] mask)
    {
        foreach (bool value in mask)
        {
            if (value)
            {
                return true;
            }
        }

        return false;
    }

    private static int Overlap(bool[] a, bool[] b)
    {
        int count = 0;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] && b[i])
            {
                count++;
            }
        }

        return count;
    }

    private static void OrInto(bool[] target, bool[] source)
    {
        for (int i = 0; i < target.Length; i++)
        {
            target[i] |= source[i];
        }
    }

    /// <summary>One SAM 3 instance that passed a threshold: its score, its key, and its mask at the working resolution.</summary>
    private sealed record Candidate(float Score, HairDeckInstanceKey Key, bool[] Mask);
}

/// <summary>Identifies one SAM 3 instance: the prompt it was found for and its index in that prompt's list.</summary>
internal readonly record struct HairDeckInstanceKey(string Prompt, int Index);

/// <summary>The recipe's output at the working resolution.</summary>
internal sealed record HairDeckRecipeResult(
    float[] Alpha,
    IReadOnlyList<HairDeckInstanceKey> HairKeys,
    IReadOnlyList<HairDeckInstanceKey> FaceKeys,
    HairDeckPrepStatus Status,
    bool BoxFallback,
    double? Agreement,
    int UnselectedPersonCount);
