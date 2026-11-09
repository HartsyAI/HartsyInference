using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Element-wise comparison of C# tensors against Python reference arrays, and the report the parity tests print.</summary>
internal static class ParityCompare
{
    /// <summary>An element passes when <c>|actual - expected| ≤ AbsTolerance + RelTolerance·|expected|</c>. fp32 accumulation
    /// over the transformer and the sparse convolutions stays well inside this; a larger gap is a real difference.</summary>
    internal const double AbsTolerance = 1e-4;
    internal const double RelTolerance = 1e-4;

    /// <summary>Compares two arrays of the same length. <c>MaxRel</c> is <c>max|actual - expected| / max|expected|</c>, a
    /// per-tensor scale; the element-wise criterion is the one that decides pass or fail.</summary>
    internal static ParityResult Compare(string name, ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, string? note = null)
    {
        if (actual.Length != expected.Length)
            return ParityResult.Failed(name, $"length {actual.Length} differs from reference {expected.Length}");

        double maxAbs = 0.0, maxExpected = 0.0;
        int failures = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = actual[i], e = expected[i];
            if (!double.IsFinite(a) || !double.IsFinite(e))
            {
                failures++;
                continue;
            }

            double d = Math.Abs(a - e);
            maxAbs = Math.Max(maxAbs, d);
            maxExpected = Math.Max(maxExpected, Math.Abs(e));
            if (d > AbsTolerance + RelTolerance * Math.Abs(e))
                failures++;
        }

        double maxRel = maxExpected > 0.0 ? maxAbs / maxExpected : maxAbs;
        return new ParityResult(name, expected.Length, maxAbs, maxRel, failures, note);
    }

    /// <summary>Matches each reference row to the actual row with the same coordinate quad. Returns, per reference row, the actual
    /// row index, or -1 when no actual row carries that coordinate. Fails when the two coordinate sets differ in size.</summary>
    internal static int[] MatchRows(int[] actualCoords, int[] expectedCoords)
    {
        int actualRows = actualCoords.Length / 4, expectedRows = expectedCoords.Length / 4;
        Dictionary<(int, int, int, int), int> index = new(actualRows);
        for (int r = 0; r < actualRows; r++)
            index.TryAdd((actualCoords[r * 4], actualCoords[r * 4 + 1], actualCoords[r * 4 + 2], actualCoords[r * 4 + 3]), r);

        int[] order = new int[expectedRows];
        for (int r = 0; r < expectedRows; r++)
        {
            (int, int, int, int) key = (expectedCoords[r * 4], expectedCoords[r * 4 + 1], expectedCoords[r * 4 + 2], expectedCoords[r * 4 + 3]);
            order[r] = index.TryGetValue(key, out int at) ? at : -1;
        }

        return order;
    }

    /// <summary>Coordinate stage: passes when the two voxel sets are identical. Row order may differ; it is noted, not judged.</summary>
    internal static ParityResult CompareCoords(string name, int[] actualCoords, int[] expectedCoords, out int[] order)
    {
        order = MatchRows(actualCoords, expectedCoords);
        int actualRows = actualCoords.Length / 4;
        int unmatched = order.Count(index => index < 0);
        if (actualRows != order.Length || unmatched != 0)
            return ParityResult.Failed(name, $"voxel sets differ: actual {actualRows} rows, reference {order.Length}, unmatched reference rows {unmatched}");

        bool identity = order.Select((index, row) => index == row).All(same => same);
        return new ParityResult(name, order.Length, 0.0, 0.0, 0, identity ? "row order identical to reference" : "row order differs; rows matched by coordinate");
    }

    /// <summary>Gathers rows <c>order[r]</c> of a width-<paramref name="width"/> array into reference order.</summary>
    internal static float[] GatherRows(ReadOnlySpan<float> source, int[] order, int width)
    {
        float[] gathered = new float[order.Length * width];
        for (int r = 0; r < order.Length; r++)
            source.Slice(order[r] * width, width).CopyTo(gathered.AsSpan(r * width, width));
        return gathered;
    }

    /// <summary>Prints one line per stage in pipeline order and fails the test at the first stage that is out of tolerance.</summary>
    internal static void Finish(IReadOnlyList<ParityResult> results, ITestOutputHelper output)
    {
        output.WriteLine($"{"stage",-28} {"count",8} {"max_abs",12} {"max_rel",12}  status  note");
        foreach (ParityResult result in results)
        {
            string status = result.Passed ? "PASS" : "FAIL";
            output.WriteLine($"{result.Name,-28} {result.Count,8} {result.MaxAbs,12:E4} {result.MaxRel,12:E4}  {status,-6}  {result.Note}");
        }

        ParityResult? first = results.Cast<ParityResult?>().FirstOrDefault(result => result is { Passed: false });
        if (first is not null)
            Assert.Fail($"First differing stage: {first.Value.Name} ({first.Value.Note ?? "outside tolerance"}).");
    }
}
