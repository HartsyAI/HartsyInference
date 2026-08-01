using System.Text.Json;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Exact-equality check of <see cref="SeedVr2Windowing"/> against slices emitted by ByteDance's
/// reference <c>models/dit_v2/window.py</c>. The fixture (<c>fixtures/seedvr2_windows.json</c>) is committed —
/// regenerate with <c>fixtures/seedvr2_window_fixture_dump.py &lt;path-to-SeedVR&gt;/models/dit_v2/window.py</c>. Covers
/// ragged boundaries, the t&gt;30 temporal clamp, single-window axes, and banker's-rounding area normalization.
/// Unit tier: pure index math, no GPU, no weights.</summary>
public sealed class SeedVr2WindowingTests
{
    private readonly ITestOutputHelper _output;

    public SeedVr2WindowingTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void WindowSlices_MatchReference_AllFixtureCases()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "fixtures", "seedvr2_windows.json");
        Assert.True(File.Exists(path), $"Fixture missing: {path}");

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        int cases = 0, slicesChecked = 0;
        foreach (JsonElement el in doc.RootElement.EnumerateArray())
        {
            int[] size = el.GetProperty("size").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            int[] num = el.GetProperty("num").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            string method = el.GetProperty("method").GetString()!;
            List<int[]> expected = el.GetProperty("slices").EnumerateArray()
                .Select(s => s.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToList();

            SeedVr2WindowSlice[] actual = method == "720pwin_by_size_bysize"
                ? SeedVr2Windowing.MakeWindows(size[0], size[1], size[2], num[0], num[1], num[2])
                : SeedVr2Windowing.MakeShiftedWindows(size[0], size[1], size[2], num[0], num[1], num[2]);

            Assert.True(expected.Count == actual.Length,
                $"{method} ({size[0]},{size[1]},{size[2]}): expected {expected.Count} windows, got {actual.Length}");
            for (int i = 0; i < actual.Length; i++)
            {
                int[] e = expected[i];
                SeedVr2WindowSlice a = actual[i];
                Assert.True(
                    e[0] == a.T0 && e[1] == a.T1 && e[2] == a.H0 && e[3] == a.H1 && e[4] == a.W0 && e[5] == a.W1,
                    $"{method} ({size[0]},{size[1]},{size[2]}) window {i}: expected " +
                    $"[{e[0]},{e[1]},{e[2]},{e[3]},{e[4]},{e[5]}], got " +
                    $"[{a.T0},{a.T1},{a.H0},{a.H1},{a.W0},{a.W1}]");
            }
            cases++;
            slicesChecked += actual.Length;
        }
        _output.WriteLine($"Verified {cases} cases, {slicesChecked} slices, exact match.");
        Assert.True(cases >= 40, "Fixture unexpectedly small — regeneration may have truncated it.");
    }

    [Fact]
    public void RegularWindows_TileGridExactly()
    {
        // Independent structural property (not fixture-derived): the unshifted partition covers every token once.
        foreach ((int t, int h, int w) in new[] { (7, 45, 80), (13, 60, 106), (1, 1, 1), (31, 44, 79) })
        {
            SeedVr2WindowSlice[] slices = SeedVr2Windowing.MakeWindows(t, h, w, 4, 3, 3);
            bool[] covered = new bool[t * h * w];
            foreach (SeedVr2WindowSlice s in slices)
            {
                for (int it = s.T0; it < s.T1; it++)
                for (int ih = s.H0; ih < s.H1; ih++)
                for (int iw = s.W0; iw < s.W1; iw++)
                {
                    int idx = (it * h + ih) * w + iw;
                    Assert.False(covered[idx], $"Token ({it},{ih},{iw}) covered twice in ({t},{h},{w})");
                    covered[idx] = true;
                }
            }
            Assert.DoesNotContain(false, covered);
        }
    }
}
