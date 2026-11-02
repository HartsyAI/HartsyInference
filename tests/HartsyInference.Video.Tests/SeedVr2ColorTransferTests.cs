using System.Text.Json;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>Pins the LAB and AdaIN colour transfers to ComfyUI's SeedVR2 post-processing (<c>comfy/ldm/seedvr/color_fix.py</c>).
/// The fixture comes from <c>tests/python-reference/seedvr2_reference/dump_seedvr2_color_transfer.py</c>; the planes are
/// deterministic and tie-free, so the comparison is by value, not by tie-break order.</summary>
public sealed class SeedVr2ColorTransferTests
{
    /// <summary>Float32 reference vs double-precision C#: the LAB cube roots and the 2.4 power dominate the drift.</summary>
    private const float Tolerance = 2e-3f;

    [Fact]
    public void Lab_MatchesComfyUiReference()
    {
        foreach (Case c in LoadCases())
        {
            float[] model = [.. c.Model];
            SeedVr2ColorFix.TransferLab(model, c.Reference, c.Height, c.Width);
            AssertClose(c.Lab, model, $"lab {c.Height}x{c.Width}");
        }
    }

    [Fact]
    public void AdaIn_MatchesComfyUiReference()
    {
        foreach (Case c in LoadCases())
        {
            float[] model = [.. c.Model];
            SeedVr2ColorFix.TransferAdaIn(model, c.Reference, 3, c.Height, c.Width);
            AssertClose(c.AdaIn, model, $"adain {c.Height}x{c.Width}");
        }
    }

    [Fact]
    public void Transfers_LeaveTheInputUnchangedWhenItAlreadyMatches()
    {
        // The reference is its own input: a transfer to itself is the identity (within rounding) for AdaIN.
        Case c = LoadCases()[0];
        float[] model = [.. c.Reference];
        SeedVr2ColorFix.TransferAdaIn(model, c.Reference, 3, c.Height, c.Width);
        AssertClose(c.Reference, model, "adain self-transfer");
    }

    private static void AssertClose(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0;
        int worstIndex = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            if (diff > worst)
            {
                worst = diff;
                worstIndex = i;
            }
        }
        Assert.True(worst <= Tolerance, $"{label}: max |diff| {worst} at {worstIndex} (expected {expected[Math.Max(0, worstIndex)]}, got {actual[Math.Max(0, worstIndex)]})");
    }

    private static Case[] LoadCases()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "fixtures", "seedvr2_color_transfer.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(e => new Case(
            e.GetProperty("height").GetInt32(),
            e.GetProperty("width").GetInt32(),
            Floats(e, "model"),
            Floats(e, "reference"),
            Floats(e, "lab"),
            Floats(e, "adain"))).ToArray();
    }

    private static float[] Floats(JsonElement e, string name) =>
        e.GetProperty(name).EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();

    private sealed record Case(int Height, int Width, float[] Model, float[] Reference, float[] Lab, float[] AdaIn);
}
