using System.Text.Json;
using HartsyInference.LLM.Sampling;
using Xunit;

namespace HartsyInference.LLM.Tests.Sampling;

/// <summary>The DRY penalty against the upstream Haru sampler's penalties on fixed histories
/// (<c>tools/haru/dry_reference.py</c>, <c>haru.sampling.DryState</c>).</summary>
public sealed class DryPenaltyStepTests
{
    private static readonly string Path = System.IO.Path.Combine(AppContext.BaseDirectory, "Fixtures", "Dry", "dry_cases.json");

    public static TheoryData<string> CaseNames()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path));
        TheoryData<string> names = [];
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            names.Add(c.GetProperty("name").GetString()!);
        }
        return names;
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Penalties_MatchUpstream(string name)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path));
        int vocab = doc.RootElement.GetProperty("vocab").GetInt32();
        JsonElement c = doc.RootElement.GetProperty("cases").EnumerateArray().First(e => e.GetProperty("name").GetString() == name);
        JsonElement cfg = c.GetProperty("config");
        DryOptions options = new()
        {
            Multiplier = cfg.GetProperty("multiplier").GetSingle(), Base = cfg.GetProperty("base").GetSingle(),
            AllowedLength = cfg.GetProperty("allowed_length").GetInt32(), Lookback = cfg.GetProperty("lookback").GetInt32(),
            MaxMatchLength = cfg.GetProperty("max_match_length").GetInt32(), MaxPenalty = cfg.GetProperty("max_penalty").GetSingle(),
        };
        int[] exempt = [.. c.GetProperty("exempt").EnumerateArray().Select(static e => e.GetInt32())];
        int[] history = [.. c.GetProperty("history").EnumerateArray().Select(static e => e.GetInt32())];
        Dictionary<int, float> expected = c.GetProperty("penalties").EnumerateObject().ToDictionary(static p => int.Parse(p.Name), static p => p.Value.GetSingle());

        float[] logits = new float[vocab];
        new DryPenaltyStep(options, exempt).Apply(logits, history);

        for (int t = 0; t < vocab; t++)
        {
            float want = expected.TryGetValue(t, out float p) ? -p : 0f;
            Assert.True(MathF.Abs(logits[t] - want) <= 1e-5f * MathF.Max(1f, MathF.Abs(want)), $"token {t}: {logits[t]} vs {want}");
        }
    }

    [Fact]
    public void FromSteps_AppliesTheStepsInOrderBeforeTheDraw()
    {
        // Token 3 follows the repeated 5 5 5: DRY must push it below token 4, which greedy then takes.
        int[] history = [3, 5, 5, 3, 5, 5];
        SamplerChain chain = SamplerChain.FromSteps([new DryPenaltyStep(new DryOptions(), []), new TemperatureStep(0.7f)], greedy: true, seed: 1);
        float[] logits = new float[8];
        logits[3] = 1.0f;
        logits[4] = 0.5f;
        Assert.Equal(4, chain.Next(logits, history));
        float[] plain = new float[8];
        plain[3] = 1.0f;
        plain[4] = 0.5f;
        Assert.Equal(3, SamplerChain.FromSteps([], greedy: true, seed: 1).Next(plain, history));
    }

    [Fact]
    public void Options_RefuseNonsense() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new DryPenaltyStep(new DryOptions { Lookback = 0 }, []));
}
