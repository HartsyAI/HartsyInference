using System.Text.Json;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;
using Kind = HartsyInference.Diffusion.Pipelines.LongCat.LongCatAvatarSchedule.Kind;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The avatar sigma lists against the pinned <c>get_timesteps_sigmas</c> (tests/python-reference/longcat_avatar/dump_longcat_avatar_schedule.py,
/// @ 58eb336). Skipped, and so not evidence, until HARTSY_LONGCAT_AVATAR_DIR names a dump that has schedule.json.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAvatarScheduleParityTests
{
    private const string EnvName = "HARTSY_LONGCAT_AVATAR_DIR";

    [LongCatAvatarScheduleFact]
    public void EveryCaseMatchesTheReferenceSigmas()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "schedule.json")));
        List<string> failures = [];
        foreach (JsonElement c in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            string modelType = c.GetProperty("model_type").GetString()!;
            bool distill = c.GetProperty("distill").GetBoolean();
            int steps = c.GetProperty("steps").GetInt32();
            Kind kind = !distill ? Kind.Linear : modelType == "avatar-v1.5" ? Kind.Distill8 : Kind.Distill50;
            float[] expected = c.GetProperty("sigmas").EnumerateArray().Select(static e => (float)e.GetDouble()).ToArray();

            float[] actual = LongCatAvatarSchedule.Sigmas(steps, kind);
            string label = $"{modelType} distill={distill} steps={steps}";
            if (actual.Length != expected.Length)
            {
                failures.Add($"{label}: {actual.Length} sigmas, reference has {expected.Length}");
                continue;
            }
            for (int i = 0; i < actual.Length; i++)
                if (Math.Abs(actual[i] - expected[i]) > 1e-6f)
                    failures.Add($"{label}: sigma[{i}] {actual[i]} vs {expected[i]}");
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private sealed class LongCatAvatarScheduleFactAttribute : FactAttribute
    {
        public LongCatAvatarScheduleFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "schedule.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no schedule.json; the LongCat avatar schedule reference comparison did not run.";
        }
    }
}
