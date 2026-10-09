using System.Text.Json;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The DSpark head and the target's draft taps on the synthetic model, against the upstream run in <c>fixtures/dspark_forward.json</c> (written by
/// <c>dump_dspark_fixture.py</c>). The target is the backbone of <c>model_forward.json</c>; the draft weights are random and seeded. Ungated: runs on the CPU lane.</summary>
/// <remarks>Each decode position is drafted from a fresh window seeded with the committed rows before it, and must reproduce upstream's incremental window. The
/// positions past the 8-token window cover the wrap. Tolerance as the host fixture: float32 accumulation order, exact softmax.</remarks>
public sealed class DeepSeekV41DSparkFixtureTests
{
    [Fact]
    public void Target_Taps_Reproduce_Upstream_Main_Hidden_At_Every_Position()
    {
        using CpuBackend cpu = new();
        float[] tapped = DeepSeekV41DSparkFixture.RunTarget(cpu, DeepSeekV41DSparkFixture.Ids);
        DeepSeekV41DSparkFixture.AssertClose(DeepSeekV41DSparkFixture.MainHidden, tapped, "main_hidden");
    }

    [Fact]
    public void Each_Decode_Draft_Matches_Upstream_From_A_Fresh_Seed()
    {
        using CpuBackend cpu = new();
        DeepSeekV41DSpark dspark = DeepSeekV41DSparkFixture.BuildDSpark(cpu);
        int[] ids = DeepSeekV41DSparkFixture.Ids;
        int width = DeepSeekV41DSparkFixture.Width;
        float[] upstream = DeepSeekV41DSparkFixture.MainHidden;
        foreach ((int pos, JsonElement expected) in DeepSeekV41DSparkFixture.Drafts())
        {
            DeepSeekV41DSparkState state = dspark.CreateState(DeepSeekV41DSparkFixture.MaxTokens);
            dspark.Seed(upstream.AsSpan(0, pos * width), pos, state);
            DeepSeekV41DSparkFixture.Compare(dspark.Draft(ids[pos], upstream.AsSpan(pos * width, width), pos, state), expected, pos);
        }
    }

    [Fact]
    public void Drafts_From_The_Target_Taps_Match_Upstream_At_Every_Decode_Step()
    {
        using CpuBackend cpu = new();
        DeepSeekV41DSpark dspark = DeepSeekV41DSparkFixture.BuildDSpark(cpu);
        int[] ids = DeepSeekV41DSparkFixture.Ids;
        int width = DeepSeekV41DSparkFixture.Width;
        float[] tapped = DeepSeekV41DSparkFixture.RunTarget(cpu, ids);
        foreach ((int pos, JsonElement expected) in DeepSeekV41DSparkFixture.Drafts())
        {
            DeepSeekV41DSparkState state = dspark.CreateState(DeepSeekV41DSparkFixture.MaxTokens);
            // the window holds the committed rows before pos; the draft reads pos's own tap, the row of the current token
            dspark.Seed(tapped.AsSpan(0, pos * width), pos, state);
            DeepSeekV41DSparkFixture.Compare(dspark.Draft(ids[pos], tapped.AsSpan(pos * width, width), pos, state), expected, pos);
        }
    }
}
