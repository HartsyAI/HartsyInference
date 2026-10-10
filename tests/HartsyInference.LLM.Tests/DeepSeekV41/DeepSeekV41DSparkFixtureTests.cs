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

}
