using HartsyInference.Core.Backends;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cpu.Tests.Attention;

/// <summary>The attention conformance cases the CPU serves, against the contract reference: F32, short sequences. A case
/// the CPU's legacy entry cannot carry must be refused, not answered wrongly.</summary>
public sealed class CpuAttentionConformanceTests(ITestOutputHelper output)
{
    /// <summary>F32 rows short enough for the Unit lane.</summary>
    public static IEnumerable<object[]> Rows =>
        AttnCatalog.Rows(["cpu"], c => c.Seq is not AttnSeq.Long4k && (c.Sq <= 96 || c.Flash));

    [Theory]
    [MemberData(nameof(Rows))]
    public void Attention_Matches_The_Contract(string kind, string caseId)
    {
        AttnCase c = AttnCatalog.Get(caseId);
        using CpuBackend backend = new();
        AttnInputs inputs = AttnInputs.Create(c);
        AttnOutcome outcome = AttnInvoker.Run(backend, c, inputs, out float[]? actual, out Exception? refusal);
        output.WriteLine($"{kind} {c.Id}: {outcome}{(refusal is null ? "" : " - " + refusal.Message)}");
        if (outcome == AttnOutcome.Unsupported) return;
        Assert.Equal(AttnInvoker.Expected(kind, c), outcome);
        if (outcome != AttnOutcome.Ran) return;
        int rowWidth = c.Layout == AttnLayout.TokenMajor ? c.Heads * c.HeadDim : c.HeadDim;
        AttnAssert.Agrees(AttnOracle.CheckedRows(c, actual!), AttnOracle.Expected(c, inputs), rowWidth,
            AttnTolerance.For(AttnTolerance.Worst(kind, c)), because: $"on cpu {c.Id}");
    }
}
