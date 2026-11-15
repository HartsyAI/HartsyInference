using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.GpuBenchmarks.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Evidence labels must describe the fixture that executes, before any GPU work is possible.</summary>
public sealed class AttentionEvidenceContractTests
{
    [Fact]
    public void AcceptsEveryCatalogArmWithMatchingFixtureAndFrozenBranch()
    {
        foreach (AttentionArm arm in AttentionArms.All)
            WithCall(arm, call => AttentionEvidenceContract.Validate(arm, in call, "public", "registered-branch"));
    }

    [Theory]
    [InlineData("fa-def")]
    [InlineData("fa-f16kv")]
    [InlineData("fa-splitforce")]
    [InlineData("fa-splitoff")]
    public void RejectsDenseFixturesUnderFlashLabels(string name)
    {
        AttentionArm arm = AttentionArms.Get(name);
        WithCall(arm, call => Reject(arm, call with { Route = AttentionRoute.Dense }));
    }

    [Fact]
    public void RejectsFlashFixtureUnderDenseLabel()
    {
        AttentionArm arm = AttentionArms.Get("def-f32");
        WithCall(arm, call => Reject(arm, call with { Route = AttentionRoute.Flash }));
    }

    [Theory]
    [InlineData("query")]
    [InlineData("key")]
    [InlineData("value")]
    [InlineData("output")]
    public void RejectsMismatchedOperandTypes(string operand)
    {
        AttentionArm arm = AttentionArms.Get("def-f32");
        WithCall(arm, call =>
        {
            using Tensor half = new(call.Query.Storage.Shape, DType.F16);
            AttentionOperand wrong = AttentionOperand.HeadMajor(half);
            AttentionCall changed = operand switch
            {
                "query" => call with { Query = wrong }, "key" => call with { Key = wrong },
                "value" => call with { Value = wrong }, _ => call with { Output = wrong },
            };
            Reject(arm, changed);
        });
    }

    [Fact]
    public void RejectsDifferentLegacyAssertion()
    {
        AttentionArm arm = AttentionArms.Get("cast-f16");
        WithCall(arm, call => Reject(arm, call with { Numerics = call.Numerics with { LegacyAssertedBounded = false } }));
    }

    [Fact]
    public void RejectsMissingExplicitNumericalGrant()
    {
        AttentionArm arm = AttentionArms.Get("bf16-cast");
        WithCall(arm, call => Reject(arm, call with { Numerics = AttentionNumerics.Exact }));
    }

    [Theory]
    [InlineData("public", null)]
    [InlineData("public", "")]
    [InlineData("resident", " ")]
    public void TimingCannotLearnItsExpectedBranchFromTheCandidate(string boundary, string? expected)
    {
        AttentionArm arm = AttentionArms.Get("def-f32");
        WithCall(arm, call => Assert.Throws<ArgumentException>(() =>
            AttentionEvidenceContract.Validate(arm, in call, boundary, expected)));
    }

    [Fact]
    public void DiscoveryCanRunWithoutAnExpectedBranchAndLeavesTheFixtureUnchanged()
    {
        AttentionArm arm = AttentionArms.Get("fa-f16kv");
        WithCall(arm, call =>
        {
            AttentionCall before = call;
            AttentionEvidenceContract.Validate(arm, in call, "observe", null);
            Assert.Equal(before, call);
        });
    }

    private static void Reject(AttentionArm arm, AttentionCall call) =>
        Assert.Throws<InvalidDataException>(() => AttentionEvidenceContract.Validate(arm, in call, "observe", null));

    private static void WithCall(AttentionArm arm, Action<AttentionCall> check)
    {
        TensorShape shape = new(1, 1, 1, 64);
        using Tensor q = new(shape, arm.Dtype);
        using Tensor k = new(shape, arm.KvDtype);
        using Tensor v = new(shape, arm.KvDtype);
        using Tensor output = new(shape, arm.Dtype);
        check(new AttentionCall
        {
            Query = AttentionOperand.HeadMajor(q), Key = AttentionOperand.HeadMajor(k),
            Value = AttentionOperand.HeadMajor(v), Output = AttentionOperand.HeadMajor(output), Scale = 0.125f,
            Route = arm.Entry == "flash" ? AttentionRoute.Flash : AttentionRoute.Dense,
            Numerics = arm.Grant ?? AttentionNumerics.FromLegacy(arm.AllowF16),
        });
    }
}
