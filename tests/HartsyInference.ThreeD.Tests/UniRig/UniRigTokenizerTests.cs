using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Tests.UniRig;

public sealed class UniRigTokenizerTests
{
    [Fact]
    public void Vocabulary_MatchesTheShippedCheckpointLayout()
    {
        Assert.Equal(267, UniRigTokenizer.VocabSize);
        Assert.Equal(256, UniRigTokenizer.BranchToken);
        Assert.Equal(257, UniRigTokenizer.BosToken);
        Assert.Equal(258, UniRigTokenizer.EosToken);
        Assert.Equal(259, UniRigTokenizer.PadToken);
        Assert.Equal(260, UniRigTokenizer.SpringToken);
        Assert.Equal(261, UniRigTokenizer.PartToken("body"));
        Assert.Equal(262, UniRigTokenizer.PartToken("hand"));
        Assert.Equal(263, UniRigTokenizer.ClsNoneToken);
        Assert.Equal(264, UniRigTokenizer.ClsToken("vroid"));
        Assert.Equal(265, UniRigTokenizer.ClsToken("mixamo"));
        Assert.Equal(266, UniRigTokenizer.ClsToken("articulationxl"));
        Assert.Equal(UniRigTokenizer.ClsNoneToken, UniRigTokenizer.ClsToken("unknown"));
    }

    [Theory]
    [InlineData(-1f, 0)]
    [InlineData(1f, 255)]
    [InlineData(0f, 128)]
    [InlineData(-5f, 0)]
    [InlineData(5f, 255)]
    public void Discretize_ClipsAndRoundsToBins(float value, int expected) => Assert.Equal(expected, UniRigTokenizer.Discretize(value));

    [Fact]
    public void Undiscretize_ReturnsBinCentres()
    {
        // Upstream rounds on the way in and adds half a bin on the way out, so a round trip can be off by up to one bin (2/256).
        Assert.Equal(-1f + 1f / 256f, UniRigTokenizer.Undiscretize(0), 6);
        Assert.Equal(1f - 1f / 256f, UniRigTokenizer.Undiscretize(255), 6);
        for (int bin = 0; bin < 256; bin += 17)
        {
            float centre = UniRigTokenizer.Undiscretize(bin);
            Assert.True(Math.Abs(centre - (-1f + 2f * (bin + 0.5f) / 256f)) < 1e-6f);
        }
    }

    [Fact]
    public void RoundTrip_ChainAndBranch_RecoversJointsParentsAndNames()
    {
        // root(0) -> spine(1) -> head(2); branch: arm(3) hangs off spine(1).
        float[] joints = [0f, 0f, -0.5f, 0f, 0f, 0f, 0f, 0f, 0.6f, 0.4f, 0f, 0.1f];
        float[] parents = [0f, 0f, -0.5f, 0f, 0f, -0.5f, 0f, 0f, 0f, 0f, 0f, 0f];
        UniRigTokenizeInput input = new()
        {
            Joints = joints,
            ParentJoints = parents,
            Branch = [false, false, false, true],
            Cls = "articulationxl",
            PartsBias = new Dictionary<int, string?> { [0] = null },
        };

        int[] tokens = UniRigTokenizer.Tokenize(input);
        Assert.Equal(UniRigTokenizer.BosToken, tokens[0]);
        Assert.Equal(UniRigTokenizer.ArticulationXlToken, tokens[1]);
        Assert.Equal(UniRigTokenizer.SpringToken, tokens[2]);
        Assert.Equal(UniRigTokenizer.EosToken, tokens[^1]);
        Assert.Equal(2 + 1 + 3 * 3 + 7 + 1, tokens.Length);

        UniRigSkeleton skeleton = UniRigTokenizer.Detokenize(tokens);
        Assert.Equal(4, skeleton.JointCount);
        Assert.Equal([-1, 0, 1, 1], skeleton.Parents);
        Assert.Equal(["bone_0", "bone_1", "bone_2", "bone_3"], skeleton.Names);
        Assert.Single(skeleton.Parts);
        Assert.Null(skeleton.Parts[0]);
        for (int i = 0; i < joints.Length; i++)
        {
            Assert.True(Math.Abs(skeleton.Joints[i] - joints[i]) <= 2f / 256f, $"joint component {i}: {skeleton.Joints[i]} vs {joints[i]}");
        }
    }

    [Fact]
    public void Detokenize_TailsFollowTheUpstreamRules()
    {
        float[] joints = [0f, 0f, -0.5f, 0f, 0f, 0f, 0f, 0f, 0.6f, 0.4f, 0f, 0.1f];
        float[] parents = [0f, 0f, -0.5f, 0f, 0f, -0.5f, 0f, 0f, 0f, 0f, 0f, 0f];
        int[] tokens = UniRigTokenizer.Tokenize(new UniRigTokenizeInput
        {
            Joints = joints,
            ParentJoints = parents,
            Branch = [false, false, false, true],
            Cls = "articulationxl",
        });
        UniRigSkeleton s = UniRigTokenizer.Detokenize(tokens);

        // Joint 0 has one child: its tail is that child's head.
        Assert.Equal(s.Joints[3], s.Tails[0]);
        Assert.Equal(s.Joints[5], s.Tails[2]);
        // Joint 2 is a leaf: head + 0.5 * (head - parent head).
        Assert.Equal(s.Joints[6] + 0.5f * (s.Joints[6] - s.Joints[3]), s.Tails[6], 5);
        Assert.Equal(s.Joints[8] + 0.5f * (s.Joints[8] - s.Joints[5]), s.Tails[8], 5);
    }

    [Fact]
    public void Detokenize_AcceptsTrailingPadding()
    {
        int[] tokens = UniRigTokenizer.Tokenize(new UniRigTokenizeInput
        {
            Joints = [0f, 0f, 0f],
            ParentJoints = [0f, 0f, 0f],
            Branch = [false],
            Cls = "articulationxl",
        });
        int[] padded = [.. tokens, UniRigTokenizer.PadToken, UniRigTokenizer.PadToken];
        Assert.Equal(1, UniRigTokenizer.Detokenize(padded).JointCount);
    }

    [Fact]
    public void Detokenize_ParentIsTheNearestEarlierJointToTheParentPosition()
    {
        // Joint 2 names a parent position equal to joint 0's head although joint 1 precedes it.
        int[] tokens =
        [
            UniRigTokenizer.BosToken, UniRigTokenizer.ArticulationXlToken,
            100, 100, 100,
            120, 120, 120,
            UniRigTokenizer.BranchToken, 100, 100, 100, 140, 140, 140,
            UniRigTokenizer.EosToken,
        ];
        UniRigSkeleton s = UniRigTokenizer.Detokenize(tokens);
        Assert.Equal([-1, 0, 0], s.Parents);
    }

    [Theory]
    [MemberData(nameof(MalformedSequences))]
    public void Detokenize_RejectsMalformedSequences(int[] tokens) =>
        Assert.Throws<HartsyInferenceException>(() => UniRigTokenizer.Detokenize(tokens));

    public static TheoryData<int[]> MalformedSequences() => new()
    {
        { [] },
        { [UniRigTokenizer.EosToken] },
        { [UniRigTokenizer.BosToken, UniRigTokenizer.ArticulationXlToken, 1, 2, 3] },
        { [UniRigTokenizer.BosToken, UniRigTokenizer.ArticulationXlToken, 1, 2, UniRigTokenizer.EosToken] },
        { [UniRigTokenizer.BosToken, UniRigTokenizer.ArticulationXlToken, UniRigTokenizer.EosToken] },
        { [UniRigTokenizer.BosToken, UniRigTokenizer.ArticulationXlToken, 1, 2, 3, UniRigTokenizer.BranchToken, 4, 5, 6, UniRigTokenizer.EosToken] },
        { [UniRigTokenizer.BosToken, 300, 1, 2, 3, UniRigTokenizer.EosToken] },
    };
}
