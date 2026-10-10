using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using Xunit;
using static HartsyInference.Cpu.Tests.Moe.MoeTestData;

namespace HartsyInference.Cpu.Tests.Moe;

public sealed class MoeReferenceTests
{
    private static void AssertBitEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"element {i}: expected {expected[i]:R} got {actual[i]:R}");
    }

    private static (int[] Idx, float[] W) Route(float[] logits, int e, in MoeRouteArgs args, float[]? bias = null,
        float[]? altBias = null, int[]? kinds = null)
    {
        int tokens = logits.Length / e;
        using Tensor l = F32(logits, tokens, e);
        using Tensor idx = EmptyI32(tokens, args.TopK);
        using Tensor w = EmptyF32(tokens, args.TopK);
        using Tensor? b = bias is null ? null : F32(bias, e);
        using Tensor? a = altBias is null ? null : F32(altBias, e);
        using Tensor? k = kinds is null ? null : I32(kinds, tokens);
        using CpuBackend cpu = new();
        cpu.MoeRoute(idx, w, l, args, b, a, k);
        return (ReadI32(idx), ReadF32(w));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Softmax_Route_Is_Bit_Identical_To_The_Moe_Test_Port(bool renormalize)
    {
        const int Tokens = 9, E = 8, K = 2;
        float[] logits = Random(Tokens * E, 11, 2.5f);
        int[] expIdx = new int[Tokens * K];
        float[] expW = new float[Tokens * K];
        for (int t = 0; t < Tokens; t++) MoeRoutingPorts.SoftmaxRoute(logits, t, E, K, renormalize, expIdx, expW);

        (int[] idx, float[] w) = Route(logits, E, new MoeRouteArgs(E, K, MoeRouteScoring.Softmax, Renormalize: renormalize));

        Assert.Equal(expIdx, idx);
        AssertBitEqual(expW, w);
    }

    [Fact]
    public void Sigmoid_Bias_Grouped_Route_Is_Bit_Identical_To_The_Moe_Test_Port()
    {
        const int Tokens = 12, E = 32, K = 4, Groups = 8, Kept = 4;
        float[] logits = Random(Tokens * E, 21, 3f);
        float[] bias = Random(E, 22, 0.4f);
        int[] expIdx = new int[Tokens * K];
        float[] expW = new float[Tokens * K];
        for (int t = 0; t < Tokens; t++)
            MoeRoutingPorts.GroupRoute(logits, bias, t, E, K, Groups, Kept, 2.5f, expIdx, expW);

        MoeRouteArgs args = new(E, K, MoeRouteScoring.Sigmoid, Groups, Kept, 0f, true, 1e-20f, 2.5f);
        (int[] idx, float[] w) = Route(logits, E, args, bias);

        Assert.Equal(expIdx, idx);
        AssertBitEqual(expW, w);
    }

    [Fact]
    public void SqrtSoftplus_Route_Matches_The_Upstream_Torch_Gate_Fixture()
    {
        string path = Path.Combine(RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures",
            "moe_route_sqrtsoftplus.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement r = doc.RootElement;
        int tokens = r.GetProperty("tokens").GetInt32(), e = r.GetProperty("experts").GetInt32(), k = r.GetProperty("topk").GetInt32();
        float[] Floats(string n) => r.GetProperty(n).EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
        int[] Ints(string n) => r.GetProperty(n).EnumerateArray().Select(v => v.GetInt32()).ToArray();

        MoeRouteArgs args = new(e, k, MoeRouteScoring.SqrtSoftplus, Renormalize: true, RenormEpsilon: 1e-20f,
            Scale: (float)r.GetProperty("routeScale").GetDouble(), LogitDivisor: (float)r.GetProperty("gateTemp").GetDouble());
        (int[] idx, float[] w) = Route(Floats("logits"), e, args, Floats("bias"), Floats("biasVl"), Ints("tokenKinds"));

        Assert.Equal(Ints("indices"), idx);
        float[] expected = Floats("weights");
        Assert.Equal(tokens * k, w.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - w[i]) <= 1e-6f, $"weight {i}: torch {expected[i]:R} vs {w[i]:R}");
    }

    [Fact]
    public void Exact_Ties_Resolve_To_The_Lowest_Index()
    {
        // Equal logits give equal scores everywhere, so the pick order is index order.
        float[] logits = new float[2 * 8];
        (int[] idx, float[] w) = Route(logits, 8, new MoeRouteArgs(8, 3, MoeRouteScoring.Softmax, Renormalize: true));
        Assert.Equal(new[] { 0, 1, 2, 0, 1, 2 }, idx);
        Assert.All(w, v => Assert.Equal(1f / 3f, v, 6));

        // A bias tie between experts 2 and 5 must also go to 2 first, with unequal raw scores kept out of the weights.
        float[] bias = { 0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f };
        float[] flat = new float[8];
        (int[] tieIdx, _) = Route(flat, 8, new MoeRouteArgs(8, 2, MoeRouteScoring.Sigmoid), bias);
        Assert.Equal(new[] { 2, 5 }, tieIdx);
    }

    [Fact]
    public void K_Equal_To_E_And_K_Of_One_Are_Accepted_And_Out_Of_Range_K_Throws()
    {
        float[] logits = Random(4 * 6, 5);
        (int[] all, float[] w) = Route(logits, 6, new MoeRouteArgs(6, 6, MoeRouteScoring.Softmax, Renormalize: true));
        for (int t = 0; t < 4; t++)
        {
            Assert.Equal(6, all.Skip(t * 6).Take(6).Distinct().Count());
            Assert.Equal(1f, w.Skip(t * 6).Take(6).Sum(), 5);
        }
        Route(logits, 6, new MoeRouteArgs(6, 1, MoeRouteScoring.Softmax));
        Assert.Throws<ArgumentOutOfRangeException>(() => Route(logits, 6, new MoeRouteArgs(6, 0, MoeRouteScoring.Softmax)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Route(logits, 6, new MoeRouteArgs(6, 7, MoeRouteScoring.Softmax)));
    }

    [Fact]
    public void Route_Rejects_Inconsistent_Group_Configuration_And_Missing_Token_Kinds()
    {
        float[] logits = Random(2 * 8, 6);
        Assert.Throws<ArgumentException>(() =>
            Route(logits, 8, new MoeRouteArgs(8, 2, MoeRouteScoring.Sigmoid, GroupCount: 3, GroupsKept: 1)));
        Assert.Throws<ArgumentException>(() =>
            Route(logits, 8, new MoeRouteArgs(8, 2, MoeRouteScoring.Sigmoid), new float[8], new float[8], null));
    }

    [Fact]
    public void Dispatch_Groups_Pairs_By_Expert_In_Stable_Order_And_Leaves_Empty_Experts_Empty()
    {
        // Three tokens x k=2 over 5 experts; expert 3 is never chosen and expert 4 is chosen once.
        int[] pairs = { 2, 0, 0, 2, 2, 4 };
        using Tensor topk = I32(pairs, 3, 2);
        using Tensor counts = EmptyI32(5), offsets = EmptyI32(6), perm = EmptyI32(6), slot = EmptyI32(6);
        using CpuBackend cpu = new();
        cpu.MoeBuildDispatch(counts, offsets, perm, slot, topk, 5);

        Assert.Equal(new[] { 2, 0, 3, 0, 1 }, ReadI32(counts));
        Assert.Equal(new[] { 0, 2, 2, 5, 5, 6 }, ReadI32(offsets));
        Assert.Equal(new[] { 0, 1, 0, 1, 2, 2 }, ReadI32(perm));
        Assert.Equal(new[] { 2, 0, 1, 3, 4, 5 }, ReadI32(slot));
    }

    [Fact]
    public void Dispatch_Drops_Out_Of_Range_Experts_And_Pads_The_Tail()
    {
        using Tensor topk = I32(new[] { 1, -1, 7, 1 }, 2, 2);
        using Tensor counts = EmptyI32(2), offsets = EmptyI32(3), perm = EmptyI32(4), slot = EmptyI32(4);
        using CpuBackend cpu = new();
        cpu.MoeBuildDispatch(counts, offsets, perm, slot, topk, 2);

        Assert.Equal(new[] { 0, 2 }, ReadI32(counts));
        Assert.Equal(new[] { 0, 0, 2 }, ReadI32(offsets));
        Assert.Equal(new[] { 0, 1, -1, -1 }, ReadI32(perm));
        Assert.Equal(new[] { 0, -1, -1, 1 }, ReadI32(slot));
    }

    [Fact]
    public void Combine_Sums_Weighted_Expert_Rows_In_Slot_Order_And_Honors_Accumulate_And_Dropped_Slots()
    {
        const int H = 3, K = 2;
        float[] rows = { 1, 2, 3, 10, 20, 30, 100, 200, 300 };
        using Tensor expertOut = F32(rows, 3, H);
        using Tensor slots = I32(new[] { 2, 0, 1, -1 }, 2, K);
        using Tensor weights = F32(new[] { 0.5f, 2f, 3f, 99f }, 2, K);
        using Tensor output = F32(new float[] { 1, 1, 1, 1, 1, 1 }, 2, H);
        using CpuBackend cpu = new();

        cpu.MoeCombine(output, expertOut, slots, weights, K, accumulate: false);
        Assert.Equal(new float[] { 52, 104, 156, 30, 60, 90 }, ReadF32(output));

        cpu.MoeCombine(output, expertOut, slots, weights, K, accumulate: true);
        Assert.Equal(new float[] { 104, 208, 312, 60, 120, 180 }, ReadF32(output));
    }

    [Fact]
    public void Combine_Rejects_Ragged_Expert_Rows_And_Out_Of_Range_Slots()
    {
        using CpuBackend cpu = new();
        using Tensor slots = I32(new[] { 0, 1 }, 1, 2);
        using Tensor weights = F32(new[] { 1f, 1f }, 1, 2);
        using Tensor output = F32(new float[] { 0, 0, 0 }, 1, 3);
        using Tensor ragged = F32(new float[] { 1, 2, 3, 4 }, 4);
        Assert.Throws<ArgumentException>(() => cpu.MoeCombine(output, ragged, slots, weights, 2, false));
        using Tensor rows = F32(new float[] { 1, 2, 3 }, 1, 3);
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.MoeCombine(output, rows, slots, weights, 2, false));
        using Tensor zeroWidth = EmptyF32(2, 0);
        Assert.Throws<ArgumentException>(() => cpu.MoeCombine(zeroWidth, rows, slots, weights, 2, false));
    }

    [Fact]
    public void TopK_Orders_By_Value_Then_Lowest_Index_And_Can_Sort_By_Index()
    {
        float[] row = { 1f, 5f, 5f, 3f, 5f, 2f, 3f };
        using Tensor input = F32(row, 1, 7);
        using Tensor values = EmptyF32(1, 4), indices = EmptyI32(1, 4);
        using CpuBackend cpu = new();

        cpu.TopKLastDim(values, indices, input, 4);
        Assert.Equal(new[] { 1, 2, 4, 3 }, ReadI32(indices));
        Assert.Equal(new[] { 5f, 5f, 5f, 3f }, ReadF32(values));

        cpu.TopKLastDim(values, indices, input, 4, sortByIndex: true);
        Assert.Equal(new[] { 1, 2, 3, 4 }, ReadI32(indices));
        Assert.Equal(new[] { 5f, 5f, 3f, 5f }, ReadF32(values));
    }

}
