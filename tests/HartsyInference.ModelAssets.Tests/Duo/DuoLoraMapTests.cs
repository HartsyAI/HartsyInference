using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Duo;

public sealed class DuoLoraMapTests
{
    [Theory]
    [InlineData("transformer_blocks.7.attn.to_q", "blocks.7.attn.qkv_proj.weight", 0, 3, false)]
    [InlineData("transformer_blocks.7.attn.to_k", "blocks.7.attn.qkv_proj.weight", 1, 3, false)]
    [InlineData("transformer_blocks.49.attn.to_v", "blocks.49.attn.qkv_proj.weight", 2, 3, false)]
    [InlineData("transformer_blocks.0.attn.to_out.0", "blocks.0.attn.out_proj.weight", 0, 1, false)]
    [InlineData("transformer_blocks.3.ff.net.0.proj", "blocks.3.mlp.fc1.weight", 0, 1, true)]
    [InlineData("transformer_blocks.3.ff.net.2", "blocks.3.mlp.fc2.weight", 0, 1, false)]
    [InlineData("token_refiner.refiner_blocks.1.attn.to_k", "token_refiner.blocks.1.attn.qkv_proj.weight", 1, 3, false)]
    [InlineData("token_refiner.refiner_blocks.0.ff.net.0.proj", "token_refiner.blocks.0.mlp.fc1.weight", 0, 1, true)]
    [InlineData("proj_in", "video_patch_proj.weight", 0, 1, false)]
    [InlineData("audio_proj_in", "audio_patch_proj.weight", 0, 1, false)]
    [InlineData("context_embedder", "condition_proj.weight", 0, 1, false)]
    [InlineData("proj_out", "final_layer.video_out.weight", 0, 1, false)]
    [InlineData("audio_proj_out", "final_layer.audio_out.weight", 0, 1, false)]
    [InlineData("transformer_blocks.2.attn.to_q.base_layer", "blocks.2.attn.qkv_proj.weight", 0, 3, false)]
    public void TryMap_ResolvesTheDiffusersNameToTheBackboneWeight(string module, string key, int slot, int slots, bool swap)
    {
        Assert.True(DuoLoraKeyMap.TryMap(module, out DuoLoraTarget target));
        Assert.Equal(new DuoLoraTarget(key, slot, slots, swap), target);
    }

    [Theory]
    [InlineData("role_embeddings")]
    [InlineData("transformer_blocks.x.attn.to_q")]
    [InlineData("transformer_blocks.1.attn.add_q_proj")]
    [InlineData("time_embedder.linear_1")]
    [InlineData("norm_out.linear")]
    public void TryMap_RefusesAModuleTheReleasedLoraDoesNotHave(string module)
        => Assert.False(DuoLoraKeyMap.TryMap(module, out _));

    [Fact]
    public void ModuleOf_SplitsTheTensorKey()
    {
        Assert.Equal("transformer_blocks.1.ff.net.2", DuoLoraKeyMap.ModuleOf("transformer_blocks.1.ff.net.2.lora_A.default.weight", out bool a));
        Assert.True(a);
        Assert.Equal("proj_in", DuoLoraKeyMap.ModuleOf("proj_in.lora_B.default.weight", out a));
        Assert.False(a);
        Assert.Null(DuoLoraKeyMap.ModuleOf("role_embeddings.weight", out _));
    }

    [Fact]
    public void QkvLayout_MovesWholeRowsBetweenHeadInterleavedAndBlockOrder()
    {
        // 2 heads x headDim 2: interleaved rows are h0q(0,1) h0k(2,3) h0v(4,5) h1q(6,7) h1k(8,9) h1v(10,11); blocks are Q(h0 h1) K(h0 h1) V(h0 h1).
        using Tensor interleaved = new(new TensorShape(12, 3), DType.F32);
        Span<float> data = interleaved.AsSpan<float>();
        for (int row = 0; row < 12; row++)
            for (int col = 0; col < 3; col++) data[row * 3 + col] = row * 10 + col;

        using Tensor blocks = DuoQkvLayout.ToBlockOrder(interleaved, heads: 2, headDim: 2);

        int[] expectedRows = [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11];
        for (int row = 0; row < 12; row++)
            Assert.Equal(expectedRows[row] * 10f, blocks.AsSpan<float>()[row * 3]);
        using Tensor back = DuoQkvLayout.ToHeadInterleaved(blocks, 2, 2);
        Assert.Equal(data.ToArray(), back.AsSpan<float>().ToArray());
    }

    [Fact]
    public void QkvLayout_KeepsBf16BitsAndRejectsAWrongRowCount()
    {
        using Tensor interleaved = new(new TensorShape(12, 1), DType.BF16);
        for (int i = 0; i < 12; i++) interleaved.AsSpan<ushort>()[i] = (ushort)(0x3f80 + i);
        using Tensor blocks = DuoQkvLayout.ToBlockOrder(interleaved, 2, 2);
        Assert.Equal((ushort)(0x3f80 + 6), blocks.AsSpan<ushort>()[2]);
        using Tensor bad = new(new TensorShape(11, 1), DType.BF16);
        Assert.Throws<ArgumentException>(() => DuoQkvLayout.ToBlockOrder(bad, 2, 2));
    }

    [Fact]
    public void Schedule_ReproducesTheDistinctTimesOfTheTwoStages()
    {
        double[] distill = DuoSchedule.Grid(DuoSchedule.DistillSteps, DuoSchedule.DistillShift);
        Assert.Equal(5, distill.Length);
        Assert.Equal(new[] { 0.0, 0.0625, 1.0 / 6.0, 0.375, 1.0 }, distill, new DoubleTolerance(1e-12));
        double[] causal = DuoSchedule.Grid(DuoSchedule.CausalSteps, DuoSchedule.CausalShift);
        Assert.Equal(51, causal.Length);
        for (int i = 0; i <= 50; i++) Assert.Equal(i / 50.0, causal[i], 12);
        Assert.Equal(51, DuoSchedule.AdalnTimes(50, 1.0).Length);
        Assert.Throws<ArgumentOutOfRangeException>(() => DuoSchedule.Grid(0, 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => DuoSchedule.Grid(4, 0.0));
    }

    private sealed class DoubleTolerance(double tolerance) : IEqualityComparer<double>
    {
        public bool Equals(double x, double y) => Math.Abs(x - y) <= tolerance;

        public int GetHashCode(double obj) => 0;
    }
}
