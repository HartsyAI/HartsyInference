using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Tests the Matrix-Game 3.0 key routing: ActionModule/Plücker keys bypass the Wan rename table (whose
/// <c>cross_attn</c>/<c>modulation</c> rules would corrupt them), core Wan keys reuse the verbatim rename rules,
/// distilled bundles slice the student prefix and drop critic/EMA copies, and the shape inferencer resolves the
/// 5120/40 vs 3072/30 config contradiction from the weights.</summary>
public unsafe class MatrixGame3CheckpointConverterTests
{
    [Theory]
    // Matrix-Game-specific keys pass through (action_module normalized to action_model).
    [InlineData("blocks.0.action_model.t_qkv.weight", "blocks.0.action_model.t_qkv.weight")]
    [InlineData("blocks.0.action_module.t_qkv.weight", "blocks.0.action_model.t_qkv.weight")]
    [InlineData("blocks.3.action_model.mouse_attn_q.weight", "blocks.3.action_model.mouse_attn_q.weight")]
    [InlineData("blocks.3.action_model.keyboard_attn_kv.weight", "blocks.3.action_model.keyboard_attn_kv.weight")]
    [InlineData("patch_embedding_wancamctrl.weight", "patch_embedding_wancamctrl.weight")]
    [InlineData("c2ws_hidden_states_layer1.weight", "c2ws_hidden_states_layer1.weight")]
    // Core Wan keys get the original→diffusers renames.
    [InlineData("blocks.0.self_attn.q.weight", "blocks.0.attn1.to_q.weight")]
    [InlineData("blocks.0.cross_attn.o.weight", "blocks.0.attn2.to_out.0.weight")]
    [InlineData("blocks.0.modulation", "blocks.0.scale_shift_table")]
    [InlineData("head.head.weight", "proj_out.weight")]
    [InlineData("time_embedding.0.weight", "condition_embedder.time_embedder.linear_1.weight")]
    // Student-slice prefix strip.
    [InlineData("student.blocks.0.self_attn.q.weight", "blocks.0.attn1.to_q.weight")]
    [InlineData("generator.blocks.0.action_model.t_qkv.weight", "blocks.0.action_model.t_qkv.weight")]
    public void MapKey_RoutesMatrixGameAndWanKeys(string key, string expected)
    {
        Assert.Equal(expected, MatrixGame3CheckpointConverter.MapKey(key, fromOriginalNaming: true));
    }

    [Theory]
    [InlineData("critic.blocks.0.self_attn.q.weight")]
    [InlineData("ema.blocks.0.self_attn.q.weight")]
    [InlineData("fake_score.head.head.weight")]
    public void MapKey_DropsNonStudentCopies(string key)
    {
        Assert.Null(MatrixGame3CheckpointConverter.MapKey(key, fromOriginalNaming: true));
    }

    [Fact]
    public void Convert_SlicesStudentAndCollectsActionBlocks()
    {
        Dictionary<string, Tensor> raw = new()
        {
            ["student.blocks.0.self_attn.q.weight"] = Scalar(),
            ["student.blocks.0.action_model.t_qkv.weight"] = Scalar(),
            ["student.blocks.5.action_model.t_qkv.weight"] = Scalar(),
            ["student.head.head.weight"] = Scalar(),
            ["critic.blocks.0.self_attn.q.weight"] = Scalar(),
        };
        MatrixGame3CheckpointConverter.ConvertedWeights converted = MatrixGame3CheckpointConverter.Convert(raw);

        Assert.Equal(4, converted.Transformer.Count);
        Assert.True(converted.Transformer.ContainsKey("blocks.0.attn1.to_q.weight"));
        Assert.True(converted.Transformer.ContainsKey("proj_out.weight"));
        Assert.Equal(new[] { 0, 5 }, converted.ActionBlocks);
    }

    [Fact]
    public void InferShape_ReadsRealDimsFromWeights()
    {
        // The TI2V-5B-shaped reading: dim 3072, 30 layers, ffn 14336 → 24 heads.
        Dictionary<string, Tensor> w = new()
        {
            ["patch_embedding.weight"] = Zeros(3072, 48, 1, 2, 2),
            ["blocks.0.ffn.net.0.proj.weight"] = Zeros(14336, 3072),
            ["blocks.0.attn1.to_q.weight"] = Zeros(3072, 3072),
            ["blocks.29.attn1.to_q.weight"] = Zeros(3072, 3072),
        };
        MatrixGame3CheckpointConverter.InferredShape shape = MatrixGame3CheckpointConverter.InferShape(w);
        Assert.Equal(3072, shape.Dim);
        Assert.Equal(30, shape.NumLayers);
        Assert.Equal(14336, shape.FfnDim);
        Assert.Equal(24, shape.NumHeads);
        Assert.Equal(48, shape.InChannels);
    }

    private static Tensor Scalar() => new(new TensorShape(1), DType.F32);

    private static Tensor Zeros(params int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        return new Tensor(new TensorShape(d), DType.F32);
    }
}
