using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Matrix-Game 2.0 checkpoint handling: the shared Matrix-Game key routing (original Wan naming, incl. the
/// I2V <c>img_emb</c> projection renames) plus the MG2 per-variant action-shape inference from weights.</summary>
public class MatrixGame2CheckpointConverterTests
{
    [Fact]
    public void Convert_RoutesI2VImageEmbedderAndActionKeys()
    {
        Dictionary<string, Tensor> raw = new()
        {
            ["blocks.0.self_attn.q.weight"] = Scalar(),
            ["blocks.0.cross_attn.k.weight"] = Scalar(),
            ["img_emb.proj.0.weight"] = Scalar(),
            ["img_emb.proj.1.weight"] = Scalar(),
            ["img_emb.proj.3.weight"] = Scalar(),
            ["img_emb.proj.4.weight"] = Scalar(),
            ["blocks.0.action_model.keyboard_embed.0.weight"] = Zeros(128, 2),   // GTA: 2-key
            ["blocks.0.action_model.mouse_mlp.0.weight"] = Zeros(8, 8),
            ["blocks.7.action_model.keyboard_embed.0.weight"] = Zeros(128, 2),
            ["head.head.weight"] = Scalar(),
        };
        MatrixGame3CheckpointConverter.ConvertedWeights converted = MatrixGame2CheckpointConverter.Convert(raw);

        Assert.True(converted.Transformer.ContainsKey("blocks.0.attn1.to_q.weight"));
        Assert.True(converted.Transformer.ContainsKey("blocks.0.attn2.to_k.weight"));
        Assert.True(converted.Transformer.ContainsKey("condition_embedder.image_embedder.norm1.weight"));
        Assert.True(converted.Transformer.ContainsKey("condition_embedder.image_embedder.ff.net.0.proj.weight"));
        Assert.True(converted.Transformer.ContainsKey("condition_embedder.image_embedder.ff.net.2.weight"));
        Assert.True(converted.Transformer.ContainsKey("condition_embedder.image_embedder.norm2.weight"));
        Assert.True(converted.Transformer.ContainsKey("proj_out.weight"));
        Assert.Equal(new[] { 0, 7 }, converted.ActionBlocks);

        MatrixGame2CheckpointConverter.InferredActionShape shape = MatrixGame2CheckpointConverter.InferActionShape(converted);
        Assert.Equal(2, shape.KeyboardDim);
        Assert.True(shape.HasMouse);
    }

    [Fact]
    public void InferActionShape_DetectsTempleRunKeyboardOnly()
    {
        Dictionary<string, Tensor> raw = new()
        {
            ["blocks.0.action_model.keyboard_embed.0.weight"] = Zeros(128, 7),
            ["blocks.0.self_attn.q.weight"] = Scalar(),
        };
        MatrixGame3CheckpointConverter.ConvertedWeights converted = MatrixGame2CheckpointConverter.Convert(raw);
        MatrixGame2CheckpointConverter.InferredActionShape shape = MatrixGame2CheckpointConverter.InferActionShape(converted);
        Assert.Equal(7, shape.KeyboardDim);
        Assert.False(shape.HasMouse);
    }

    private static Tensor Scalar() => new(new TensorShape(1), DType.F32);

    private static Tensor Zeros(int a, int b) => new(new TensorShape(a, b), DType.F32);
}
