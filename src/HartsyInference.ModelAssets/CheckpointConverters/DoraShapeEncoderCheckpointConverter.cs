using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>
/// Typed weights of the Dora-VAE 1.1 shape encoder (the point-cloud encoder of HairDeck's point-cloud model), read from
/// the released checkpoint's <c>shape_model.</c> tensors. The decoder and the KL layers are not read: HairDeck conditions
/// on the encoder's pre-KL tokens only.
/// </summary>
public static class DoraShapeEncoderCheckpointConverter
{
    /// <summary>Width of the encoder (tokens and the attention space).</summary>
    public const int Width = 768;

    /// <summary>Attention heads of every block (head dimension 64).</summary>
    public const int Heads = 12;

    /// <summary>Number of self-attention blocks after the cross-attention.</summary>
    public const int SelfBlocks = 8;

    /// <summary>Width of the point embedding: the Fourier features of xyz (3 + 6 * 8) and the normal (3).</summary>
    public const int PointEmbeddingWidth = 54;

    /// <summary>A pre-norm cross-attention block: <c>ln_1</c> on the queries, <c>ln_2</c> on the data, <c>ln_3</c> before the MLP.</summary>
    public sealed record CrossBlock(
        Tensor Ln1Weight, Tensor Ln1Bias,
        Tensor Ln2Weight, Tensor Ln2Bias,
        Tensor Ln3Weight, Tensor Ln3Bias,
        HairDeckSetDitCheckpointConverter.Linear Query,
        HairDeckSetDitCheckpointConverter.Linear KeyValue,
        HairDeckSetDitCheckpointConverter.Linear Output,
        HairDeckSetDitCheckpointConverter.Linear MlpIn,
        HairDeckSetDitCheckpointConverter.Linear MlpOut);

    /// <summary>A pre-norm self-attention block: <c>ln_1</c> before the fused qkv projection, <c>ln_2</c> before the MLP.</summary>
    public sealed record SelfBlock(
        Tensor Ln1Weight, Tensor Ln1Bias,
        Tensor Ln2Weight, Tensor Ln2Bias,
        HairDeckSetDitCheckpointConverter.Linear Qkv,
        HairDeckSetDitCheckpointConverter.Linear Output,
        HairDeckSetDitCheckpointConverter.Linear MlpIn,
        HairDeckSetDitCheckpointConverter.Linear MlpOut);

    /// <summary>The encoder: two point-embedding projections, two cross-attention blocks (the coarse and the sharp
    /// branch), <see cref="SelfBlocks"/> self-attention blocks and the output LayerNorm.</summary>
    public sealed record Weights(
        HairDeckSetDitCheckpointConverter.Linear InputProj,
        HairDeckSetDitCheckpointConverter.Linear InputProjSharp,
        CrossBlock Cross,
        CrossBlock CrossSharp,
        IReadOnlyList<SelfBlock> Blocks,
        Tensor LnPostWeight,
        Tensor LnPostBias);

    /// <summary>Reads the encoder. <paramref name="prefix"/> is the name of the shape model in the checkpoint
    /// (<c>"state_dict.shape_model."</c> in the released Lightning checkpoint, <c>""</c> after conversion).</summary>
    public static Weights Convert(IReadOnlyDictionary<string, Tensor> all, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(all);
        HairDeckSetDitCheckpointConverter.Linear inputProj = HairDeckSetDitCheckpointConverter.Lin(all, prefix + "encoder.input_proj", Width, PointEmbeddingWidth);
        HairDeckSetDitCheckpointConverter.Linear inputProjSharp = HairDeckSetDitCheckpointConverter.Lin(all, prefix + "encoder.input_proj1", Width, PointEmbeddingWidth);
        CrossBlock cross = ReadCross(all, prefix + "encoder.cross_attn");
        CrossBlock crossSharp = ReadCross(all, prefix + "encoder.cross_attn1");
        List<SelfBlock> blocks = new(SelfBlocks);
        for (int i = 0; i < SelfBlocks; i++)
        {
            blocks.Add(ReadSelf(all, $"{prefix}encoder.self_attn.resblocks.{i}"));
        }
        return new Weights(
            inputProj,
            inputProjSharp,
            cross,
            crossSharp,
            blocks,
            HairDeckTensorReader.Take(all, prefix + "encoder.ln_post.weight", Width),
            HairDeckTensorReader.Take(all, prefix + "encoder.ln_post.bias", Width));
    }

    private static CrossBlock ReadCross(IReadOnlyDictionary<string, Tensor> all, string p) => new(
        HairDeckTensorReader.Take(all, p + ".ln_1.weight", Width), HairDeckTensorReader.Take(all, p + ".ln_1.bias", Width),
        HairDeckTensorReader.Take(all, p + ".ln_2.weight", Width), HairDeckTensorReader.Take(all, p + ".ln_2.bias", Width),
        HairDeckTensorReader.Take(all, p + ".ln_3.weight", Width), HairDeckTensorReader.Take(all, p + ".ln_3.bias", Width),
        new HairDeckSetDitCheckpointConverter.Linear(HairDeckTensorReader.Take(all, p + ".attn.c_q.weight", Width, Width), null),
        new HairDeckSetDitCheckpointConverter.Linear(HairDeckTensorReader.Take(all, p + ".attn.c_kv.weight", 2 * Width, Width), null),
        HairDeckSetDitCheckpointConverter.Lin(all, p + ".attn.c_proj", Width, Width),
        HairDeckSetDitCheckpointConverter.Lin(all, p + ".mlp.c_fc", 4 * Width, Width),
        HairDeckSetDitCheckpointConverter.Lin(all, p + ".mlp.c_proj", Width, 4 * Width));

    private static SelfBlock ReadSelf(IReadOnlyDictionary<string, Tensor> all, string p) => new(
        HairDeckTensorReader.Take(all, p + ".ln_1.weight", Width), HairDeckTensorReader.Take(all, p + ".ln_1.bias", Width),
        HairDeckTensorReader.Take(all, p + ".ln_2.weight", Width), HairDeckTensorReader.Take(all, p + ".ln_2.bias", Width),
        new HairDeckSetDitCheckpointConverter.Linear(HairDeckTensorReader.Take(all, p + ".attn.c_qkv.weight", 3 * Width, Width), null),
        HairDeckSetDitCheckpointConverter.Lin(all, p + ".attn.c_proj", Width, Width),
        HairDeckSetDitCheckpointConverter.Lin(all, p + ".mlp.c_fc", 4 * Width, Width),
        HairDeckSetDitCheckpointConverter.Lin(all, p + ".mlp.c_proj", Width, 4 * Width));
}
