using System.Text;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Audio;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Tests.Audio;

/// <summary>Layout detection and loading of a user-selected MiniMax-Music-3 flow transformer, on synthetic headers and tiny tensors (no weights).</summary>
public sealed class MiniMaxMusic3FlowCheckpointTests : IDisposable
{
    private const int Hidden = 16;
    private const int InDim = 32;
    private readonly string _folder = Directory.CreateTempSubdirectory("mm3-flow-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Fact]
    public void Detect_Names_Each_Layout()
    {
        Assert.Equal(MiniMaxMusic3DitLayout.DiffusersDit, MiniMaxMusic3DitLayouts.Detect(DiffusersKeys()));
        Assert.Equal(MiniMaxMusic3DitLayout.DiffusersDitWithConditionEncoder,
            MiniMaxMusic3DitLayouts.Detect(DiffusersKeys().Concat(ConditionKeys())));
        Assert.Equal(MiniMaxMusic3DitLayout.ConditionEncoderOnly, MiniMaxMusic3DitLayouts.Detect(ConditionKeys()));
        Assert.Equal(MiniMaxMusic3DitLayout.ComfyDit, MiniMaxMusic3DitLayouts.Detect(
            ["diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight", "cond_layer_logits"]));
        Assert.Equal(MiniMaxMusic3DitLayout.FusedQkvCppDit, MiniMaxMusic3DitLayouts.Detect(["dit.blk.0.attn_qkv.weight"]));
        Assert.Equal(MiniMaxMusic3DitLayout.Unrecognized, MiniMaxMusic3DitLayouts.Detect(["model.layers.0.self_attn.q_proj.weight"]));
    }

    [Fact]
    public void Open_Diffusers_File_Borrows_Every_Tensor_And_Has_No_Condition()
    {
        // The Hartsy repack: the official transformer, one fp32 file, an arbitrary file name.
        string path = WriteSafetensors("minimaxmusic3-base_fp32.safetensors", DiffusersDit(DType.F32));
        using MiniMaxMusic3FlowWeights weights = MiniMaxMusic3FlowCheckpoint.Open(path);
        Assert.Null(weights.ConditionEncoder);
        Assert.Equal(DiffusersKeys().Count(), weights.Transformer.Count);
        Assert.Equal(DType.F32, weights.Transformer["time_proj.weight"].DType);
        Assert.Equal(new TensorShape(Hidden, InDim, 1), weights.Transformer["preprocess_conv.weight"].Shape);
    }

    [Fact]
    public void Open_Diffusers_Fp16_Widens_The_Host_Read_Tensor()
    {
        string path = WriteSafetensors("dit_fp16.safetensors", DiffusersDit(DType.F16));
        using MiniMaxMusic3FlowWeights weights = MiniMaxMusic3FlowCheckpoint.Open(path);
        Assert.Equal(DType.F32, weights.Transformer["time_proj.weight"].DType);
        Assert.Equal(DType.F16, weights.Transformer["transformer_blocks.0.attn.to_q.weight"].DType);
        // F16 norm scales and biases corrupt the DiT once its weights are resident on the device.
        Assert.Equal(DType.F32, weights.Transformer["transformer_blocks.0.norm1.weight"].DType);
        Assert.Equal(DType.F32, weights.Transformer["transformer_blocks.0.ff_in.bias"].DType);
    }

    [Fact]
    public void Open_Merged_File_Splits_The_Condition_Encoder_Out_In_F32()
    {
        Dictionary<string, Tensor> all = DiffusersDit(DType.F16);
        foreach (string key in ConditionKeys())
        {
            all[key] = Filled(new TensorShape(Hidden, InDim), DType.F16, 1f);
        }
        using MiniMaxMusic3FlowWeights weights = MiniMaxMusic3FlowCheckpoint.Open(WriteSafetensors("merged.safetensors", all));
        Assert.NotNull(weights.ConditionEncoder);
        Assert.Equal(ConditionKeys().Count(), weights.ConditionEncoder!.Count);
        Assert.All(weights.ConditionEncoder.Values, tensor => Assert.Equal(DType.F32, tensor.DType));
        Assert.DoesNotContain("proj.weight", weights.Transformer.Keys);
    }

    [Fact]
    public void Open_Shards_Loads_The_Whole_Set_From_A_Shard_The_Index_Or_The_Folder()
    {
        Dictionary<string, Tensor> all = DiffusersDit(DType.F32);
        Dictionary<string, Tensor> first = [];
        Dictionary<string, Tensor> second = [];
        foreach ((string key, Tensor tensor) in all)
        {
            (key.StartsWith("transformer_blocks.", StringComparison.Ordinal) ? second : first)[key] = tensor;
        }
        string folder = Directory.CreateDirectory(Path.Combine(_folder, "transformer")).FullName;
        string shard1 = Path.Combine(folder, "diffusion_pytorch_model-00001-of-00002.safetensors");
        string shard2 = Path.Combine(folder, "diffusion_pytorch_model-00002-of-00002.safetensors");
        WriteSafetensors(shard1, first);
        WriteSafetensors(shard2, second);
        Dictionary<string, string> map = first.Keys.ToDictionary(key => key, _ => Path.GetFileName(shard1));
        foreach (string key in second.Keys)
        {
            map[key] = Path.GetFileName(shard2);
        }
        string index = Path.Combine(folder, "diffusion_pytorch_model.safetensors.index.json");
        File.WriteAllText(index, JsonSerializer.Serialize(new { metadata = new { }, weight_map = map }));

        foreach (string entry in new[] { shard2, index, folder })
        {
            using MiniMaxMusic3FlowWeights weights = MiniMaxMusic3FlowCheckpoint.Open(entry);
            Assert.Equal(all.Count, weights.Transformer.Count);
        }
    }

    [Fact]
    public void Open_ComfyDit_Splits_Fused_Qkv_Q_First()
    {
        Tensor fused = new Tensor(new TensorShape(3 * Hidden, InDim), DType.F32);
        for (int i = 0; i < fused.Shape.ElementCount; i++)
        {
            fused.AsSpan<float>()[i] = i;
        }
        Dictionary<string, Tensor> comfy = new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight"] = fused,
            ["cond_layer_logits"] = Filled(new TensorShape(8), DType.F16, 0f),
        };
        using MiniMaxMusic3FlowWeights weights = MiniMaxMusic3FlowCheckpoint.Open(WriteSafetensors("comfy.safetensors", comfy));
        float[] v = weights.Transformer["transformer_blocks.0.attn.to_v.weight"].AsReadOnlySpan<float>().ToArray();
        Assert.Equal(2f * Hidden * InDim, v[0]);
        Assert.Equal(DType.F32, weights.ConditionEncoder!["layer_weight_logits"].DType);
    }

    [Fact]
    public void Open_ComfyDit_Int8_ConvRot_Dequantizes_Through_The_Companion_Fold()
    {
        const int hidden = 64;
        const int inDim = 256;
        float[] values = new float[3 * hidden * inDim];
        Random random = new Random(7);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(random.NextDouble() * 2 - 1);
        }
        using Tensor source = new Tensor(new TensorShape(3 * hidden, inDim), DType.F32);
        values.CopyTo(source.AsSpan<float>());
        (Tensor packed, Tensor rowScale) = Int8ConvRotCodec.QuantizeFromF32(Copy(source), 256);
        string descriptor = "{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": 256}";
        Tensor blob = new Tensor(new TensorShape(Encoding.UTF8.GetByteCount(descriptor)), DType.U8);
        Encoding.UTF8.GetBytes(descriptor).CopyTo(blob.AsSpan<byte>());
        const string layer = "diffusion_transformer.transformer.layers.0.self_attn.to_qkv";
        Dictionary<string, Tensor> file = new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            [$"{layer}.weight"] = packed, [$"{layer}.weight_scale"] = rowScale, [$"{layer}.comfy_quant"] = blob,
        };

        using MiniMaxMusic3FlowWeights weights = MiniMaxMusic3FlowCheckpoint.Open(WriteSafetensors("int8.safetensors", file));
        Tensor query = weights.Transformer["transformer_blocks.0.attn.to_q.weight"];
        Assert.Equal(DType.BF16, query.DType);
        using Tensor widened = query.CastTo(DType.F32);
        float worst = 0f;
        for (int i = 0; i < hidden * inDim; i++)
        {
            worst = Math.Max(worst, Math.Abs(widened.AsReadOnlySpan<float>()[i] - values[i]));
        }
        Assert.True(worst < 0.03f, $"int8 ConvRot round trip drifted by {worst}");
    }

    [Fact]
    public void Open_Gguf_Quantized_Dit_Relabels_Shapes_And_Keeps_Quantized_Projections()
    {
        string path = Path.Combine(_folder, "dit-q8_0.gguf");
        GgufQuantizer.ConvertDictionaryToGguf(DiffusersDit(DType.F32), path, GgufQuantPolicy.Q8_0, "mm3-dit");
        using MiniMaxMusic3FlowWeights weights = MiniMaxMusic3FlowCheckpoint.Open(path);
        Tensor query = weights.Transformer["transformer_blocks.0.attn.to_q.weight"];
        Assert.Equal(DType.Q8_0, query.DType);
        Assert.Equal(new TensorShape(Hidden, InDim), query.Shape);
        Assert.Equal(new TensorShape(Hidden, InDim, 1), weights.Transformer["preprocess_conv.weight"].Shape);
        Assert.Equal(DType.F32, weights.Transformer["time_proj.weight"].DType);
        Assert.Equal(DType.F32, weights.Transformer["transformer_blocks.0.norm1.bias"].DType);
    }

    [Fact]
    public void Open_Refuses_A_Foreign_File_And_Names_What_It_Found()
    {
        string path = WriteSafetensors("language_model.safetensors", new Dictionary<string, Tensor>
        {
            ["model.layers.0.self_attn.q_proj.weight"] = Filled(new TensorShape(Hidden, InDim), DType.F32, 0f),
        });
        NotSupportedException error = Assert.Throws<NotSupportedException>(() => MiniMaxMusic3FlowCheckpoint.Open(path));
        Assert.Contains("language_model.safetensors", error.Message, StringComparison.Ordinal);
        Assert.Contains("unrecognized tensor set", error.Message, StringComparison.Ordinal);
        Assert.Contains("Supported:", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Open_Refuses_A_Condition_Encoder_Selected_As_The_Transformer()
    {
        Dictionary<string, Tensor> condition = ConditionKeys().ToDictionary(key => key,
            _ => Filled(new TensorShape(Hidden, InDim), DType.F32, 0f));
        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => MiniMaxMusic3FlowCheckpoint.Open(WriteSafetensors("condition.safetensors", condition)));
        Assert.Contains("condition encoder alone", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Open_Refuses_A_Diffusers_File_With_An_Unknown_Tensor()
    {
        Dictionary<string, Tensor> all = DiffusersDit(DType.F32);
        all["transformer_blocks.0.attn.mystery.weight"] = Filled(new TensorShape(Hidden, InDim), DType.F32, 0f);
        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => MiniMaxMusic3FlowCheckpoint.Open(WriteSafetensors("extra.safetensors", all)));
        Assert.Contains("mystery", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Open_Reports_A_Missing_File()
    {
        Assert.Throws<FileNotFoundException>(() => MiniMaxMusic3FlowCheckpoint.Open(Path.Combine(_folder, "absent.safetensors")));
    }

    [Theory]
    [InlineData("q8", "q8_0")]
    [InlineData("q4", "q4_k")]
    [InlineData("q4_k", "q4_k")]
    [InlineData("q4_0", "q4_0")]
    [InlineData("q5", "q5_k")]
    [InlineData("q6_k", "q6_k")]
    [InlineData("MiniMax-Music-3:q8_0", "q8_0")]
    [InlineData("base", null)]
    [InlineData("", null)]
    public void ResolveQuant_Reads_The_Variant_Suffix(string variant, string? expected) =>
        Assert.Equal(expected, MiniMaxMusic3MusicModel.ResolveQuant(variant));

    [Fact]
    public void PrepareTransformer_Leaves_A_Half_Precision_Transformer_Uncached()
    {
        Dictionary<string, Tensor> half = DiffusersDit(DType.F16);
        IReadOnlyDictionary<string, Tensor> prepared = MiniMaxMusic3WeightPolicy.PrepareTransformer(half, "test/repo", "q8_0", out IDisposable? owned);
        Assert.Same(half, prepared);
        Assert.Null(owned);
    }

    private static IEnumerable<string> DiffusersKeys() =>
    [
        "preprocess_conv.weight", "postprocess_conv.weight", "time_proj.weight", "time_embed.linear_1.weight",
        "time_embed.linear_1.bias", "time_embed.linear_2.weight", "time_embed.linear_2.bias", "proj_in.weight", "proj_out.weight",
        "transformer_blocks.0.norm1.weight", "transformer_blocks.0.norm1.bias", "transformer_blocks.0.norm2.weight",
        "transformer_blocks.0.norm2.bias", "transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.to_v.weight", "transformer_blocks.0.attn.to_out.0.weight", "transformer_blocks.0.ff_in.weight",
        "transformer_blocks.0.ff_in.bias", "transformer_blocks.0.ff_out.weight", "transformer_blocks.0.ff_out.bias",
    ];

    private static IEnumerable<string> ConditionKeys() => ["layer_weight_logits", "layer_scale", "proj.weight", "proj.bias"];

    private static Dictionary<string, Tensor> DiffusersDit(DType dtype)
    {
        Dictionary<string, Tensor> dit = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (string key in DiffusersKeys())
        {
            TensorShape shape = key switch
            {
                "preprocess_conv.weight" => new TensorShape(Hidden, InDim, 1),
                "postprocess_conv.weight" => new TensorShape(Hidden, Hidden, 1),
                "time_proj.weight" => new TensorShape(128, 1),
                _ when key.EndsWith(".bias", StringComparison.Ordinal) || key.Contains("norm", StringComparison.Ordinal)
                    => new TensorShape(Hidden),
                _ => new TensorShape(Hidden, InDim),
            };
            dit[key] = Filled(shape, dtype, 0.5f);
        }
        return dit;
    }

    private static Tensor Filled(TensorShape shape, DType dtype, float value)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        tensor.AsSpan<float>().Fill(value);
        if (dtype == DType.F32)
        {
            return tensor;
        }
        using (tensor)
        {
            return tensor.CastTo(dtype);
        }
    }

    private static Tensor Copy(Tensor source)
    {
        Tensor copy = new Tensor(source.Shape, source.DType);
        source.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        return copy;
    }

    private string WriteSafetensors(string name, IReadOnlyDictionary<string, Tensor> tensors)
    {
        string path = Path.IsPathRooted(name) ? name : Path.Combine(_folder, name);
        List<SafeTensorsStreamWriter.Entry> entries = [];
        foreach ((string key, Tensor tensor) in tensors)
        {
            entries.Add(new SafeTensorsStreamWriter.Entry(key, tensor.DType, tensor.Shape, () => (tensor, false)));
        }
        SafeTensorsStreamWriter.Write(path, entries);
        return path;
    }
}
