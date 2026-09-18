using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the contract Flux.1 and SD3 gained when they moved onto <see cref="CheckpointSource"/>: the
/// converters read a container's view of a checkpoint, so they refuse a dictionary whose quantization companions are
/// still loose — folding after a rename pairs nothing and leaves the weight running at real/scale, which renders as
/// noise rather than failing — and they read the key layouts a GGUF repack actually ships.</summary>
public sealed class Flux1Sd3ContainerConversionTests : IDisposable
{
    private readonly string _tempDir;

    public Flux1Sd3ContainerConversionTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"flux1_sd3_container_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); }
        catch (IOException) { /* a mmap the OS has not released yet is not a test failure */ }
    }

    private static Tensor Scalar(float value)
    {
        Tensor tensor = new Tensor(new TensorShape(1), DType.F32);
        tensor.AsSpan<float>()[0] = value;
        return tensor;
    }

    private string Write(Dictionary<string, Tensor> weights, string name = "model.safetensors")
    {
        string path = Path.Combine(_tempDir, name);
        SafeTensorsWriter.Save(path, weights);
        return path;
    }

    [Fact]
    public void FluxConvert_RefusesACheckpointWhoseCompanionsAreStillLoose()
    {
        Dictionary<string, Tensor> raw = new(StringComparer.Ordinal)
        {
            ["double_blocks.0.img_attn.qkv.weight"] = new Tensor(new TensorShape(9, 3), DType.F8E4M3),
            ["double_blocks.0.img_attn.qkv.scale_weight"] = Scalar(0.0195f),
        };
        try
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(() => FluxCheckpointConverter.Convert(raw));
            Assert.Contains("double_blocks.0.img_attn.qkv.scale_weight", error.Message, StringComparison.Ordinal);
            Assert.Contains("CheckpointSource.Open", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            foreach (Tensor tensor in raw.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void FluxConvert_AcceptsTheFoldedViewTheContainerHandsIt()
    {
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal)
        {
            // A double_blocks key is what flips the converter into its BFL branch; the bare spelling is the one a
            // published Flux.1 GGUF uses (city96's flux1-dev-*.gguf drops the model.diffusion_model. wrapper).
            ["double_blocks.0.img_attn.norm.query_norm.scale"] = new Tensor(new TensorShape(4), DType.F32),
            ["img_in.weight"] = new Tensor(new TensorShape(4, 8), DType.F8E4M3),
            ["img_in.scale_weight"] = Scalar(0.0195f),
        };
        try
        {
            using CheckpointSource source = CheckpointSource.Open(Write(weights));
            FluxCheckpointConverter.ConvertedWeights converted = FluxCheckpointConverter.Convert(source.Weights);

            // The scale rides the renamed weight: img_in.weight became x_embedder.weight, and img_in.scale_weight
            // has no rename rule of its own, which is exactly why the container folds before the converter runs.
            Assert.Equal(0.0195f, converted.Transformer["x_embedder.weight"].Fp8ScaleFactor);
            Assert.Contains("transformer_blocks.0.attn.norm_q.weight", converted.Transformer.Keys);
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Sd3Convert_RefusesACheckpointWhoseCompanionsAreStillLoose()
    {
        Dictionary<string, Tensor> raw = new(StringComparer.Ordinal)
        {
            ["model.diffusion_model.x_embedder.proj.weight"] = new Tensor(new TensorShape(4, 8), DType.F8E4M3),
            ["model.diffusion_model.x_embedder.proj.scale_weight"] = Scalar(0.0312f),
        };
        try
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(() => Sd3CheckpointConverter.Convert(raw));
            Assert.Contains("model.diffusion_model.x_embedder.proj.scale_weight", error.Message, StringComparison.Ordinal);
            Assert.Contains("CheckpointSource.Open", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            foreach (Tensor tensor in raw.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Sd3Convert_AcceptsTheFoldedViewTheContainerHandsIt()
    {
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal)
        {
            ["model.diffusion_model.x_embedder.proj.weight"] = new Tensor(new TensorShape(4, 8), DType.F8E4M3),
            ["model.diffusion_model.x_embedder.proj.scale_weight"] = Scalar(0.0312f),
        };
        try
        {
            using CheckpointSource source = CheckpointSource.Open(Write(weights));
            Sd3CheckpointConverter.ConvertedWeights converted = Sd3CheckpointConverter.Convert(source.Weights);

            Assert.Equal(0.0312f, converted.Transformer["pos_embed.proj.weight"].Fp8ScaleFactor);
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Sd3Convert_ReadsAnMmditSavedWithoutTheStabilityWrapper()
    {
        // The layout every published SD3/SD3.5 GGUF ships: city96's sd3.5_large-Q4_0.gguf declares
        // general.architecture "sd3" and names its tensors bare. Requiring the wrapper dropped all of them.
        Dictionary<string, Tensor> weights = BareMmdit();
        try
        {
            using CheckpointSource source = CheckpointSource.Open(Write(weights, "bare.safetensors"));
            Sd3CheckpointConverter.ConvertedWeights converted = Sd3CheckpointConverter.Convert(source.Weights);

            AssertMmditConverted(converted);
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Sd3Convert_StillReadsAnMmditUnderTheStabilityWrapper()
    {
        Dictionary<string, Tensor> bare = BareMmdit();
        Dictionary<string, Tensor> wrapped = new(bare.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in bare) wrapped["model.diffusion_model." + entry.Key] = entry.Value;
        try
        {
            using CheckpointSource source = CheckpointSource.Open(Write(wrapped, "wrapped.safetensors"));
            Sd3CheckpointConverter.ConvertedWeights converted = Sd3CheckpointConverter.Convert(source.Weights);

            AssertMmditConverted(converted);
        }
        finally
        {
            foreach (Tensor tensor in bare.Values) tensor.Dispose();
        }
    }

    /// <summary>One MMDiT block plus the embedders around it, in the bare Stability naming, sized so the fused QKV splits evenly.</summary>
    private static Dictionary<string, Tensor> BareMmdit()
    {
        return new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["pos_embed"] = new Tensor(new TensorShape(1, 16, 8), DType.F32),
            ["x_embedder.proj.weight"] = new Tensor(new TensorShape(8, 16, 2, 2), DType.F32),
            ["context_embedder.weight"] = new Tensor(new TensorShape(8, 12), DType.F32),
            ["t_embedder.mlp.0.weight"] = new Tensor(new TensorShape(8, 4), DType.F32),
            ["y_embedder.mlp.0.weight"] = new Tensor(new TensorShape(8, 4), DType.F32),
            ["joint_blocks.0.x_block.attn.qkv.weight"] = new Tensor(new TensorShape(24, 8), DType.F32),
            ["joint_blocks.0.x_block.attn.proj.weight"] = new Tensor(new TensorShape(8, 8), DType.F32),
            ["joint_blocks.0.context_block.attn.proj.weight"] = new Tensor(new TensorShape(8, 8), DType.F32),
            ["final_layer.linear.weight"] = new Tensor(new TensorShape(16, 8), DType.F32),
        };
    }

    private static void AssertMmditConverted(Sd3CheckpointConverter.ConvertedWeights converted)
    {
        Assert.Contains("pos_embed.pos_embed", converted.Transformer.Keys);
        Assert.Contains("pos_embed.proj.weight", converted.Transformer.Keys);
        Assert.Contains("context_embedder.weight", converted.Transformer.Keys);
        Assert.Contains("time_text_embed.timestep_embedder.linear_1.weight", converted.Transformer.Keys);
        Assert.Contains("time_text_embed.text_embedder.linear_1.weight", converted.Transformer.Keys);
        Assert.Contains("transformer_blocks.0.attn.to_q.weight", converted.Transformer.Keys);
        Assert.Contains("transformer_blocks.0.attn.to_out.0.weight", converted.Transformer.Keys);
        Assert.Contains("transformer_blocks.0.attn.to_add_out.weight", converted.Transformer.Keys);
        Assert.Contains("proj_out.weight", converted.Transformer.Keys);
        // The split QKV is the converter's own allocation; nothing else in the dict owns memory this test made.
        foreach (string key in new[] { "transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.attn.to_k.weight", "transformer_blocks.0.attn.to_v.weight" })
        {
            converted.Transformer[key].Dispose();
        }
    }
}
