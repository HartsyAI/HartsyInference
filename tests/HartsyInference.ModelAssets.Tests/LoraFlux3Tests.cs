using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>FLUX.3 LoRA support: detection in each spelling ahead of the Flux arms that share its joint-block roots, the
/// diffusers/kohya/BFL name resolution, and merges proving a fused-qkv adapter lands in the split projections and a
/// diffusers <c>norm_out.linear</c> adapter lands with its <c>[scale | shift]</c> halves reordered.</summary>
public sealed unsafe class LoraFlux3Tests(ITestOutputHelper output) : IDisposable
{
    private const int Rank = 2;
    private const int Hidden = 4;

    private readonly string _tempDir = Directory.CreateTempSubdirectory("lora_flux3_tests_").FullName;

    [Theory]
    [InlineData("transformer.mode_blocks.image.0.attn.to_q.lora_A.weight")]
    [InlineData("transformer.txt_mode_blocks.3.attn.to_out.0.lora_B.weight")]
    [InlineData("transformer.single_transformer_blocks.7.ff.linear_in.lora_A.weight")]
    [InlineData("transformer.x_embedders.image.lora_A.weight")]
    [InlineData("diffusion_model.content_mode_blocks.image_cond.0.qkv.lora_A.weight")]
    [InlineData("diffusion_model.single_blocks.2.q_proj.lora_down.weight")]
    [InlineData("single_stream_modulations.txt.lin.lora_A.weight")]
    [InlineData("lora_unet_content_mode_blocks_image_0_mlp_in.lora_down.weight")]
    [InlineData("lora_unet_single_blocks_5_attn_out.lora_up.weight")]
    public void Detect_ClaimsEveryFlux3Spelling(string key)
    {
        Assert.Equal(LoraFormat.Flux3, LoraFormatDetector.Detect(Descriptors(key)));
    }

    [Fact]
    public void Detect_LeavesFluxKeysSharingTheJointRootsAlone()
    {
        // Diffusers Flux.1 single blocks have the same attention names as FLUX.3's joint blocks.
        Assert.Equal(LoraFormat.DiffusersFlux, LoraFormatDetector.Detect(Descriptors(
            "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
            "transformer.single_transformer_blocks.0.proj_mlp.lora_A.weight")));
        // Flux.2's final norm shares diffusers' `norm_out.linear` name, so it is no evidence either.
        Assert.Equal(LoraFormat.DiffusersFlux, LoraFormatDetector.Detect(Descriptors(
            "transformer.norm_out.linear.lora_A.weight",
            "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight")));
        Assert.Equal(LoraFormat.ComfyBflDit, LoraFormatDetector.Detect(Descriptors(
            "diffusion_model.single_blocks.3.linear1.lora_down.weight")));
        Assert.Equal(LoraFormat.KohyaFlux, LoraFormatDetector.Detect(Descriptors(
            "lora_unet_single_blocks_3_linear1.lora_down.weight")));
        // A scalar alone is not evidence of an adapter.
        Assert.Equal(LoraFormat.Unknown, LoraFormatDetector.Detect(Descriptors("transformer.mode_blocks.image.0.attn.to_q.alpha")));
    }

    [Theory]
    [InlineData("transformer.mode_blocks.image_cond.1.attn.to_k", "content_mode_blocks.image_cond.1.k_proj", true)]
    [InlineData("transformer.txt_mode_blocks.0.attn.to_out.0", "txt_mode_blocks.0.attn_out", true)]
    [InlineData("transformer.single_transformer_blocks.4.ff.linear_out", "single_blocks.4.mlp_out", true)]
    [InlineData("transformer.early_stream_modulations.image_cond.linear", "early_stream_modulations.image_cond.lin", true)]
    [InlineData("transformer.context_embedder", "txt_in", true)]
    [InlineData("transformer.target_timestep_embedder.linear_2", "ref_time_in.out_layer", true)]
    [InlineData("transformer.norm_out.linear", Flux3KeyNames.FinalAdaLn, true)]
    [InlineData("diffusion_model.txt_mode_blocks.0.q_proj", "txt_mode_blocks.0.q_proj", false)]
    [InlineData("single_blocks.1.mlp_in", "single_blocks.1.mlp_in", false)]
    [InlineData("lora_unet_content_mode_blocks_image_cond_12_v_proj", "content_mode_blocks.image_cond.12.v_proj", false)]
    [InlineData("lora_unet_early_stream_modulations_txt_lin", "early_stream_modulations.txt.lin", false)]
    [InlineData("lora_unet_ref_time_in_in_layer", "ref_time_in.in_layer", false)]
    [InlineData("lora_unet_final_layer_image_adaLN_modulation_1", Flux3KeyNames.FinalAdaLn, false)]
    public void TryResolve_MapsEachSpellingToTheBflModule(string root, string expected, bool diffusers)
    {
        Assert.True(Flux3LoraMapper.TryResolve(root, out string module, out bool diffusersSpelling));
        Assert.Equal(expected, module);
        Assert.Equal(diffusers, diffusersSpelling);
    }

    [Theory]
    [InlineData("transformer.transformer_blocks.0.attn.to_q")]
    [InlineData("transformer.single_transformer_blocks.0.proj_mlp")]
    [InlineData("lora_unet_single_blocks_0_linear1")]
    [InlineData("lora_unet_content_mode_blocks_image_x_q_proj")]
    [InlineData("text_encoder.layers.0.self_attn.q_proj")]
    public void TryResolve_RefusesModulesFlux3DoesNotHave(string root)
    {
        Assert.False(Flux3LoraMapper.TryResolve(root, out _, out _));
    }

    [Fact]
    public void Load_SimpleTunerExtract_ParsesDiffusersNamesWithTheirAlpha()
    {
        Dictionary<string, Tensor> tensors = [];
        foreach (string module in new[] { "mode_blocks.image.0.attn.to_q", "txt_mode_blocks.0.attn.to_out.0", "single_transformer_blocks.1.attn.to_v" })
        {
            tensors[$"transformer.{module}.lora_A.weight"] = Filled(DType.F16, [Rank, Hidden], _ => 0.5f);
            tensors[$"transformer.{module}.lora_B.weight"] = Filled(DType.F16, [Hidden, Rank], _ => 0.25f);
            tensors[$"transformer.{module}.alpha"] = Filled(DType.F16, [], _ => 4f);
        }
        string path = Save("simpletuner_extract", tensors, new Dictionary<string, string> { ["format"] = "simpletuner-peft-lora-extract" });

        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.Flux3, file.Format);
        Assert.Equal(
            ["content_mode_blocks.image.0.q_proj.weight", "single_blocks.1.v_proj.weight", "txt_mode_blocks.0.attn_out.weight"],
            file.Layers.Select(l => l.TargetKey).Order(StringComparer.Ordinal));
        Assert.All(file.Layers, l => Assert.Equal(LoraTarget.Transformer, l.Target));
        // The 0-dim F16 alpha is read, not defaulted to rank: scale = 4 / 2.
        Assert.All(file.Layers, l => Assert.Equal(2f, l.Delta.Scale));
    }

    [Fact]
    public void Load_PeftDora_IsRefusedRatherThanAppliedAsPlainLora()
    {
        // The magnitude carries no role suffix the vocabulary knows, so without the refusal the A/B pair would load
        // alone and merge as an ordinary LoRA: wrong weights, no error.
        Dictionary<string, Tensor> tensors = new()
        {
            ["transformer.mode_blocks.image.0.attn.to_q.lora_A.weight"] = Filled(DType.F32, [Rank, Hidden], _ => 0.5f),
            ["transformer.mode_blocks.image.0.attn.to_q.lora_B.weight"] = Filled(DType.F32, [Hidden, Rank], _ => 0.25f),
            ["transformer.mode_blocks.image.0.attn.to_q.lora_magnitude_vector.weight"] = Filled(DType.F32, [Hidden], _ => 1f),
        };
        string path = Save("peft_dora", tensors);

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => LoraFile.Load(path));
        Assert.Contains("lora_magnitude_vector", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Merge_DiffusersNormOut_LandsWithItsHalvesSwapped()
    {
        // B rows [scale | shift] = [0.25 | 0.75]; A = 0.5 and alpha = rank, so the delta is 2·0.5·B = B.
        Dictionary<string, Tensor> tensors = new()
        {
            ["transformer.norm_out.linear.lora_A.weight"] = Filled(DType.F32, [Rank, Hidden], _ => 0.5f),
            ["transformer.norm_out.linear.lora_B.weight"] = Filled(DType.F32, [2 * Hidden, Rank], i => i < Hidden * Rank ? 0.25f : 0.75f),
            ["transformer.mode_blocks.image.0.attn.to_q.lora_A.weight"] = Filled(DType.F32, [Rank, Hidden], _ => 0.5f),
            ["transformer.mode_blocks.image.0.attn.to_q.lora_B.weight"] = Filled(DType.F32, [Hidden, Rank], _ => 0.25f),
        };
        string path = Save("norm_out", tensors);
        Dictionary<string, Tensor> weights = new()
        {
            [Flux3KeyNames.FinalAdaLn + ".weight"] = Filled(DType.F32, [2 * Hidden, Hidden], _ => 1f),
            ["content_mode_blocks.image.0.q_proj.weight"] = Filled(DType.F32, [Hidden, Hidden], _ => 1f),
        };

        using LoraStack stack = Merge(path, weights, expectedMerged: 2);

        // BFL rows are [shift | scale]: the shift delta (0.75) comes first.
        AssertRows(weights[Flux3KeyNames.FinalAdaLn + ".weight"], row => row < Hidden ? 1.75f : 1.25f);
        AssertRows(weights["content_mode_blocks.image.0.q_proj.weight"], _ => 1.25f);
    }

    [Fact]
    public void Merge_FusedQkvAdapter_SpreadsOverTheSplitProjections()
    {
        // Up rows [q | k | v] = [0.25 | 0.5 | 0.75]; A = 0.5, alpha = rank, so each delta row is 2·0.5·B = B.
        Dictionary<string, Tensor> tensors = new()
        {
            ["diffusion_model.content_mode_blocks.image_cond.0.qkv.lora_A.weight"] = Filled(DType.F32, [Rank, Hidden], _ => 0.5f),
            ["diffusion_model.content_mode_blocks.image_cond.0.qkv.lora_B.weight"] =
                Filled(DType.F32, [3 * Hidden, Rank], i => 0.25f * (1 + i / (Hidden * Rank))),
            ["lora_unet_single_blocks_0_mlp_out.lora_down.weight"] = Filled(DType.F32, [Rank, Hidden], _ => 0.5f),
            ["lora_unet_single_blocks_0_mlp_out.lora_up.weight"] = Filled(DType.F32, [Hidden, Rank], _ => 0.25f),
        };
        string path = Save("fused_qkv", tensors);
        Dictionary<string, Tensor> weights = new()
        {
            ["content_mode_blocks.image_cond.0.q_proj.weight"] = Filled(DType.F32, [Hidden, Hidden], _ => 1f),
            ["content_mode_blocks.image_cond.0.k_proj.weight"] = Filled(DType.F32, [Hidden, Hidden], _ => 1f),
            ["content_mode_blocks.image_cond.0.v_proj.weight"] = Filled(DType.F32, [Hidden, Hidden], _ => 1f),
            ["single_blocks.0.mlp_out.weight"] = Filled(DType.F32, [Hidden, Hidden], _ => 1f),
        };

        using LoraStack stack = Merge(path, weights, expectedMerged: 4);

        AssertRows(weights["content_mode_blocks.image_cond.0.q_proj.weight"], _ => 1.25f);
        AssertRows(weights["content_mode_blocks.image_cond.0.k_proj.weight"], _ => 1.5f);
        AssertRows(weights["content_mode_blocks.image_cond.0.v_proj.weight"], _ => 1.75f);
        AssertRows(weights["single_blocks.0.mlp_out.weight"], _ => 1.25f);
    }

    /// <summary>The published extract must load as FLUX.3 and every layer must land on a weight the converted full
    /// build carries — the synthetic tests pin the rules, this pins the file the world ships.</summary>
    [Trait("Category", "Integration")]
    [Trait("Category", "RealWeights")]
    [Fact]
    public void RealDistilledExtract_EveryLayerTargetsAConvertedKleinWeight()
    {
        if (!RealWeightGate.Require(output.WriteLine, TestPaths.Flux3.DistilledLoraExtract, TestPaths.Flux3.Klein)) return;

        using LoraFile file = LoraFile.Load(TestPaths.Flux3.DistilledLoraExtract);
        Assert.Equal(LoraFormat.Flux3, file.Format);
        Assert.NotEmpty(file.Layers);

        using CheckpointSource source = CheckpointSource.Open(TestPaths.Flux3.Klein);
        Dictionary<string, Tensor> converted = Flux3CheckpointConverter.ConvertTransformer(source.Weights);
        try
        {
            string[] missing = file.Layers.Select(l => l.TargetKey).Where(k => !converted.ContainsKey(k)).ToArray();
            Assert.True(missing.Length == 0, $"{missing.Length} targets absent, e.g. {string.Join(", ", missing.Take(5))}");
            output.WriteLine($"{file.Layers.Count} layers, all on converted Klein weights.");
        }
        finally
        {
            foreach ((string key, Tensor tensor) in converted)
            {
                if (!ReferenceEquals(tensor, source.Weights.GetValueOrDefault(key))) tensor.Dispose();
            }
        }
    }

    /// <summary>Merges the file into <paramref name="weights"/>. The returned stack owns the merged tensors, so it must
    /// outlive every read of them.</summary>
    private static LoraStack Merge(string path, Dictionary<string, Tensor> weights, int expectedMerged)
    {
        Dictionary<string, Tensor> originals = new Dictionary<string, Tensor>(weights);
        using CpuBackend backend = new CpuBackend();
        LoraStack stack = new LoraStack();
        stack.AddFromPath(path, strength: 1.0f);
        Assert.Equal(expectedMerged, stack.ApplyTo(weights, LoraTarget.Transformer, backend));
        DisposeAll(originals);
        return stack;
    }

    private static void AssertRows(Tensor weight, Func<int, float> expectedForRow)
    {
        float* p = (float*)weight.DataPointer;
        long rows = weight.Shape[0];
        long cols = weight.Shape[1];
        for (int row = 0; row < rows; row++)
        {
            float expected = expectedForRow(row);
            for (int col = 0; col < cols; col++)
            {
                float actual = p[row * cols + col];
                Assert.True(MathF.Abs(actual - expected) < 1e-5f, $"[{row},{col}] = {actual}, expected {expected}");
            }
        }
    }

    private static Tensor Filled(DType dtype, long[] shape, Func<int, float> value)
    {
        Tensor tensor = new Tensor(new TensorShape(shape), dtype);
        if (dtype == DType.F16)
        {
            Span<Half> span = tensor.AsSpan<Half>();
            for (int i = 0; i < span.Length; i++) span[i] = (Half)value(i);
        }
        else
        {
            Span<float> span = tensor.AsSpan<float>();
            for (int i = 0; i < span.Length; i++) span[i] = value(i);
        }
        return tensor;
    }

    private string Save(string name, Dictionary<string, Tensor> tensors, IReadOnlyDictionary<string, string>? metadata = null)
    {
        string path = Path.Combine(_tempDir, name + ".safetensors");
        SafeTensorsWriter.Save(path, tensors, metadata);
        DisposeAll(tensors);
        return path;
    }

    private static Dictionary<string, SafeTensorDescriptor> Descriptors(params string[] keys)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = [];
        foreach (string key in keys)
        {
            descriptors[key] = new SafeTensorDescriptor
            {
                Name = key,
                DType = DType.F16,
                Shape = new TensorShape(Rank, Hidden),
                DataOffset = 0,
                ByteLength = Rank * Hidden * 2,
            };
        }
        return descriptors;
    }

    private static void DisposeAll(Dictionary<string, Tensor> tensors)
    {
        foreach (Tensor tensor in tensors.Values) tensor.Dispose();
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(_tempDir, recursive: true);
        }
        catch (IOException)
        {
            // best-effort cleanup
        }
    }
}
