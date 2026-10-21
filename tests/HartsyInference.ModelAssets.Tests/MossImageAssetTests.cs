using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Gguf.KeyMappers;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.Registry;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>MOSS-Image's asset plumbing: LoRA detection and name resolution in every spelling (PEFT under
/// <c>transformer.</c>, ComfyUI under <c>diffusion_model.</c>, bare, kohya), a merge landing on the transformer's own keys,
/// and the sharded diffusers folder OpenMOSS ships: detection from the shard index when the signature straddles shards,
/// a picked shard loading its whole set, and the build named by the pipeline folder rather than the shard.</summary>
public sealed unsafe class MossImageAssetTests : IDisposable
{
    private const int Rank = 2;
    private const int Hidden = 4;

    private readonly string _tempDir = Directory.CreateTempSubdirectory("moss_image_assets_").FullName;

    [Theory]
    [InlineData("transformer.layers.3.proj.q.lora_A.weight")]
    [InlineData("transformer.text_embedder.blocks.0.qkv.lora_B.weight")]
    [InlineData("diffusion_model.layers.41.proj.ffn_up_gate.lora_A.weight")]
    [InlineData("diffusion_model.shared_block_modulation.lora_down.weight")]
    [InlineData("layers.0.proj.o.lora_A.weight")]
    [InlineData("lora_unet_layers_12_proj_ffn_down.lora_down.weight")]
    [InlineData("lora_unet_text_embedder_blocks_1_ffn_gate.lora_up.weight")]
    [InlineData("lora_unet_video_head_modulation.lora_down.weight")]
    [InlineData("lora_transformer_layers_0_proj_q.lora_down.weight")]
    [InlineData("lora_diffusion_model_layers_5_proj_ffn_down.lora_up.weight")]
    public void Detect_ClaimsEveryMossImageSpelling(string key)
    {
        Assert.Equal(LoraFormat.MossImage, LoraFormatDetector.Detect(Descriptors(key)));
    }

    [Fact]
    public void Detect_LeavesOtherFamiliesAlone()
    {
        // Wan names its timestep MLP the same way, so it is no evidence of MOSS-Image.
        Assert.Equal(LoraFormat.DiffusersFlux, LoraFormatDetector.Detect(Descriptors("transformer.time_embedding.0.lora_A.weight")));
        // ERNIE-Image / Ideogram 4 also root their blocks at layers.N, under other leaves.
        Assert.Equal(LoraFormat.DiffusersBareDit, LoraFormatDetector.Detect(Descriptors("layers.0.self_attention.to_q.lora_A.weight")));
        Assert.Equal(LoraFormat.Flux3, LoraFormatDetector.Detect(Descriptors("transformer.mode_blocks.image.0.attn.to_q.lora_A.weight")));
        // ai-toolkit's underscored Flux spelling stays Flux: no MOSS-Image module hides behind it.
        Assert.Equal(LoraFormat.AiToolkitFlux,
            LoraFormatDetector.Detect(Descriptors("lora_transformer_single_transformer_blocks_0_attn_to_q.lora_A.weight")));
    }

    [Theory]
    [InlineData("transformer.layers.7.proj.v", "layers.7.proj.v")]
    [InlineData("diffusion_model.text_embedder.proj", "text_embedder.proj")]
    [InlineData("model.diffusion_model.final_linear_video", "final_linear_video")]
    [InlineData("time_embedding.2", "time_embedding.2")]
    [InlineData("lora_unet_layers_0_proj_ffn_up_gate", "layers.0.proj.ffn_up_gate")]
    [InlineData("lora_unet_text_embedder_blocks_1_proj", "text_embedder.blocks.1.proj")]
    [InlineData("lora_unet_text_embedder_proj", "text_embedder.proj")]
    [InlineData("lora_unet_time_embedding_0", "time_embedding.0")]
    [InlineData("lora_unet_shared_block_modulation", "shared_block_modulation")]
    [InlineData("lora_transformer_layers_3_proj_k", "layers.3.proj.k")]
    [InlineData("lora_diffusion_model_layers_0_proj_ffn_up_gate", "layers.0.proj.ffn_up_gate")]
    [InlineData("lora_unet_layers_41_proj_q", "layers.41.proj.q")]
    [InlineData("lora_unet_layers_41_proj_v", "layers.41.proj.v")]
    [InlineData("lora_unet_layers_41_proj_o", "layers.41.proj.o")]
    [InlineData("lora_unet_layers_41_proj_ffn_down", "layers.41.proj.ffn_down")]
    public void TryResolve_MapsEachSpellingToTheTransformerModule(string root, string expected)
    {
        Assert.True(MossImageLoraMapper.TryResolve(root, out string module));
        Assert.Equal(expected, module);
    }

    [Theory]
    [InlineData("transformer.layers.0.q_norm")]
    [InlineData("transformer.layers.0.proj.qkv")]
    [InlineData("lora_unet_layers_x_proj_q")]
    [InlineData("lora_unet_layers_0_proj_q_extra")]
    [InlineData("text_encoder.layers.0.self_attn.q_proj")]
    public void TryResolve_RefusesModulesMossImageDoesNotHave(string root)
    {
        Assert.False(MossImageLoraMapper.TryResolve(root, out _));
    }

    [Fact]
    public void Merge_ComfyAndKohyaAdapters_LandOnTheTransformerKeys()
    {
        // A = 0.5 and alpha = rank, so each delta is 2·0.5·B = B.
        Dictionary<string, Tensor> tensors = new()
        {
            ["diffusion_model.layers.0.proj.q.lora_A.weight"] = Filled([Rank, Hidden], _ => 0.5f),
            ["diffusion_model.layers.0.proj.q.lora_B.weight"] = Filled([Hidden, Rank], _ => 0.25f),
            ["lora_unet_layers_1_proj_ffn_down.lora_down.weight"] = Filled([Rank, Hidden], _ => 0.5f),
            ["lora_unet_layers_1_proj_ffn_down.lora_up.weight"] = Filled([Hidden, Rank], _ => 0.75f),
        };
        string path = Save("comfy_kohya", tensors);
        Dictionary<string, Tensor> weights = new()
        {
            ["layers.0.proj.q.weight"] = Filled([Hidden, Hidden], _ => 1f),
            ["layers.1.proj.ffn_down.weight"] = Filled([Hidden, Hidden], _ => 1f),
        };
        Dictionary<string, Tensor> originals = new(weights);
        using CpuBackend backend = new();
        using LoraStack stack = new();
        stack.AddFromPath(path, strength: 1.0f);
        Assert.Equal(2, stack.ApplyTo(weights, LoraTarget.Transformer, backend));
        foreach (Tensor tensor in originals.Values) tensor.Dispose();

        Assert.All(weights["layers.0.proj.q.weight"].AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(1.25f, v, 5));
        Assert.All(weights["layers.1.proj.ffn_down.weight"].AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(1.75f, v, 5));
    }

    [Fact]
    public void Merge_SplitQkvAdapters_LandOnTheirThirdsOfTheFusedComfyWeight()
    {
        // A = 0.5 and alpha = rank, so each delta is 2·0.5·B = B: q, k and v rows get 0.25, 0.5 and 0.75.
        Dictionary<string, Tensor> tensors = new()
        {
            ["lora_unet_layers_0_proj_q.lora_down.weight"] = Filled([Rank, Hidden], _ => 0.5f),
            ["lora_unet_layers_0_proj_q.lora_up.weight"] = Filled([Hidden, Rank], _ => 0.25f),
            ["lora_unet_layers_0_proj_k.lora_down.weight"] = Filled([Rank, Hidden], _ => 0.5f),
            ["lora_unet_layers_0_proj_k.lora_up.weight"] = Filled([Hidden, Rank], _ => 0.5f),
            ["lora_unet_layers_0_proj_v.lora_down.weight"] = Filled([Rank, Hidden], _ => 0.5f),
            ["lora_unet_layers_0_proj_v.lora_up.weight"] = Filled([Hidden, Rank], _ => 0.75f),
            ["lora_unet_layers_0_proj_ffn_up_gate.lora_down.weight"] = Filled([Rank, Hidden], _ => 0.5f),
            ["lora_unet_layers_0_proj_ffn_up_gate.lora_up.weight"] = Filled([Hidden, Rank], _ => 0f),
        };
        string path = Save("fused_qkv", tensors);
        Dictionary<string, Tensor> weights = new()
        {
            ["layers.0.proj.qkv.weight"] = Filled([3 * Hidden, Hidden], _ => 1f),
            ["layers.0.proj.ffn_up_gate.weight"] = Filled([Hidden, Hidden], _ => 1f),
        };
        Dictionary<string, Tensor> originals = new(weights);
        using CpuBackend backend = new();
        using LoraStack stack = new();
        stack.AddFromPath(path, strength: 1.0f);
        // The three Q/K/V adapters land on one fused weight, so two weights change: proj.qkv and ffn_up_gate.
        Assert.Equal(2, stack.ApplyTo(weights, LoraTarget.Transformer, backend));
        foreach (Tensor tensor in originals.Values) tensor.Dispose();

        float[] fused = weights["layers.0.proj.qkv.weight"].AsReadOnlySpan<float>().ToArray();
        int third = Hidden * Hidden;
        Assert.All(fused[..third], v => Assert.Equal(1.25f, v, 5));
        Assert.All(fused[third..(2 * third)], v => Assert.Equal(1.5f, v, 5));
        Assert.All(fused[(2 * third)..], v => Assert.Equal(1.75f, v, 5));
    }

    [Fact]
    public void ModuleTemplates_ResolveKohyaNamesWithManyIndices()
    {
        LoraModuleTemplates templates = new(["blocks.{i}.attn.{i}.to_q", "blocks.{i}.ff"]);
        Assert.Equal("blocks.12.attn.3.to_q", templates.FromKohya("blocks_12_attn_3_to_q"));
        Assert.Equal("blocks.0.ff", templates.FromKohya("blocks_0_ff"));
        Assert.Null(templates.FromKohya("blocks_0_ff_extra"));
        Assert.True(templates.IsModule("blocks.7.attn.0.to_q"));
        Assert.False(templates.IsModule("blocks.7.attn.to_q"));
        Assert.Equal("blocks.{i}.ff", templates.TemplateOf("blocks.3.ff"));
    }

    [Fact]
    public void Gguf_FamilyRowRecognisesTheTransformerAheadOfErnieImage()
    {
        string[] keys = [.. MossImageCheckpointConverter.SignatureKeys];
        DiffusionGgufKeyMapper? match = DiffusionGgufFamilies.InDetectionOrder.FirstOrDefault(family => family.MatchesByKeys(keys));
        Assert.NotNull(match);
        Assert.Equal("moss_image", match!.Architecture);
    }

    /// <summary>OpenMOSS's diffusers export splits the transformer across shards, and nothing guarantees the shared
    /// modulation, a joint block and the output head share one: the first shard alone must not decide the family.</summary>
    [Fact]
    public void Detection_ReadsTheShardIndexWhenTheSignatureStraddlesShards()
    {
        (string transformer, string[] shards) = WriteShardedTransformer("MOSS-Image-Base");
        Assert.Equal(ModelArchitecture.Unknown, ModelArchitectureDetector.Detect(CheckpointHeader.Read(shards[0]).Descriptors.Keys));
        Assert.Equal(ModelArchitecture.MossImage, ModelArchitectureDetector.DetectFromFile(shards[0]));
        Assert.Equal(ModelArchitecture.MossImage, ModelArchitectureDetector.DetectFromFile(ModelLayoutResolver.Resolve(Path.GetDirectoryName(transformer)!).RepresentativeFile));
        Assert.Equal(shards, HfCheckpointDirectory.CheckpointFiles(shards[1]));
        Assert.Equal(shards, HfCheckpointDirectory.CheckpointFiles(transformer));
        Assert.Equal(Path.GetDirectoryName(transformer), HfCheckpointDirectory.FindPipelineRoot(shards[0], "transformer"));
    }

    [Fact]
    public void Probe_NamesTheBuildByItsPipelineFolder_AndReadsEveryShard()
    {
        (string _, string[] shards) = WriteShardedTransformer("MOSS-Image-SFT");
        CheckpointProbe probe = CheckpointProbe.Read(shards[0]);
        Assert.Contains("MOSS-Image-SFT", probe.FileNames);
        Assert.Contains("transformer", probe.FileNames);
        Assert.Contains("video_head_modulation.weight", probe.Keys);
        Assert.Contains("shared_block_modulation.weight", probe.Keys);
    }

    /// <summary>Diffusers repos may ship a variant index (<c>.fp16.</c>) beside the canonical one: a folder resolves to the
    /// canonical set, and a shard resolves to the set of whichever index names it.</summary>
    [Fact]
    public void CheckpointFiles_PicksTheCanonicalIndex_AndTheIndexThatNamesAShard()
    {
        (string transformer, string[] shards) = WriteShardedTransformer("MOSS-Image-Variants");
        string variant = Path.Combine(transformer, "diffusion_pytorch_model.fp16.safetensors");
        SafeTensorsWriter.Save(variant, new Dictionary<string, Tensor> { ["video_embedder.bias"] = Filled([Hidden], _ => 0f) });
        File.WriteAllText(Path.Combine(transformer, "diffusion_pytorch_model.fp16.safetensors.index.json"),
            "{\"weight_map\": {\"video_embedder.bias\": \"diffusion_pytorch_model.fp16.safetensors\"}}");

        Assert.Equal(shards, HfCheckpointDirectory.WeightFiles(transformer));
        Assert.Equal(shards, HfCheckpointDirectory.CheckpointFiles(shards[1]));
        Assert.Equal([Path.GetFullPath(variant)], HfCheckpointDirectory.CheckpointFiles(variant));
        Assert.Equal(["video_embedder.bias"], HfCheckpointDirectory.IndexedTensorNames(variant));
    }

    [Fact]
    public void CheckpointFiles_LeavesAFileNoIndexNamesAlone()
    {
        string single = Path.Combine(_tempDir, "loose", "moss-image-dit.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(single)!);
        SafeTensorsWriter.Save(single, new Dictionary<string, Tensor> { ["video_embedder.bias"] = Filled([Hidden], _ => 0f) });
        Assert.Equal([Path.GetFullPath(single)], HfCheckpointDirectory.CheckpointFiles(single));
        Assert.Null(HfCheckpointDirectory.IndexedTensorNames(single));
    }

    /// <summary>A two-shard transformer in <c>{root}/transformer/</c> with a diffusers index and a <c>model_index.json</c>:
    /// the first shard holds the text side and the shared modulation, the second the joint blocks and the head.</summary>
    private (string Transformer, string[] Shards) WriteShardedTransformer(string rootName)
    {
        string root = Path.Combine(_tempDir, rootName);
        string transformer = Path.Combine(root, "transformer");
        Directory.CreateDirectory(transformer);
        File.WriteAllText(Path.Combine(root, "model_index.json"), "{\"_class_name\": \"MOSSImagePipeline\"}");
        string[] first = ["video_embedder.bias", "text_embedder.proj.weight", "text_embedder.blocks.0.qkv.weight", "shared_block_modulation.weight", "time_embedding.0.bias"];
        string[] second = ["layers.0.q_norm.weight", "layers.0.proj.ffn_up_gate.weight", "final_linear_video.weight", "video_head_modulation.weight"];
        string[] shards =
        [
            Path.Combine(transformer, "diffusion_pytorch_model-00001-of-00002.safetensors"),
            Path.Combine(transformer, "diffusion_pytorch_model-00002-of-00002.safetensors"),
        ];
        SafeTensorsWriter.Save(shards[0], first.ToDictionary(k => k, _ => Filled([Hidden], _ => 0f)));
        SafeTensorsWriter.Save(shards[1], second.ToDictionary(k => k, _ => Filled([Hidden], _ => 0f)));
        IEnumerable<string> map = first.Select(k => $"\"{k}\": \"{Path.GetFileName(shards[0])}\"")
            .Concat(second.Select(k => $"\"{k}\": \"{Path.GetFileName(shards[1])}\""));
        File.WriteAllText(Path.Combine(transformer, "diffusion_pytorch_model.safetensors.index.json"),
            "{\"metadata\": {}, \"weight_map\": {" + string.Join(", ", map) + "}}");
        return (transformer, [.. shards.Select(Path.GetFullPath)]);
    }

    private static Tensor Filled(long[] shape, Func<int, float> value)
    {
        Tensor tensor = new Tensor(new TensorShape(shape), DType.F32);
        Span<float> span = tensor.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = value(i);
        return tensor;
    }

    private string Save(string name, Dictionary<string, Tensor> tensors)
    {
        string path = Path.Combine(_tempDir, name + ".safetensors");
        SafeTensorsWriter.Save(path, tensors);
        foreach (Tensor tensor in tensors.Values) tensor.Dispose();
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
