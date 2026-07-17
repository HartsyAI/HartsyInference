using System.Text;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class AdapterLoaderTests : IDisposable
{
    private readonly string _tempDir;

    public AdapterLoaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sharpinf-adapter-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void ControlNetLoader_DetectsSd15FromKeySignature()
    {
        string path = CreateSafeTensorsFile("sd15-canny.safetensors", new()
        {
            ["input_blocks.0.0.weight"] = (DType.F32, [320, 4, 3, 3], new float[320 * 4 * 3 * 3]),
            ["input_hint_block.0.weight"] = (DType.F32, [16, 3, 3, 3], new float[16 * 3 * 3 * 3]),
            ["controlnet_down_blocks.0.weight"] = (DType.F32, [320, 320, 1, 1], new float[320 * 320]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sd15, file.BaseModel);
        Assert.Equal(ControlNetMode.Canny, file.Mode);
        Assert.Equal(768, file.Config.CrossAttentionDim);
    }

    [Fact]
    public void ControlNetLoader_DetectsSdxlFromCrossAttnDim()
    {
        string path = CreateSafeTensorsFile("controlnet-sdxl-depth.safetensors", new()
        {
            ["down_blocks.0.attentions.0.proj_in.weight"] = (DType.F32, [640, 2048], new float[640 * 2048]),
            ["controlnet_down_blocks.0.weight"] = (DType.F32, [320, 320, 1, 1], new float[320 * 320]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sdxl, file.BaseModel);
        Assert.Equal(ControlNetMode.Depth, file.Mode);
        Assert.Equal(2048, file.Config.CrossAttentionDim);
    }

    [Fact]
    public void ControlNetLoader_DetectsFluxFromTransformerBlocks()
    {
        // Architecture-defining subset of a diffusers FluxControlNetModel header (FromDescriptors requires the
        // embedders to derive dims). Full-depth detection + union handling live in FluxControlNetTests.
        string path = CreateSafeTensorsFile("flux-controlnet-canny.safetensors", new()
        {
            ["x_embedder.weight"] = (DType.F32, [3072, 64], new float[3072 * 64]),
            ["controlnet_x_embedder.weight"] = (DType.F32, [3072, 64], new float[3072 * 64]),
            ["context_embedder.weight"] = (DType.F32, [3072, 4096], new float[3072 * 4096]),
            ["transformer_blocks.0.attn.to_q.weight"] = (DType.F32, [3072, 3072], new float[3072 * 3072]),
            ["controlnet_blocks.0.weight"] = (DType.F32, [3072, 3072], new float[3072 * 3072]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Flux, file.BaseModel);
        Assert.Equal(ControlNetMode.Canny, file.Mode);
        Assert.Equal(4096, file.Config.CrossAttentionDim);
        Assert.NotNull(file.FluxConfig);
        Assert.Equal(1, file.FluxConfig!.Depth);
        Assert.Equal(24, file.FluxConfig.NumHeads);
    }

    [Fact]
    public void ControlNetLoader_ModeOverrideTakesPrecedence()
    {
        string path = CreateSafeTensorsFile("ambiguous-name.safetensors", new()
        {
            ["input_blocks.0.0.weight"] = (DType.F32, [320, 4, 3, 3], new float[320 * 4 * 3 * 3]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path, modeOverride: ControlNetMode.OpenPose);
        Assert.Equal(ControlNetMode.OpenPose, file.Mode);
    }

    [Fact]
    public void ControlNetLoader_LdmLayout_Sd15_ConvertsToDiffusersKeys()
    {
        // Miniature of a control_v11p_sd15_* checkpoint: control_model.-prefixed LDM keys covering
        // every ControlNet-specific tower plus one representative of each encoder key family.
        string path = CreateSafeTensorsFile("control_v11p_sd15_canny_fp16.safetensors", new()
        {
            ["control_model.input_blocks.0.0.weight"] = (DType.F32, [8, 4, 3, 3], new float[8 * 4 * 3 * 3]),
            ["control_model.input_blocks.0.0.bias"] = (DType.F32, [8], new float[8]),
            ["control_model.input_blocks.1.0.in_layers.0.weight"] = (DType.F32, [8], new float[8]),
            ["control_model.input_blocks.1.0.emb_layers.1.weight"] = (DType.F32, [8, 16], new float[8 * 16]),
            ["control_model.input_blocks.3.0.op.weight"] = (DType.F32, [8, 8, 3, 3], new float[8 * 8 * 3 * 3]),
            ["control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"] = (DType.F32, [8, 768], new float[8 * 768]),
            ["control_model.time_embed.0.weight"] = (DType.F32, [16, 8], new float[16 * 8]),
            ["control_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight"] = (DType.F32, [8, 8], new float[8 * 8]),
            ["control_model.middle_block.2.out_layers.3.weight"] = (DType.F32, [8, 8, 3, 3], new float[8 * 8 * 3 * 3]),
            ["control_model.input_hint_block.0.weight"] = (DType.F32, [4, 3, 3, 3], new float[4 * 3 * 3 * 3]),
            ["control_model.input_hint_block.6.weight"] = (DType.F32, [4, 4, 3, 3], new float[4 * 4 * 3 * 3]),
            ["control_model.input_hint_block.14.weight"] = (DType.F32, [8, 4, 3, 3], new float[8 * 4 * 3 * 3]),
            ["control_model.zero_convs.0.0.weight"] = (DType.F32, [8, 8, 1, 1], new float[8 * 8]),
            ["control_model.zero_convs.0.0.bias"] = (DType.F32, [8], new float[8]),
            ["control_model.middle_block_out.0.weight"] = (DType.F32, [8, 8, 1, 1], new float[8 * 8]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sd15, file.BaseModel);
        Assert.Equal(ControlNetMode.Canny, file.Mode);
        Assert.Equal(768, file.Config.CrossAttentionDim);

        Assert.Contains("conv_in.weight", file.Weights);
        Assert.Contains("conv_in.bias", file.Weights);
        Assert.Contains("down_blocks.0.resnets.0.norm1.weight", file.Weights);
        Assert.Contains("down_blocks.0.resnets.0.time_emb_proj.weight", file.Weights);
        Assert.Contains("down_blocks.0.downsamplers.0.conv.weight", file.Weights);
        Assert.Contains("down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight", file.Weights);
        Assert.Contains("time_embedding.linear_1.weight", file.Weights);
        Assert.Contains("mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight", file.Weights);
        Assert.Contains("mid_block.resnets.1.conv2.weight", file.Weights);
        Assert.Contains("controlnet_cond_embedding.conv_in.weight", file.Weights);
        Assert.Contains("controlnet_cond_embedding.blocks.2.weight", file.Weights);
        Assert.Contains("controlnet_cond_embedding.conv_out.weight", file.Weights);
        Assert.Contains("controlnet_down_blocks.0.weight", file.Weights);
        Assert.Contains("controlnet_down_blocks.0.bias", file.Weights);
        Assert.Contains("controlnet_mid_block.weight", file.Weights);

        // Every LDM key must have been consumed — no control_model.* survivors.
        Assert.Equal(15, file.Weights.Count);
        Assert.DoesNotContain(file.Weights.Keys, k => k.StartsWith("control_model.", StringComparison.Ordinal));
    }

    [Fact]
    public void ControlNetLoader_LdmLayout_Sdxl_DetectsFromLabelEmbAndContextDim()
    {
        string path = CreateSafeTensorsFile("controlnet-xl-ldm-depth.safetensors", new()
        {
            ["control_model.input_blocks.0.0.weight"] = (DType.F32, [8, 4, 3, 3], new float[8 * 4 * 3 * 3]),
            ["control_model.label_emb.0.0.weight"] = (DType.F32, [16, 32], new float[16 * 32]),
            ["control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"] = (DType.F32, [8, 2048], new float[8 * 2048]),
            ["control_model.zero_convs.0.0.weight"] = (DType.F32, [8, 8, 1, 1], new float[8 * 8]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sdxl, file.BaseModel);
        Assert.Equal(2048, file.Config.CrossAttentionDim);
        Assert.Contains("add_embedding.linear_1.weight", file.Weights);
        Assert.Contains("down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight", file.Weights);
        Assert.Contains("controlnet_down_blocks.0.weight", file.Weights);
    }

    [Fact]
    public void ControlNetLoader_LdmLayout_Sd21ContextDim_Throws()
    {
        string path = CreateSafeTensorsFile("controlnet-sd21-canny.safetensors", new()
        {
            ["control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"] = (DType.F32, [8, 1024], new float[8 * 1024]),
            ["control_model.zero_convs.0.0.weight"] = (DType.F32, [8, 8, 1, 1], new float[8 * 8]),
        });

        Assert.Throws<HartsyInferenceException>(() => ControlNetLoader.Load(path));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ControlNetLoader_RealLdmCheckpoint_LoadsAndConverts()
    {
        // Real lllyasviel/comfyanonymous fp16 repack in LDM layout. Header parse + mmap only — no GPU.
        string dir = Environment.GetEnvironmentVariable("HARTSY_CONTROLNET_DIR")
            ?? "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Models/controlnet";
        string path = Path.Combine(dir, "control_v11p_sd15_canny_fp16.safetensors");
        if (!File.Exists(path)) return;

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sd15, file.BaseModel);
        Assert.Equal(ControlNetMode.Canny, file.Mode);
        Assert.Equal(340, file.Weights.Count);
        Assert.Contains("conv_in.weight", file.Weights);
        Assert.Contains("controlnet_cond_embedding.conv_in.weight", file.Weights);
        Assert.Contains("controlnet_cond_embedding.conv_out.weight", file.Weights);
        Assert.Contains("controlnet_mid_block.weight", file.Weights);
        for (int i = 0; i < 12; i++)
        {
            Assert.Contains($"controlnet_down_blocks.{i}.weight", file.Weights);
            Assert.Contains($"controlnet_down_blocks.{i}.bias", file.Weights);
        }

        // The converted dictionary must satisfy the full diffusers ControlNet layout end-to-end.
        using ControlNet adapter = new ControlNet(file.Config, UNetConfig.Sd15);
        adapter.LoadWeights(file.Weights);
        Assert.Equal(12, adapter.DownResidualCount);

        foreach ((string name, ControlNetMode expectedMode) in new (string, ControlNetMode)[]
        {
            ("control_v11p_sd15_openpose_fp16.safetensors", ControlNetMode.OpenPose),
            ("control_v11f1p_sd15_depth_fp16.safetensors", ControlNetMode.Depth),
        })
        {
            string siblingPath = Path.Combine(dir, name);
            if (!File.Exists(siblingPath)) continue;
            using ControlNetFile sibling = ControlNetLoader.Load(siblingPath);
            Assert.Equal(ControlNetBaseModel.Sd15, sibling.BaseModel);
            Assert.Equal(expectedMode, sibling.Mode);
            Assert.Contains("controlnet_mid_block.weight", sibling.Weights);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ControlNetLoader_RealSegCheckpoint_LoadsAndConverts()
    {
        // control_v11p_sd15_seg conditioned on ADE20K palette maps (UperNetSegPreprocessor output).
        string dir = Environment.GetEnvironmentVariable("HARTSY_CONTROLNET_DIR")
            ?? "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Models/controlnet";
        string path = Path.Combine(dir, "control_v11p_sd15_seg_fp16.safetensors");
        if (!File.Exists(path)) return;

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sd15, file.BaseModel);
        Assert.Equal(ControlNetMode.Segmentation, file.Mode);
        Assert.Equal(340, file.Weights.Count);
        Assert.Contains("controlnet_cond_embedding.conv_in.weight", file.Weights);
        Assert.Contains("controlnet_mid_block.weight", file.Weights);

        using ControlNet adapter = new ControlNet(file.Config, UNetConfig.Sd15);
        adapter.LoadWeights(file.Weights);
        Assert.Equal(12, adapter.DownResidualCount);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ControlNetLoader_RealDiffusersXlCheckpoint_StillLoads()
    {
        // diffusers_xl_canny_full ships in diffusers layout already — must keep taking the
        // non-converting path untouched by the LDM support.
        string dir = Environment.GetEnvironmentVariable("HARTSY_CONTROLNET_DIR")
            ?? "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Models/controlnet";
        string path = Path.Combine(dir, "diffusers_xl_canny_full.safetensors");
        if (!File.Exists(path)) return;

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sdxl, file.BaseModel);
        Assert.Equal(ControlNetMode.Canny, file.Mode);
        Assert.Contains("add_embedding.linear_1.weight", file.Weights);
        Assert.Contains("controlnet_mid_block.weight", file.Weights);
    }

    [Fact]
    public void ControlNetLoader_UnrecognizedFile_Throws()
    {
        string path = CreateSafeTensorsFile("garbage.safetensors", new()
        {
            ["some.random.key"] = (DType.F32, [4, 4], new float[16]),
        });

        Assert.Throws<HartsyInferenceException>(() => ControlNetLoader.Load(path));
    }

    [Fact]
    public void IpAdapterLoader_DetectsSd15Standard()
    {
        string path = CreateSafeTensorsFile("ip-adapter_sd15.safetensors", new()
        {
            ["image_proj.weight"] = (DType.F32, [768 * 4, 1024], new float[768 * 4 * 1024]),
            ["ip_adapter.0.to_k_ip.weight"] = (DType.F32, [768, 768], new float[768 * 768]),
            ["ip_adapter.0.to_v_ip.weight"] = (DType.F32, [768, 768], new float[768 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.Equal(IpAdapterBaseModel.Sd15, file.BaseModel);
        Assert.False(file.Config.IsPlus);
        Assert.False(file.Config.IsFaceId);
        Assert.Equal(4, file.Config.NumImageTokens);
        Assert.Equal(768, file.Config.CrossAttentionDim);
    }

    [Fact]
    public void IpAdapterLoader_DetectsPlus_From_NameAnd_NormKey()
    {
        string path = CreateSafeTensorsFile("ip-adapter-plus_sdxl.safetensors", new()
        {
            ["image_proj.norm.weight"] = (DType.F32, [768], new float[768]),
            ["image_proj.proj_in.weight"] = (DType.F32, [768, 1024], new float[768 * 1024]),
            ["ip_adapter.0.to_k_ip.weight"] = (DType.F32, [2048, 2048], new float[2048 * 2048]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.Equal(IpAdapterBaseModel.Sdxl, file.BaseModel);
        Assert.True(file.Config.IsPlus);
        Assert.Equal(16, file.Config.NumImageTokens);
        Assert.Equal(2048, file.Config.CrossAttentionDim);
    }

    [Fact]
    public void IpAdapterLoader_DetectsFaceId_From_FilenameKeyword()
    {
        string path = CreateSafeTensorsFile("ip-adapter-faceid_sd15.safetensors", new()
        {
            ["image_proj.weight"] = (DType.F32, [768 * 4, 512], new float[768 * 4 * 512]),
            ["ip_adapter.0.to_k_ip.weight"] = (DType.F32, [768, 768], new float[768 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.True(file.Config.IsFaceId);
        Assert.Contains("ArcFace", file.Config.ClipImageModel);
    }

    [Fact]
    public void IpAdapterLoader_StandardWithNormKeys_NotDetectedAsPlus()
    {
        // Real ip-adapter_sd15.safetensors carries image_proj.norm.{weight,bias} (the standard
        // projection's post-Linear LayerNorm) — the norm key must NOT trip Plus detection.
        string path = CreateSafeTensorsFile("ip-adapter_sd15_std.safetensors", new()
        {
            ["image_proj.proj.weight"] = (DType.F32, [768 * 4, 1024], new float[768 * 4 * 1024]),
            ["image_proj.proj.bias"] = (DType.F32, [768 * 4], new float[768 * 4]),
            ["image_proj.norm.weight"] = (DType.F32, [768], new float[768]),
            ["image_proj.norm.bias"] = (DType.F32, [768], new float[768]),
            ["ip_adapter.1.to_k_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.1.to_v_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.Equal(IpAdapterBaseModel.Sd15, file.BaseModel);
        Assert.False(file.Config.IsPlus);
        Assert.Equal(4, file.Config.NumImageTokens);
        Assert.Equal(768, file.Config.CrossAttentionDim);
        Assert.Equal(1024, file.Config.ImageEmbeddingDim);
    }

    [Fact]
    public void IpAdapter_Sd15Standard_LoadsAndProjectsFromSyntheticCheckpoint()
    {
        // Tiny SD1.5 standard checkpoint in the real key layout (2 cross-attn layers instead of 16).
        string path = CreateSafeTensorsFile("ip-adapter_sd15_tiny.safetensors", new()
        {
            ["image_proj.proj.weight"] = (DType.F32, [768 * 4, 1024], new float[768 * 4 * 1024]),
            ["image_proj.proj.bias"] = (DType.F32, [768 * 4], new float[768 * 4]),
            ["image_proj.norm.weight"] = (DType.F32, [768], OnesArray(768)),
            ["image_proj.norm.bias"] = (DType.F32, [768], new float[768]),
            ["ip_adapter.1.to_k_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.1.to_v_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.3.to_k_ip.weight"] = (DType.F32, [640, 768], new float[640 * 768]),
            ["ip_adapter.3.to_v_ip.weight"] = (DType.F32, [640, 768], new float[640 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        using IpAdapter adapter = new IpAdapter(file.Config);
        adapter.LoadWeights(file.Weights);

        Assert.Equal(2, adapter.CrossAttentionLayerCount);
        Assert.Equal(4, adapter.NumImageTokens);
        Assert.Equal(new TensorShape(320, 768), adapter.GetToKIpWeight(0).Shape);
        Assert.Equal(new TensorShape(640, 768), adapter.GetToVIpWeight(1).Shape);

        using CpuBackend backend = new CpuBackend();
        using Tensor clipEmbed = new Tensor(new TensorShape(1, 1024), DType.F32);
        clipEmbed.AsSpan<float>().Clear();
        using Tensor tokens = adapter.ProjectImage(backend, clipEmbed);
        Assert.Equal(new TensorShape(1, 4, 768), tokens.Shape);
        foreach (float v in tokens.AsSpan<float>())
        {
            Assert.Equal(0f, v);
        }
    }

    [Fact]
    public void IpAdapter_Sd15Plus_ConstructsResampler()
    {
        using IpAdapter adapter = new IpAdapter(IpAdapterConfig.Sd15Plus);
        Assert.Equal(16, adapter.NumImageTokens);
        Assert.IsType<IpAdapterPlusResampler>(adapter.ImageProjection);
    }

    [Fact]
    public void IpAdapterLoader_DetectsFaceId_FromLoraKeySignature()
    {
        // Neutral filename — detection must come from the embedded LoRA half's key signature
        // (ip_adapter.{i}.*_lora.down.weight) and derive dims from the FaceID MLP proj shapes.
        string path = CreateSafeTensorsFile("mystery-adapter.safetensors", new()
        {
            ["image_proj.proj.0.weight"] = (DType.F32, [1024, 512], new float[1024 * 512]),
            ["image_proj.proj.0.bias"] = (DType.F32, [1024], new float[1024]),
            ["image_proj.proj.2.weight"] = (DType.F32, [3072, 1024], new float[3072 * 1024]),
            ["image_proj.proj.2.bias"] = (DType.F32, [3072], new float[3072]),
            ["image_proj.norm.weight"] = (DType.F32, [768], OnesArray(768)),
            ["image_proj.norm.bias"] = (DType.F32, [768], new float[768]),
            ["ip_adapter.0.to_q_lora.down.weight"] = (DType.F32, [4, 320], new float[4 * 320]),
            ["ip_adapter.0.to_q_lora.up.weight"] = (DType.F32, [320, 4], new float[320 * 4]),
            ["ip_adapter.1.to_k_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.1.to_v_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.True(file.Config.IsFaceId);
        Assert.False(file.Config.IsPlus);
        Assert.Equal(IpAdapterBaseModel.Sd15, file.BaseModel);
        Assert.Equal(512, file.Config.ImageEmbeddingDim);
        Assert.Equal(4, file.Config.NumImageTokens);
        Assert.Contains("ArcFace", file.Config.ClipImageModel);
    }

    [Fact]
    public void IpAdapter_FaceId_LoadsAndProjects_SyntheticCheckpoint()
    {
        // Tiny FaceID checkpoint in the real (flattened .bin) key layout: MLP proj + embedded LoRA keys
        // (which LoadWeights must SKIP) + 2 cross-attn K/V pairs at the odd combined-attn indices.
        string path = CreateSafeTensorsFile("ip-adapter-faceid_sd15_tiny.safetensors", new()
        {
            ["image_proj.proj.0.weight"] = (DType.F32, [1024, 512], new float[1024 * 512]),
            ["image_proj.proj.0.bias"] = (DType.F32, [1024], OnesArray(1024)),
            ["image_proj.proj.2.weight"] = (DType.F32, [3072, 1024], new float[3072 * 1024]),
            ["image_proj.proj.2.bias"] = (DType.F32, [3072], RampArray(3072)),
            ["image_proj.norm.weight"] = (DType.F32, [768], OnesArray(768)),
            ["image_proj.norm.bias"] = (DType.F32, [768], new float[768]),
            ["ip_adapter.0.to_q_lora.down.weight"] = (DType.F32, [4, 320], new float[4 * 320]),
            ["ip_adapter.0.to_q_lora.up.weight"] = (DType.F32, [320, 4], new float[320 * 4]),
            ["ip_adapter.1.to_k_lora.down.weight"] = (DType.F32, [4, 768], new float[4 * 768]),
            ["ip_adapter.1.to_k_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.1.to_v_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.3.to_k_ip.weight"] = (DType.F32, [640, 768], new float[640 * 768]),
            ["ip_adapter.3.to_v_ip.weight"] = (DType.F32, [640, 768], new float[640 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.True(file.Config.IsFaceId);
        using IpAdapter adapter = new IpAdapter(file.Config);
        adapter.LoadWeights(file.Weights);

        Assert.Equal(2, adapter.CrossAttentionLayerCount);
        Assert.Equal(4, adapter.NumImageTokens);
        Assert.IsType<IpAdapterFaceIdProjection>(adapter.ImageProjection);

        using CpuBackend backend = new CpuBackend();
        using Tensor faceEmbed = new Tensor(new TensorShape(1, 512), DType.F32);
        faceEmbed.AsSpan<float>().Clear();
        using Tensor tokens = adapter.ProjectImage(backend, faceEmbed);
        Assert.Equal(new TensorShape(1, 4, 768), tokens.Shape);
        foreach (float v in tokens.AsSpan<float>())
        {
            Assert.True(float.IsFinite(v));
        }
    }

    [Fact]
    public void IpAdapterLoader_DetectsFaceIdPlusV2_FromFilenameAndPerceiverKeys()
    {
        // Shape-probed subset of ip-adapter-faceid-plusv2_sd15.bin (detection + BuildConfig don't
        // touch the resampler layer weights, so those are omitted to keep the file small).
        string path = CreateSafeTensorsFile("ip-adapter-faceid-plusv2_sd15.safetensors", new()
        {
            ["image_proj.proj.0.weight"] = (DType.F32, [1024, 512], new float[1024 * 512]),
            ["image_proj.proj.2.weight"] = (DType.F32, [3072, 1024], new float[3072 * 1024]),
            ["image_proj.norm.weight"] = (DType.F32, [768], OnesArray(768)),
            ["image_proj.perceiver_resampler.proj_in.weight"] = (DType.F32, [768, 1280], new float[768 * 1280]),
            ["ip_adapter.1.to_k_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.1.to_v_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.Equal(IpAdapterBaseModel.Sd15, file.BaseModel);
        Assert.True(file.Config.IsFaceId);
        Assert.True(file.Config.IsPlus);
        Assert.True(file.Config.IsFaceIdV2);
        Assert.Equal(512, file.Config.ImageEmbeddingDim);
        Assert.Equal(1280, file.Config.ClipEmbeddingDim);
        Assert.Equal(4, file.Config.NumImageTokens);
        Assert.Equal(768, file.Config.CrossAttentionDim);
    }

    [Fact]
    public void IpAdapterLoader_FaceIdPlus_WithoutV2Name_IsNotV2()
    {
        string path = CreateSafeTensorsFile("ip-adapter-faceid-plus_sd15.safetensors", new()
        {
            ["image_proj.proj.0.weight"] = (DType.F32, [1024, 512], new float[1024 * 512]),
            ["image_proj.proj.2.weight"] = (DType.F32, [3072, 1024], new float[3072 * 1024]),
            ["image_proj.norm.weight"] = (DType.F32, [768], OnesArray(768)),
            ["image_proj.perceiver_resampler.proj_in.weight"] = (DType.F32, [768, 1280], new float[768 * 1280]),
            ["ip_adapter.1.to_k_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.1.to_v_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.True(file.Config.IsFaceId);
        Assert.True(file.Config.IsPlus);
        Assert.False(file.Config.IsFaceIdV2);
    }

    [Fact]
    public void IpAdapter_FaceIdPlus_LoadsAndProjects_SyntheticWeights()
    {
        // Tiny ProjPlusModel: crossDim=128 (2 heads × 64), clipDim=96, idDim=64, 4 tokens, depth 4.
        const int cross = 128, clip = 96, id = 64, tokens = 4, seq = 5;
        Dictionary<string, Tensor> weights = new();
        AddTensor(weights, "image_proj.proj.0.weight", [id * 2, id]);
        AddTensor(weights, "image_proj.proj.0.bias", [id * 2]);
        AddTensor(weights, "image_proj.proj.2.weight", [tokens * cross, id * 2]);
        AddTensor(weights, "image_proj.proj.2.bias", [tokens * cross]);
        AddTensor(weights, "image_proj.norm.weight", [cross], fill: 1f);
        AddTensor(weights, "image_proj.norm.bias", [cross]);
        AddTensor(weights, "image_proj.perceiver_resampler.proj_in.weight", [cross, clip]);
        AddTensor(weights, "image_proj.perceiver_resampler.proj_in.bias", [cross]);
        AddTensor(weights, "image_proj.perceiver_resampler.proj_out.weight", [cross, cross]);
        AddTensor(weights, "image_proj.perceiver_resampler.proj_out.bias", [cross]);
        AddTensor(weights, "image_proj.perceiver_resampler.norm_out.weight", [cross], fill: 1f);
        AddTensor(weights, "image_proj.perceiver_resampler.norm_out.bias", [cross]);
        for (int i = 0; i < 4; i++)
        {
            string layer = $"image_proj.perceiver_resampler.layers.{i}";
            AddTensor(weights, $"{layer}.0.norm1.weight", [cross], fill: 1f);
            AddTensor(weights, $"{layer}.0.norm1.bias", [cross]);
            AddTensor(weights, $"{layer}.0.norm2.weight", [cross], fill: 1f);
            AddTensor(weights, $"{layer}.0.norm2.bias", [cross]);
            AddTensor(weights, $"{layer}.0.to_q.weight", [cross, cross]);
            AddTensor(weights, $"{layer}.0.to_kv.weight", [2 * cross, cross]);
            AddTensor(weights, $"{layer}.0.to_out.weight", [cross, cross]);
            AddTensor(weights, $"{layer}.1.0.weight", [cross], fill: 1f);
            AddTensor(weights, $"{layer}.1.0.bias", [cross]);
            AddTensor(weights, $"{layer}.1.1.weight", [4 * cross, cross]);
            AddTensor(weights, $"{layer}.1.3.weight", [cross, 4 * cross]);
        }
        AddTensor(weights, "ip_adapter.1.to_k_ip.weight", [64, cross]);
        AddTensor(weights, "ip_adapter.1.to_v_ip.weight", [64, cross]);

        IpAdapterConfig config = new()
        {
            BaseModel = IpAdapterBaseModel.Sd15,
            ClipImageModel = "InsightFace ArcFace + ViT-H/14",
            ImageEmbeddingDim = id,
            NumImageTokens = tokens,
            CrossAttentionDim = cross,
            ClipEmbeddingDim = clip,
            IsPlus = true,
            IsFaceId = true,
            IsFaceIdV2 = true,
        };
        using IpAdapter adapter = new(config);
        adapter.LoadWeights(weights);
        Assert.IsType<IpAdapterFaceIdPlusProjection>(adapter.ImageProjection);
        Assert.Equal(tokens, adapter.NumImageTokens);

        using CpuBackend backend = new CpuBackend();
        using Tensor faceEmbed = MakeRamp([1, id], scale: 0.01f);
        using Tensor clipEmbeds = MakeRamp([1, seq, clip], scale: 0.02f);

        // Single-input path must refuse — FaceID-Plus needs both inputs.
        Assert.Throws<InvalidOperationException>(() => adapter.ProjectImage(backend, faceEmbed));

        using Tensor tokensOut = adapter.ProjectImage(backend, faceEmbed, clipEmbeds, shortcutScale: 1.0f);
        Assert.Equal(new TensorShape(1, tokens, cross), tokensOut.Shape);
        foreach (float v in tokensOut.AsSpan<float>())
        {
            Assert.True(float.IsFinite(v));
        }

        // v2 shortcut algebra on the same weights: out(s) = mlp + s·r  ⇒  out(1) − out(0.5) = 0.5·r ≠ 0.
        using Tensor half = adapter.ProjectImage(backend, faceEmbed, clipEmbeds, shortcutScale: 0.5f);
        float diff = 0f;
        ReadOnlySpan<float> a = tokensOut.AsSpan<float>();
        ReadOnlySpan<float> b = half.AsSpan<float>();
        for (int i = 0; i < a.Length; i++) diff += MathF.Abs(a[i] - b[i]);
        Assert.True(diff > 0f, "shortcut scale had no effect on a v2 FaceID-Plus projection.");

        foreach (Tensor t in weights.Values) t.Dispose();
    }

    private static void AddTensor(Dictionary<string, Tensor> weights, string key, long[] shape, float fill = float.NaN)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        Span<float> span = t.AsSpan<float>();
        if (float.IsNaN(fill))
        {
            // Small deterministic pseudo-random values keep the forward numerically tame.
            for (int i = 0; i < span.Length; i++) span[i] = ((i * 37 + key.Length * 11) % 19 - 9) * 0.01f;
        }
        else
        {
            span.Fill(fill);
        }
        weights[key] = t;
    }

    private static Tensor MakeRamp(long[] shape, float scale)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        Span<float> span = t.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = ((i % 23) - 11) * scale;
        return t;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void IpAdapter_FaceId_RealBinCheckpoint_LoadsAndProjects()
    {
        // Real h94/IP-Adapter-FaceID torch-pickle checkpoint (ip-adapter-faceid_sdxl.bin or _sd15.bin).
        string? binPath = Environment.GetEnvironmentVariable("FACEID_BIN");
        if (string.IsNullOrEmpty(binPath) || !File.Exists(binPath))
        {
            return; // SKIPPED: set FACEID_BIN to a real ip-adapter-faceid_*.bin
        }

        using IpAdapterFile file = IpAdapterLoader.Load(binPath);
        Assert.True(file.Config.IsFaceId);
        Assert.Equal(512, file.Config.ImageEmbeddingDim);
        Assert.Equal(4, file.Config.NumImageTokens);

        using IpAdapter adapter = new IpAdapter(file.Config);
        adapter.LoadWeights(file.Weights);
        int expectedLayers = file.BaseModel == IpAdapterBaseModel.Sdxl ? 70 : 16;
        Assert.Equal(expectedLayers, adapter.CrossAttentionLayerCount);

        using CpuBackend backend = new CpuBackend();
        using Tensor faceEmbed = new Tensor(new TensorShape(1, 512), DType.F32);
        Span<float> span = faceEmbed.AsSpan<float>();
        Random rng = new(42);
        float sumSq = 0f;
        for (int i = 0; i < span.Length; i++) { span[i] = (float)(rng.NextDouble() * 2 - 1); sumSq += span[i] * span[i]; }
        float inv = 1f / MathF.Sqrt(sumSq);
        for (int i = 0; i < span.Length; i++) span[i] *= inv;

        using Tensor tokens = adapter.ProjectImage(backend, faceEmbed);
        Assert.Equal(new TensorShape(1, file.Config.NumImageTokens, file.Config.CrossAttentionDim), tokens.Shape);
        float sumAbs = 0f;
        foreach (float v in tokens.AsSpan<float>())
        {
            Assert.True(float.IsFinite(v));
            sumAbs += MathF.Abs(v);
        }
        Assert.True(sumAbs > 0f, "FaceID projection produced all-zero tokens on a real checkpoint.");
    }

    [Fact]
    public void ControlNetConditioning_DefaultWindow_IsAlwaysActive()
    {
        using ControlNet controlNet = new ControlNet(ControlNetConfig.Sd15(ControlNetMode.Canny), UNetConfig.Sd15);
        using Tensor condImage = new Tensor(new TensorShape(1, 3, 8, 8), DType.F32);
        ControlNetConditioning conditioning = new() { Adapter = controlNet, ConditionImage = condImage };
        for (int i = 0; i < 20; i++)
        {
            Assert.True(conditioning.IsActiveAtStep(i, 20));
        }
    }

    [Fact]
    public void ControlNetConditioning_StartEndWindow_GatesSteps()
    {
        using ControlNet controlNet = new ControlNet(ControlNetConfig.Sd15(ControlNetMode.Canny), UNetConfig.Sd15);
        using Tensor condImage = new Tensor(new TensorShape(1, 3, 8, 8), DType.F32);
        ControlNetConditioning conditioning = new()
        {
            Adapter = controlNet,
            ConditionImage = condImage,
            StartFraction = 0.25f,
            EndFraction = 0.75f,
        };
        // fraction = step / totalSteps with 20 steps → active iff 0.25 <= i/20 <= 0.75 → i in [5, 15].
        Assert.False(conditioning.IsActiveAtStep(4, 20));
        Assert.True(conditioning.IsActiveAtStep(5, 20));
        Assert.True(conditioning.IsActiveAtStep(15, 20));
        Assert.False(conditioning.IsActiveAtStep(16, 20));
    }

    [Fact]
    public void ControlNetConditioning_FilterActive_HandlesAllNoneAndPartial()
    {
        using ControlNet controlNet = new ControlNet(ControlNetConfig.Sd15(ControlNetMode.Canny), UNetConfig.Sd15);
        using Tensor condImage = new Tensor(new TensorShape(1, 3, 8, 8), DType.F32);
        ControlNetConditioning early = new()
        {
            Adapter = controlNet,
            ConditionImage = condImage,
            EndFraction = 0.4f,
        };
        ControlNetConditioning late = new()
        {
            Adapter = controlNet,
            ConditionImage = condImage,
            StartFraction = 0.6f,
        };
        List<ControlNetConditioning> stack = [early, late];

        Assert.Null(ControlNetConditioning.FilterActive(null, 0, 20));
        Assert.Null(ControlNetConditioning.FilterActive([], 0, 20));

        // Step 0 (fraction 0): only the early adapter.
        IReadOnlyList<ControlNetConditioning>? atStart = ControlNetConditioning.FilterActive(stack, 0, 20);
        Assert.NotNull(atStart);
        Assert.Same(early, Assert.Single(atStart));

        // Step 10 (fraction 0.5): neither window covers it.
        Assert.Null(ControlNetConditioning.FilterActive(stack, 10, 20));

        // Step 19 (fraction 0.95): only the late adapter.
        IReadOnlyList<ControlNetConditioning>? atEnd = ControlNetConditioning.FilterActive(stack, 19, 20);
        Assert.NotNull(atEnd);
        Assert.Same(late, Assert.Single(atEnd));

        // All active → the original list instance comes back (no per-step allocation).
        List<ControlNetConditioning> alwaysOn = [new() { Adapter = controlNet, ConditionImage = condImage }];
        Assert.Same(alwaysOn, ControlNetConditioning.FilterActive(alwaysOn, 10, 20));
    }

    [Fact]
    public void ControlNet_Sd15AndSdxl_ResidualCounts_MatchBaseUNetSkipLayout()
    {
        // SD1.5: 1 conv_in + 4 blocks × 2 resnet skips + 3 downsample skips = 12.
        using ControlNet sd15 = new ControlNet(ControlNetConfig.Sd15(ControlNetMode.Canny), UNetConfig.Sd15);
        Assert.Equal(12, sd15.DownResidualCount);

        // SDXL: 1 conv_in + 3 blocks × 2 resnet skips + 2 downsample skips = 9.
        using ControlNet sdxl = new ControlNet(ControlNetConfig.Sdxl(ControlNetMode.Depth), UNetConfig.SdxlBase);
        Assert.Equal(9, sdxl.DownResidualCount);
    }

    [Fact]
    public void UNet_CrossAttentionLayerCounts_MatchIpAdapterCheckpointLayout()
    {
        // ip-adapter_sd15 ships 16 K/V pairs; SDXL adapters ship 70. The UNet's enumeration
        // must land on the same counts or UNet.Forward rejects the flat K/V lists.
        Assert.Equal(16, new UNet(UNetConfig.Sd15).CrossAttentionLayerCount);
        Assert.Equal(70, new UNet(UNetConfig.SdxlBase).CrossAttentionLayerCount);
    }

    private static float[] OnesArray(int count)
    {
        float[] result = new float[count];
        Array.Fill(result, 1f);
        return result;
    }

    private static float[] RampArray(int count)
    {
        float[] result = new float[count];
        for (int i = 0; i < count; i++) result[i] = (i % 17) * 0.05f;
        return result;
    }

    private string CreateSafeTensorsFile(string name, Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors)
    {
        using MemoryStream dataStream = new();
        Dictionary<string, (long start, long end)> offsets = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            long start = dataStream.Position;
            foreach (float val in kvp.Value.data)
            {
                dataStream.Write(BitConverter.GetBytes(val), 0, 4);
            }
            long end = dataStream.Position;
            offsets[kvp.Key] = (start, end);
        }
        byte[] dataBlob = dataStream.ToArray();

        Dictionary<string, object> header = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            (long start, long end) = offsets[kvp.Key];
            header[kvp.Key] = new Dictionary<string, object>
            {
                ["dtype"] = kvp.Value.dtype.Name,
                ["shape"] = kvp.Value.shape,
                ["data_offsets"] = new long[] { start, end },
            };
        }

        byte[] headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        string filePath = Path.Combine(_tempDir, name);
        using FileStream fs = new(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new(fs);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        writer.Write(dataBlob);
        return filePath;
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, true);
        }
        catch { }
    }
}
