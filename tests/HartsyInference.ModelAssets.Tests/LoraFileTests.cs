using System.Text;
using System.Text.Json;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.Lora.Mappers;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class LoraFileTests : IDisposable
{
    private readonly string _tempDir;

    public LoraFileTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"lora_tests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void Load_OpensSafetensors_PopulatesFilePath()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = (DType.F32, [4, 320], new float[4 * 320]),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = (DType.F32, [320, 4], new float[320 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "smoke_load", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(path, file.FilePath);
        Assert.NotNull(file.Layers);
        Assert.NotEmpty(file.Layers);
    }

    [Fact]
    public void Load_MissingFile_Throws()
    {
        string missing = Path.Combine(_tempDir, "does_not_exist.safetensors");
        Assert.Throws<FileNotFoundException>(() => LoraFile.Load(missing));
    }

    [Fact]
    public void Dispose_IsIdempotent()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = (DType.F32, [4, 320], new float[4 * 320]),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = (DType.F32, [320, 4], new float[320 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "dispose_check", tensors);

        LoraFile file = LoraFile.Load(path);
        file.Dispose();
        file.Dispose();
    }

    [Theory]
    [InlineData("down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q",
                "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q")]
    [InlineData("up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_out_0",
                "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0")]
    [InlineData("mid_block_attentions_0_transformer_blocks_0_ff_net_0_proj",
                "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj")]
    [InlineData("text_model_encoder_layers_5_self_attn_q_proj",
                "text_model.encoder.layers.5.self_attn.q_proj")]
    [InlineData("text_model_encoder_layers_5_mlp_fc1",
                "text_model.encoder.layers.5.mlp.fc1")]
    [InlineData("text_model_encoder_layers_5_layer_norm1",
                "text_model.encoder.layers.5.layer_norm1")]
    public void LoraKeyTransformer_PreservesCompoundIdentifiers(string input, string expected)
    {
        Assert.Equal(expected, LoraKeyTransformer.UnderscoreToDot(input));
    }

    [Fact]
    public void Detect_KohyaSd15_FromUnetBlocks()
    {
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = (DType.F32, [4, 320], new float[4 * 320]),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = (DType.F32, [320, 4], new float[320 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "sd15_detect", tensors);
        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.KohyaSd15, file.Format);
    }

    [Fact]
    public void Detect_KohyaSdxl_FromTe2()
    {
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = (DType.F32, [4, 320], new float[4 * 320]),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = (DType.F32, [320, 4], new float[320 * 4]),
            ["lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"] = (DType.F32, [4, 1280], new float[4 * 1280]),
            ["lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight"] = (DType.F32, [1280, 4], new float[1280 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "sdxl_detect", tensors);
        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.KohyaSdxl, file.Format);
    }

    [Fact]
    public void Load_KohyaSd15_PopulatesLayers_WithCorrectTargetKey()
    {
        float alphaValue = 8f;
        byte[] alphaBytes = BitConverter.GetBytes(alphaValue);
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] =
                (DType.F32, [4, 320], new float[4 * 320]),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] =
                (DType.F32, [320, 4], new float[320 * 4]),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha"] =
                (DType.F32, [], [alphaValue]),
            ["lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"] =
                (DType.F32, [4, 768], new float[4 * 768]),
            ["lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight"] =
                (DType.F32, [768, 4], new float[768 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "sd15_real", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.KohyaSd15, file.Format);
        Assert.Equal(2, file.Layers.Count);

        LoraLayer unetLayer = file.Layers.Single(l => l.Target == LoraTarget.UNet);
        Assert.Equal("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight", unetLayer.TargetKey);
        Assert.Equal(4, unetLayer.Rank);
        Assert.Equal(8f, unetLayer.Alpha);
        Assert.Equal(LoraVariant.StandardLora, unetLayer.Variant);

        LoraLayer clipLayer = file.Layers.Single(l => l.Target == LoraTarget.ClipL);
        Assert.Equal("text_model.encoder.layers.0.self_attn.q_proj.weight", clipLayer.TargetKey);
        Assert.Equal(4, clipLayer.Rank);
        Assert.Equal(4f, clipLayer.Alpha); // defaults to rank when not stored
    }

    [Fact]
    public void Load_KohyaSdxl_DualClip_RoutesTe1ToClipL_Te2ToClipG()
    {
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            // need a UNet key so SDXL is detected (any down/up/mid_block prefix)
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] =
                (DType.F32, [4, 320], new float[4 * 320]),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] =
                (DType.F32, [320, 4], new float[320 * 4]),
            ["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"] =
                (DType.F32, [4, 768], new float[4 * 768]),
            ["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight"] =
                (DType.F32, [768, 4], new float[768 * 4]),
            ["lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"] =
                (DType.F32, [4, 1280], new float[4 * 1280]),
            ["lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight"] =
                (DType.F32, [1280, 4], new float[1280 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "sdxl_dualclip", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.KohyaSdxl, file.Format);
        Assert.Equal(3, file.Layers.Count);

        LoraLayer clipL = file.Layers.Single(l => l.Target == LoraTarget.ClipL);
        Assert.Equal("text_model.encoder.layers.0.self_attn.q_proj.weight", clipL.TargetKey);
        Assert.Equal(768, clipL.LoraDown.Shape[1]);

        LoraLayer clipG = file.Layers.Single(l => l.Target == LoraTarget.ClipG);
        Assert.Equal("text_model.encoder.layers.0.self_attn.q_proj.weight", clipG.TargetKey);
        Assert.Equal(1280, clipG.LoraDown.Shape[1]);
    }

    [Fact]
    public void Load_UnknownFormat_Throws()
    {
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["random_unrelated_key.weight"] = (DType.F32, [4, 4], new float[16]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "unknown_format", tensors);
        Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => LoraFile.Load(path));
    }

    [Fact]
    public void Detect_KohyaFlux_FromDoubleBlocks()
    {
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight"] = (DType.F32, [4, 3072], new float[4 * 3072]),
            ["lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight"] = (DType.F32, [9216, 4], new float[9216 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "kohya_flux_detect", tensors);
        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.KohyaFlux, file.Format);
    }

    [Fact]
    public void Detect_AiToolkitFlux_FromTransformerPrefix()
    {
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_transformer_transformer_blocks_0_attn_to_q.lora_A.weight"] = (DType.F32, [4, 3072], new float[4 * 3072]),
            ["lora_transformer_transformer_blocks_0_attn_to_q.lora_B.weight"] = (DType.F32, [3072, 4], new float[3072 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "aitk_flux_detect", tensors);
        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.AiToolkitFlux, file.Format);
    }

    [Fact]
    public void Detect_DiffusersFlux_FromTransformerDottedKey()
    {
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"] = (DType.F32, [4, 3072], new float[4 * 3072]),
            ["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"] = (DType.F32, [3072, 4], new float[3072 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "diffusers_flux_detect", tensors);
        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.DiffusersFlux, file.Format);
    }

    [Fact]
    public void Load_KohyaFlux_FusedQkv_Splits3Ways_ImgStream()
    {
        const int hidden = 3072;
        const int rank = 4;
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight"] = (DType.F32, [rank, hidden], new float[rank * hidden]),
            ["lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight"] = (DType.F32, [3 * hidden, rank], new float[3 * hidden * rank]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "kohya_qkv_img", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.KohyaFlux, file.Format);
        Assert.Equal(3, file.Layers.Count);

        string[] expectedKeys =
        [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
        ];
        foreach (string key in expectedKeys)
        {
            LoraLayer layer = file.Layers.Single(l => l.TargetKey == key);
            Assert.Equal(LoraTarget.Transformer, layer.Target);
            Assert.Equal(rank, layer.Rank);
            Assert.Equal(hidden, layer.LoraUp.Shape[0]);
            Assert.Equal(rank, layer.LoraUp.Shape[1]);
            Assert.Equal(rank, layer.LoraDown.Shape[0]);
            Assert.Equal(hidden, layer.LoraDown.Shape[1]);
        }
    }

    [Fact]
    public void Load_KohyaFlux_FusedQkv_Splits3Ways_TxtStream_AddProjNames()
    {
        const int hidden = 3072;
        const int rank = 4;
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_double_blocks_0_txt_attn_qkv.lora_down.weight"] = (DType.F32, [rank, hidden], new float[rank * hidden]),
            ["lora_unet_double_blocks_0_txt_attn_qkv.lora_up.weight"] = (DType.F32, [3 * hidden, rank], new float[3 * hidden * rank]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "kohya_qkv_txt", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(3, file.Layers.Count);
        Assert.Contains(file.Layers, l => l.TargetKey == "transformer_blocks.0.attn.add_q_proj.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "transformer_blocks.0.attn.add_k_proj.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "transformer_blocks.0.attn.add_v_proj.weight");
    }

    [Fact]
    public void Load_KohyaFlux_FusedLinear1_Splits4Ways()
    {
        const int hidden = 3072;
        const int mlpInner = 4 * hidden;
        const int rank = 4;
        const int totalOut = 3 * hidden + mlpInner;
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_single_blocks_0_linear1.lora_down.weight"] = (DType.F32, [rank, hidden], new float[rank * hidden]),
            ["lora_unet_single_blocks_0_linear1.lora_up.weight"] = (DType.F32, [totalOut, rank], new float[totalOut * rank]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "kohya_linear1", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(4, file.Layers.Count);
        LoraLayer toQ = file.Layers.Single(l => l.TargetKey == "single_transformer_blocks.0.attn.to_q.weight");
        LoraLayer toK = file.Layers.Single(l => l.TargetKey == "single_transformer_blocks.0.attn.to_k.weight");
        LoraLayer toV = file.Layers.Single(l => l.TargetKey == "single_transformer_blocks.0.attn.to_v.weight");
        LoraLayer mlp = file.Layers.Single(l => l.TargetKey == "single_transformer_blocks.0.proj_mlp.weight");
        Assert.Equal(hidden, toQ.LoraUp.Shape[0]);
        Assert.Equal(hidden, toK.LoraUp.Shape[0]);
        Assert.Equal(hidden, toV.LoraUp.Shape[0]);
        Assert.Equal(mlpInner, mlp.LoraUp.Shape[0]);
    }

    [Fact]
    public void Load_KohyaFlux_TopLevelMappings()
    {
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            // Block key required for format detection (real Flux LoRAs always have block keys).
            ["lora_unet_double_blocks_0_img_attn_proj.lora_down.weight"] = (DType.F32, [4, 3072], new float[4 * 3072]),
            ["lora_unet_double_blocks_0_img_attn_proj.lora_up.weight"] = (DType.F32, [3072, 4], new float[3072 * 4]),
            ["lora_unet_img_in.lora_down.weight"] = (DType.F32, [4, 64], new float[4 * 64]),
            ["lora_unet_img_in.lora_up.weight"] = (DType.F32, [3072, 4], new float[3072 * 4]),
            ["lora_unet_final_layer_linear.lora_down.weight"] = (DType.F32, [4, 3072], new float[4 * 3072]),
            ["lora_unet_final_layer_linear.lora_up.weight"] = (DType.F32, [64, 4], new float[64 * 4]),
            ["lora_unet_time_in_in_layer.lora_down.weight"] = (DType.F32, [4, 256], new float[4 * 256]),
            ["lora_unet_time_in_in_layer.lora_up.weight"] = (DType.F32, [3072, 4], new float[3072 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "kohya_flux_toplevel", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.KohyaFlux, file.Format);
        Assert.Contains(file.Layers, l => l.TargetKey == "x_embedder.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "proj_out.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "time_text_embed.timestep_embedder.linear_1.weight");
    }

    [Fact]
    public void Load_AiToolkitFlux_PreservesCompoundKeys_AndDefaultsAlphaToRank()
    {
        const int hidden = 3072;
        const int rank = 16;
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_transformer_transformer_blocks_0_attn_to_q.lora_A.weight"] = (DType.F32, [rank, hidden], new float[rank * hidden]),
            ["lora_transformer_transformer_blocks_0_attn_to_q.lora_B.weight"] = (DType.F32, [hidden, rank], new float[hidden * rank]),
            ["lora_transformer_transformer_blocks_0_attn_to_out_0.lora_A.weight"] = (DType.F32, [rank, hidden], new float[rank * hidden]),
            ["lora_transformer_transformer_blocks_0_attn_to_out_0.lora_B.weight"] = (DType.F32, [hidden, rank], new float[hidden * rank]),
            ["lora_transformer_single_transformer_blocks_5_proj_mlp.lora_A.weight"] = (DType.F32, [rank, hidden], new float[rank * hidden]),
            ["lora_transformer_single_transformer_blocks_5_proj_mlp.lora_B.weight"] = (DType.F32, [4 * hidden, rank], new float[4 * hidden * rank]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "aitk_flux_compound", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.AiToolkitFlux, file.Format);
        Assert.Equal(3, file.Layers.Count);

        LoraLayer toQ = file.Layers.Single(l => l.TargetKey == "transformer_blocks.0.attn.to_q.weight");
        Assert.Equal(LoraTarget.Transformer, toQ.Target);
        Assert.Equal(rank, toQ.Rank);
        Assert.Equal(rank, toQ.Alpha); // AI Toolkit folds alpha; default = rank → scale 1.0

        LoraLayer toOut = file.Layers.Single(l => l.TargetKey == "transformer_blocks.0.attn.to_out.0.weight");
        Assert.Equal(LoraTarget.Transformer, toOut.Target);

        LoraLayer projMlp = file.Layers.Single(l => l.TargetKey == "single_transformer_blocks.5.proj_mlp.weight");
        Assert.Equal(LoraTarget.Transformer, projMlp.Target);
    }

    [Fact]
    public void Load_DiffusersFlux_StripsTransformerPrefix()
    {
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"] = (DType.F32, [4, 3072], new float[4 * 3072]),
            ["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"] = (DType.F32, [3072, 4], new float[3072 * 4]),
            ["transformer.single_transformer_blocks.10.proj_out.lora_A.weight"] = (DType.F32, [4, 15360], new float[4 * 15360]),
            ["transformer.single_transformer_blocks.10.proj_out.lora_B.weight"] = (DType.F32, [3072, 4], new float[3072 * 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "diffusers_flux_load", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.DiffusersFlux, file.Format);
        Assert.Equal(2, file.Layers.Count);
        Assert.Contains(file.Layers, l => l.TargetKey == "transformer_blocks.0.attn.to_q.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "single_transformer_blocks.10.proj_out.weight");
    }

    [Fact]
    public unsafe void LoraStack_Apply_ProducesExpectedDelta_ForZeroBaseWeight()
    {
        // Construct a LoRA where down=ones[2,4], up=ones[4,2], alpha=2 (so scale=alpha/rank=1.0).
        // delta[i,j] = sum_r up[i,r] * down[r,j] = sum of 2 ones = 2 (everywhere).
        // Base weight starts at zero → merged should be all 2.0f.
        const int outDim = 4, inDim = 4, rank = 2;
        float[] down = new float[rank * inDim];
        Array.Fill(down, 1.0f);
        float[] up = new float[outDim * rank];
        Array.Fill(up, 1.0f);
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] =
                (DType.F32, [rank, inDim], down),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] =
                (DType.F32, [outDim, rank], up),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha"] =
                (DType.F32, [], [2.0f]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "merge_zero_base", tensors);

        // Base weight: zeros [outDim, inDim], owned (so we can read its data freely).
        Tensor baseW = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        try
        {
            Span<float> baseSpan = baseW.AsSpan<float>();
            baseSpan.Clear();
            Dictionary<string, Tensor> weights = new()
            {
                ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"] = baseW,
            };

            using LoraFile file = LoraFile.Load(path);
            using LoraStack stack = new();
            using CpuBackend backend = new();
            stack.Add(file, strength: 1.0f);
            int merged = stack.ApplyTo(weights, LoraTarget.UNet, backend);

            Assert.Equal(1, merged);
            Tensor result = weights["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"];
            Assert.NotSame(baseW, result);
            Assert.Equal(DType.F32, result.DType);
            Span<float> resultSpan = result.AsSpan<float>();
            for (int i = 0; i < resultSpan.Length; i++)
            {
                Assert.Equal(2.0f, resultSpan[i], 1e-5);
            }
        }
        finally
        {
            baseW.Dispose();
        }
    }

    [Fact]
    public void LoraStack_Apply_StrengthScalesDelta()
    {
        const int outDim = 4, inDim = 4, rank = 2;
        float[] down = new float[rank * inDim];
        Array.Fill(down, 1.0f);
        float[] up = new float[outDim * rank];
        Array.Fill(up, 1.0f);
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = (DType.F32, [rank, inDim], down),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = (DType.F32, [outDim, rank], up),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha"] = (DType.F32, [], [2.0f]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "merge_strength", tensors);

        Tensor baseW = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        try
        {
            baseW.AsSpan<float>().Clear();
            Dictionary<string, Tensor> weights = new()
            {
                ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"] = baseW,
            };
            using LoraFile file = LoraFile.Load(path);
            using LoraStack stack = new();
            using CpuBackend backend = new();
            stack.Add(file, strength: 0.5f);
            stack.ApplyTo(weights, LoraTarget.UNet, backend);

            Tensor result = weights["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"];
            Assert.Equal(1.0f, result.AsSpan<float>()[0], 1e-5);  // 0.5 * 2.0 = 1.0
        }
        finally { baseW.Dispose(); }
    }

    [Fact]
    public void LoraStack_Apply_MultiLoraSumsDeltas()
    {
        // Two LoRAs targeting the same weight key — merge should accumulate both deltas.
        const int outDim = 4, inDim = 4, rank = 2;
        float[] down = new float[rank * inDim]; Array.Fill(down, 1.0f);
        float[] up = new float[outDim * rank]; Array.Fill(up, 1.0f);
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = (DType.F32, [rank, inDim], down),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = (DType.F32, [outDim, rank], up),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha"] = (DType.F32, [], [2.0f]),
        };
        string path1 = CreateSafeTensorsFile(_tempDir, "lora_a", tensors);
        string path2 = CreateSafeTensorsFile(_tempDir, "lora_b", tensors);

        Tensor baseW = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        try
        {
            baseW.AsSpan<float>().Clear();
            Dictionary<string, Tensor> weights = new()
            {
                ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"] = baseW,
            };
            using LoraFile a = LoraFile.Load(path1);
            using LoraFile b = LoraFile.Load(path2);
            using LoraStack stack = new();
            using CpuBackend backend = new();
            stack.Add(a, strength: 1.0f);
            stack.Add(b, strength: 0.5f);
            int merged = stack.ApplyTo(weights, LoraTarget.UNet, backend);

            Assert.Equal(1, merged);
            Tensor result = weights["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"];
            // delta per layer = scale * 2 = 2.0 each. Total: 1.0*2 + 0.5*2 = 3.0
            Assert.Equal(3.0f, result.AsSpan<float>()[0], 1e-5);
        }
        finally { baseW.Dispose(); }
    }

    [Fact]
    public void LoraStack_Apply_MissingTargetKey_LogsAndSkips()
    {
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = (DType.F32, [2, 4], new float[2 * 4]),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = (DType.F32, [4, 2], new float[4 * 2]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "missing_target", tensors);

        Dictionary<string, Tensor> weights = []; // empty — nothing matches
        using LoraFile file = LoraFile.Load(path);
        using LoraStack stack = new();
        using CpuBackend backend = new();
        stack.Add(file);
        int merged = stack.ApplyTo(weights, LoraTarget.UNet, backend);
        Assert.Equal(0, merged);
    }

    [Fact]
    public void LoraStack_Apply_RejectsFp8Base()
    {
        const int outDim = 4, inDim = 4, rank = 2;
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = (DType.F32, [rank, inDim], new float[rank * inDim]),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = (DType.F32, [outDim, rank], new float[outDim * rank]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "fp8_base", tensors);

        Tensor baseW = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        try
        {
            baseW.Fp8ScaleFactor = 0.001f; // simulate FP8-quantized base
            Dictionary<string, Tensor> weights = new()
            {
                ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"] = baseW,
            };
            using LoraFile file = LoraFile.Load(path);
            using LoraStack stack = new();
            using CpuBackend backend = new();
            stack.Add(file);
            Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(
                () => stack.ApplyTo(weights, LoraTarget.UNet, backend));
        }
        finally { baseW.Dispose(); }
    }

    [Fact]
    public void LoraStack_AddFromPath_OwnsFile_DisposedWithStack()
    {
        const int outDim = 4, inDim = 4, rank = 2;
        float[] down = new float[rank * inDim]; Array.Fill(down, 1.0f);
        float[] up = new float[outDim * rank]; Array.Fill(up, 1.0f);
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = (DType.F32, [rank, inDim], down),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = (DType.F32, [outDim, rank], up),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha"] = (DType.F32, [], [2.0f]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "addfrompath", tensors);

        Tensor baseW = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        try
        {
            baseW.AsSpan<float>().Clear();
            Dictionary<string, Tensor> weights = new()
            {
                ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"] = baseW,
            };
            using LoraStack stack = new();
            using CpuBackend backend = new();
            stack.AddFromPath(path, strength: 1.0f);
            int merged = stack.ApplyTo(weights, LoraTarget.UNet, backend);
            Assert.Equal(1, merged);
        }
        finally { baseW.Dispose(); }
    }

    [Fact]
    public void LoraStack_ApplyToWeights_RoutesPerTarget()
    {
        const int outDim = 4, inDim = 4, rank = 2;
        float[] downData = new float[rank * inDim]; Array.Fill(downData, 1.0f);
        float[] upData = new float[outDim * rank]; Array.Fill(upData, 1.0f);
        Dictionary<string, (DType, long[], float[])> tensors = new()
        {
            // SDXL detection requires UNet keys + lora_te2_ presence.
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = (DType.F32, [rank, inDim], downData),
            ["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = (DType.F32, [outDim, rank], upData),
            ["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"] = (DType.F32, [rank, inDim], downData),
            ["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight"] = (DType.F32, [outDim, rank], upData),
            ["lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"] = (DType.F32, [rank, inDim], downData),
            ["lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight"] = (DType.F32, [outDim, rank], upData),
        };
        string path = CreateSafeTensorsFile(_tempDir, "apply_routes", tensors);

        Tensor unetBase = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        Tensor clipLBase = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        Tensor clipGBase = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        try
        {
            unetBase.AsSpan<float>().Clear();
            clipLBase.AsSpan<float>().Clear();
            clipGBase.AsSpan<float>().Clear();
            Dictionary<string, Tensor> unetWeights = new()
            {
                ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"] = unetBase,
            };
            Dictionary<string, Tensor> clipLWeights = new()
            {
                ["text_model.encoder.layers.0.self_attn.q_proj.weight"] = clipLBase,
            };
            Dictionary<string, Tensor> clipGWeights = new()
            {
                ["text_model.encoder.layers.0.self_attn.q_proj.weight"] = clipGBase,
            };

            using LoraStack stack = new();
            using CpuBackend backend = new();
            stack.AddFromPath(path);
            int total = stack.ApplyToWeights(backend,
                unetWeights: unetWeights,
                clipLWeights: clipLWeights,
                clipGWeights: clipGWeights);

            Assert.Equal(3, total); // one per component
        }
        finally
        {
            unetBase.Dispose();
            clipLBase.Dispose();
            clipGBase.Dispose();
        }
    }

    [Fact]
    public void LoraLayer_ConstructionWithRequiredFields()
    {
        Tensor down = new Tensor(new TensorShape(4, 8), DType.F32);
        Tensor up = new Tensor(new TensorShape(8, 4), DType.F32);
        try
        {
            LoraLayer layer = new()
            {
                TargetKey = "transformer_blocks.0.attn.to_q.weight",
                Target = LoraTarget.Transformer,
                LoraDown = down,
                LoraUp = up,
                Alpha = 4f,
                Rank = 4,
                Variant = LoraVariant.StandardLora,
            };

            Assert.Equal("transformer_blocks.0.attn.to_q.weight", layer.TargetKey);
            Assert.Equal(LoraTarget.Transformer, layer.Target);
            Assert.Equal(4, layer.Rank);
            Assert.Equal(4f, layer.Alpha);
            Assert.Equal(LoraVariant.StandardLora, layer.Variant);
            Assert.Same(down, layer.LoraDown);
            Assert.Same(up, layer.LoraUp);
        }
        finally
        {
            down.Dispose();
            up.Dispose();
        }
    }

    private static string CreateSafeTensorsFile(string dir, string name,
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors)
    {
        using MemoryStream dataStream = new();
        Dictionary<string, (long start, long end)> offsets = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            long start = dataStream.Position;
            foreach (float val in kvp.Value.data)
            {
                byte[] bytes = BitConverter.GetBytes(val);
                dataStream.Write(bytes, 0, bytes.Length);
            }
            long end = dataStream.Position;
            offsets[kvp.Key] = (start, end);
        }
        byte[] dataBlob = dataStream.ToArray();

        Dictionary<string, object> headerDict = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            (long start, long end) = offsets[kvp.Key];
            headerDict[kvp.Key] = new Dictionary<string, object>
            {
                ["dtype"] = kvp.Value.dtype.Name,
                ["shape"] = kvp.Value.shape,
                ["data_offsets"] = new long[] { start, end },
            };
        }

        string headerJson = JsonSerializer.Serialize(headerDict);
        byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);

        string filePath = Path.Combine(dir, $"{name}.safetensors");
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
            if (Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, true);
            }
        }
        catch
        {
            // best-effort cleanup
        }
    }
}
