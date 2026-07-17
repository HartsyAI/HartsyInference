using System;
using System.Collections.Generic;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Vision.Segmentation;
using HartsyInference.Vision.Segmentation.Sam2;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Forward-pass smoke tests for the SAM 2 Hiera image encoder, two-way mask decoder, and full
/// <see cref="SamPipeline"/> over synthetic (deterministic random) weights. These exercise the model code end
/// to end for shape-correctness; numeric parity against real Meta SAM 2 weights is pending (see the
/// env-gated <see cref="Sam2_RealWeightParity_Skips"/> stub). Tagged SyntheticSmoke: they run synthetic
/// weights through model code and are not part of the no-GPU Unit gate.</summary>
public sealed class Sam2ForwardTests
{
    // Tiny config: 4 stages of depth 1, small dims, 256² input → S/16 = 16² feature grid, 32-ch embeddings.
    private static Sam2Config TinyConfig() => new()
    {
        ImageSize = 256,
        EncoderEmbedDim = 16,
        EncoderStageDepths = [1, 1, 1, 1],
        EncoderNumHeads = 1,
        EncoderWindowSizes = [8, 4, 2, 1],
        PromptEmbedDim = 32,
        NumMultimaskOutputs = 3,
        DecoderDepth = 2,
        DecoderNumHeads = 4,
    };

    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void HieraImageEncoder_Forward_ProducesStride16Embedding()
    {
        using IBackend backend = new CpuBackend();
        Sam2Config cfg = TinyConfig();

        HieraImageEncoder encoder = new HieraImageEncoder(cfg);
        encoder.LoadWeights(BuildEncoderWeights(cfg, seed: 11), string.Empty);

        Tensor image = Fill([1, 3, cfg.ImageSize, cfg.ImageSize], seed: 99);
        Tensor embedding = encoder.Forward(backend, image);

        int expected = cfg.ImageSize / encoder.FeatureStride; // 256 / 16 = 16
        Assert.Equal(4, embedding.Shape.Rank);
        Assert.Equal(1, (int)embedding.Shape[0]);
        Assert.Equal(cfg.PromptEmbedDim, (int)embedding.Shape[1]);
        Assert.Equal(expected, (int)embedding.Shape[2]);
        Assert.Equal(expected, (int)embedding.Shape[3]);
        AssertAllFinite(embedding);
    }

    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void SamMaskDecoder_Forward_ReturnsMultimaskLogitsAndIou()
    {
        using IBackend backend = new CpuBackend();
        Sam2Config cfg = TinyConfig();
        int dim = cfg.PromptEmbedDim;
        int hf = 16, wf = 16;

        SamMaskDecoder decoder = new SamMaskDecoder(cfg);
        decoder.LoadWeights(BuildDecoderWeights(cfg, seed: 7), string.Empty);

        Tensor embedding = Fill([1, dim, hf, wf], seed: 21);
        Tensor imagePe = Fill([1, dim, hf, wf], seed: 22);
        Tensor sparse = Fill([1, 2, dim], seed: 23); // two prompt tokens

        (Tensor maskLogits, Tensor iou) = decoder.Forward(backend, embedding, imagePe, sparse, null, multimaskOutput: true);

        Assert.Equal(4, maskLogits.Shape.Rank);
        Assert.Equal(1, (int)maskLogits.Shape[0]);
        Assert.Equal(cfg.NumMultimaskOutputs, (int)maskLogits.Shape[1]); // 3 multimask outputs
        Assert.Equal(4 * hf, (int)maskLogits.Shape[2]);
        Assert.Equal(4 * wf, (int)maskLogits.Shape[3]);

        Assert.Equal(2, iou.Shape.Rank);
        Assert.Equal(1, (int)iou.Shape[0]);
        Assert.Equal(cfg.NumMultimaskOutputs, (int)iou.Shape[1]);
        AssertAllFinite(maskLogits);
        AssertAllFinite(iou);

        // Single-mask path returns exactly one mask token.
        (Tensor single, Tensor singleIou) = decoder.Forward(backend, embedding, imagePe, sparse, null, multimaskOutput: false);
        Assert.Equal(1, (int)single.Shape[1]);
        Assert.Equal(1, (int)singleIou.Shape[1]);
    }

    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void SamPipeline_Segment_ReturnsMasksAtRequestedSize()
    {
        using IBackend backend = new CpuBackend();
        Sam2Config cfg = TinyConfig();

        HieraImageEncoder encoder = new HieraImageEncoder(cfg);
        encoder.LoadWeights(BuildEncoderWeights(cfg, seed: 3), string.Empty);
        SamPromptEncoder promptEncoder = SamPipeline.BuildPromptEncoder(BuildPromptWeights(cfg, seed: 5));
        SamMaskDecoder decoder = new SamMaskDecoder(cfg);
        decoder.LoadWeights(BuildDecoderWeights(cfg, seed: 7), string.Empty);

        using SamPipeline pipeline = new SamPipeline(backend, cfg, encoder, promptEncoder, decoder);
        Assert.Equal("sam2", pipeline.ModelName);

        Tensor image = Fill([1, 3, cfg.ImageSize, cfg.ImageSize], seed: 42);
        SamPrompt prompt = new SamPrompt
        {
            Points = [new SamPointPrompt(120, 90, Foreground: true)],
            Box = new SamBoxPrompt(20, 20, 200, 180),
        };

        SamMaskResult[] results = pipeline.Segment(prompt, image, cfg.ImageSize, cfg.ImageSize);

        Assert.Equal(cfg.NumMultimaskOutputs, results.Length);
        foreach (SamMaskResult r in results)
        {
            Assert.Equal(cfg.ImageSize, r.Width);
            Assert.Equal(cfg.ImageSize, r.Height);
            Assert.Equal(cfg.ImageSize * cfg.ImageSize, r.Mask.Length);
            Assert.InRange(r.IoUScore, 0f, 1f);
        }
        // Results are best-first (non-increasing IoU).
        for (int i = 1; i < results.Length; i++)
            Assert.True(results[i - 1].IoUScore >= results[i].IoUScore);
    }

    [Trait("Category", "Integration")]
    [Fact]
    public void Sam2_RealWeightParity_Skips()
    {
        // Real-weight parity: set SAM2_CHECKPOINT_PATH to a Meta sam2 .safetensors to exercise the full load
        // path. Skips cleanly (no GPU, no weights) so it never gates the Unit lane.
        string? path = Environment.GetEnvironmentVariable("SAM2_CHECKPOINT_PATH");
        if (string.IsNullOrEmpty(path) || !System.IO.File.Exists(path))
        {
            return;
        }

        using IBackend backend = new CpuBackend();
        Sam2Config cfg = Sam2Config.HieraBasePlus;
        using SamPipeline pipeline = SamPipeline.Load(backend, cfg, path);

        Tensor image = Fill([1, 3, cfg.ImageSize, cfg.ImageSize], seed: 1);
        SamPrompt prompt = new SamPrompt { Points = [new SamPointPrompt(512, 512, true)] };
        SamMaskResult[] results = pipeline.Segment(prompt, image, cfg.ImageSize, cfg.ImageSize);
        Assert.NotEmpty(results);
    }

    // ── Synthetic weight builders (mirror the checkpoint key layout each component loads) ──────────

    private static Dictionary<string, Tensor> BuildEncoderWeights(Sam2Config cfg, int seed)
    {
        int embed = cfg.EncoderEmbedDim;
        Dictionary<string, Tensor> w = new();
        int s = seed;

        w["trunk.patch_embed.proj.weight"] = Fill([embed, 3, 7, 7], s++);
        w["trunk.patch_embed.proj.bias"] = Fill([embed], s++);
        w["trunk.pos_embed"] = Fill([1, embed, 14, 14], s++);
        w["trunk.pos_embed_window"] = Fill([1, embed, cfg.EncoderWindowSizes[0], cfg.EncoderWindowSizes[0]], s++);

        // Block schedule for depths [1,1,1,1]: dim/heads double each stage; blocks 1..3 pool + project.
        (int dimIn, int dimOut)[] blocks =
        [
            (embed, embed),         // block 0
            (embed, embed * 2),     // block 1
            (embed * 2, embed * 4), // block 2
            (embed * 4, embed * 8), // block 3
        ];
        for (int i = 0; i < blocks.Length; i++)
        {
            (int dimIn, int dimOut) = blocks[i];
            string p = $"trunk.blocks.{i}.";
            w[$"{p}norm1.weight"] = Fill([dimIn], s++);
            w[$"{p}norm1.bias"] = Fill([dimIn], s++);
            w[$"{p}attn.qkv.weight"] = Fill([3 * dimOut, dimIn], s++);
            w[$"{p}attn.qkv.bias"] = Fill([3 * dimOut], s++);
            w[$"{p}attn.proj.weight"] = Fill([dimOut, dimOut], s++);
            w[$"{p}attn.proj.bias"] = Fill([dimOut], s++);
            w[$"{p}norm2.weight"] = Fill([dimOut], s++);
            w[$"{p}norm2.bias"] = Fill([dimOut], s++);
            w[$"{p}mlp.layers.0.weight"] = Fill([4 * dimOut, dimOut], s++);
            w[$"{p}mlp.layers.0.bias"] = Fill([4 * dimOut], s++);
            w[$"{p}mlp.layers.1.weight"] = Fill([dimOut, 4 * dimOut], s++);
            w[$"{p}mlp.layers.1.bias"] = Fill([dimOut], s++);
            if (dimIn != dimOut)
            {
                w[$"{p}proj.weight"] = Fill([dimOut, dimIn], s++);
                w[$"{p}proj.bias"] = Fill([dimOut], s++);
            }
        }

        // Neck: neck.convs.{3-stage} maps to descending resolution (channel list 8C,4C,2C,C).
        int pd = cfg.PromptEmbedDim;
        int[] stageChannels = [embed, embed * 2, embed * 4, embed * 8];
        for (int stage = 0; stage < 4; stage++)
        {
            w[$"neck.convs.{3 - stage}.conv.weight"] = Fill([pd, stageChannels[stage], 1, 1], s++);
            w[$"neck.convs.{3 - stage}.conv.bias"] = Fill([pd], s++);
        }
        return w;
    }

    private static Dictionary<string, Tensor> BuildPromptWeights(Sam2Config cfg, int seed)
    {
        int dim = cfg.PromptEmbedDim;
        Dictionary<string, Tensor> w = new();
        w["pe_layer.positional_encoding_gaussian_matrix"] = Fill([2, dim / 2], seed);
        for (int i = 0; i < 4; i++) w[$"point_embeddings.{i}.weight"] = Fill([1, dim], seed + 1 + i);
        w["not_a_point_embed.weight"] = Fill([1, dim], seed + 5);
        return w;
    }

    private static Dictionary<string, Tensor> BuildDecoderWeights(Sam2Config cfg, int seed)
    {
        int dim = cfg.PromptEmbedDim;      // 32
        int heads = cfg.DecoderNumHeads;   // 4
        int selfInternal = dim;            // ratio-1 self attention
        int crossInternal = dim / 2;       // ratio-2 cross attention
        int mlpHidden = dim * 2;           // 64
        int numMaskTokens = cfg.NumMultimaskOutputs + 1; // 4
        int hyperDim = dim / 8;            // 4
        Dictionary<string, Tensor> w = new();
        int s = seed;

        w["iou_token.weight"] = Fill([1, dim], s++);
        w["mask_tokens.weight"] = Fill([numMaskTokens, dim], s++);

        for (int l = 0; l < cfg.DecoderDepth; l++)
        {
            string p = $"transformer.layers.{l}.";
            AddAttn(w, $"{p}self_attn.", dim, selfInternal, ref s);
            AddAttn(w, $"{p}cross_attn_token_to_image.", dim, crossInternal, ref s);
            AddAttn(w, $"{p}cross_attn_image_to_token.", dim, crossInternal, ref s);
            for (int nrm = 1; nrm <= 4; nrm++)
            {
                w[$"{p}norm{nrm}.weight"] = Fill([dim], s++);
                w[$"{p}norm{nrm}.bias"] = Fill([dim], s++);
            }
            w[$"{p}mlp.lin1.weight"] = Fill([mlpHidden, dim], s++);
            w[$"{p}mlp.lin1.bias"] = Fill([mlpHidden], s++);
            w[$"{p}mlp.lin2.weight"] = Fill([dim, mlpHidden], s++);
            w[$"{p}mlp.lin2.bias"] = Fill([dim], s++);
        }

        AddAttn(w, "transformer.final_attn_token_to_image.", dim, crossInternal, ref s);
        w["transformer.norm_final_attn.weight"] = Fill([dim], s++);
        w["transformer.norm_final_attn.bias"] = Fill([dim], s++);

        w["output_upscaling.0.weight"] = Fill([dim, dim / 4, 2, 2], s++);
        w["output_upscaling.0.bias"] = Fill([dim / 4], s++);
        w["output_upscaling.1.weight"] = Fill([dim / 4], s++);
        w["output_upscaling.1.bias"] = Fill([dim / 4], s++);
        w["output_upscaling.3.weight"] = Fill([dim / 4, hyperDim, 2, 2], s++);
        w["output_upscaling.3.bias"] = Fill([hyperDim], s++);

        for (int i = 0; i < numMaskTokens; i++)
        {
            AddMlpHead(w, $"output_hypernetworks_mlps.{i}.", dim, hyperDim, ref s);
        }
        AddMlpHead(w, "iou_prediction_head.", dim, numMaskTokens, ref s);
        return w;
    }

    private static void AddAttn(Dictionary<string, Tensor> w, string prefix, int embedDim, int internalDim, ref int s)
    {
        w[$"{prefix}q_proj.weight"] = Fill([internalDim, embedDim], s++);
        w[$"{prefix}q_proj.bias"] = Fill([internalDim], s++);
        w[$"{prefix}k_proj.weight"] = Fill([internalDim, embedDim], s++);
        w[$"{prefix}k_proj.bias"] = Fill([internalDim], s++);
        w[$"{prefix}v_proj.weight"] = Fill([internalDim, embedDim], s++);
        w[$"{prefix}v_proj.bias"] = Fill([internalDim], s++);
        w[$"{prefix}out_proj.weight"] = Fill([embedDim, internalDim], s++);
        w[$"{prefix}out_proj.bias"] = Fill([embedDim], s++);
    }

    private static void AddMlpHead(Dictionary<string, Tensor> w, string prefix, int dim, int outDim, ref int s)
    {
        w[$"{prefix}layers.0.weight"] = Fill([dim, dim], s++);
        w[$"{prefix}layers.0.bias"] = Fill([dim], s++);
        w[$"{prefix}layers.1.weight"] = Fill([dim, dim], s++);
        w[$"{prefix}layers.1.bias"] = Fill([dim], s++);
        w[$"{prefix}layers.2.weight"] = Fill([outDim, dim], s++);
        w[$"{prefix}layers.2.bias"] = Fill([outDim], s++);
    }

    private static unsafe Tensor Fill(int[] dims, int seed)
    {
        long[] longDims = new long[dims.Length];
        for (int i = 0; i < dims.Length; i++) longDims[i] = dims[i];
        Tensor t = new Tensor(new TensorShape(longDims), DType.F32);
        Span<float> data = t.AsSpan<float>();
        uint state = (uint)seed * 2654435761u + 1u;
        for (int i = 0; i < data.Length; i++)
        {
            state = state * 1664525u + 1013904223u;
            // Small, zero-centered values keep attention scores + norms well-conditioned.
            data[i] = ((state >> 8) / (float)(1 << 24) - 0.5f) * 0.1f;
        }
        return t;
    }

    private static unsafe void AssertAllFinite(Tensor t)
    {
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        ReadOnlySpan<float> data = f.AsReadOnlySpan<float>();
        foreach (float v in data)
            Assert.True(float.IsFinite(v), "tensor holds a non-finite value");
    }
}
