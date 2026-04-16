using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.UNetBlocks;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;
using Xunit;

namespace SharpInference.Diffusion.Tests;

/// <summary>Smoke tests verifying the complete pipeline code path with tiny random weights. Does not produce meaningful images — validates structure only.</summary>
public sealed class PipelineSmokeTests : IDisposable
{
    private readonly CpuBackend _backend;

    public PipelineSmokeTests()
    {
        _backend = new CpuBackend();
    }

    [Fact]
    public void SeedGenerator_ProducesGaussianNoise()
    {
        TensorShape shape = new TensorShape(1, 4, 8, 8);
        Tensor noise = SeedGenerator.CreateNoise(shape, 42);

        Assert.Equal(shape, noise.Shape);
        Assert.Equal(DType.F32, noise.DType);

        // Check it's not all zeros
        ReadOnlySpan<float> data = noise.AsReadOnlySpan<float>();
        bool hasNonZero = false;
        for (int i = 0; i < data.Length; i++)
        {
            if (MathF.Abs(data[i]) > 1e-6f)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "Noise tensor should contain non-zero values");

        // Same seed = same noise
        Tensor noise2 = SeedGenerator.CreateNoise(shape, 42);
        ReadOnlySpan<float> data2 = noise2.AsReadOnlySpan<float>();
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], data2[i]);
        }

        noise.Dispose();
        noise2.Dispose();
    }

    [Fact]
    public void SeedGenerator_DifferentSeeds_DifferentNoise()
    {
        TensorShape shape = new TensorShape(1, 4, 8, 8);
        Tensor noise1 = SeedGenerator.CreateNoise(shape, 42);
        Tensor noise2 = SeedGenerator.CreateNoise(shape, 123);

        ReadOnlySpan<float> d1 = noise1.AsReadOnlySpan<float>();
        ReadOnlySpan<float> d2 = noise2.AsReadOnlySpan<float>();
        bool anyDifferent = false;
        for (int i = 0; i < d1.Length; i++)
        {
            if (MathF.Abs(d1[i] - d2[i]) > 1e-6f)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent);

        noise1.Dispose();
        noise2.Dispose();
    }

    [Fact]
    public unsafe void ImagePostProcessor_TensorToRgb_CorrectSize()
    {
        // Create a simple [1, 3, 4, 4] tensor with known values
        TensorShape shape = new TensorShape(1, 3, 4, 4);
        Tensor img = new Tensor(shape, DType.F32);
        float* ptr = (float*)img.DataPointer;

        // Fill with constant value -1 (should map to 0 after post-processing)
        for (int i = 0; i < shape.ElementCount; i++)
        {
            ptr[i] = -1.0f;
        }

        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(img);
        Assert.Equal(4 * 4 * 3, rgb.Length);

        // All pixels should be 0 (black) since input is -1
        for (int i = 0; i < rgb.Length; i++)
        {
            Assert.Equal(0, rgb[i]);
        }

        // Now fill with +1 (should map to 255 after post-processing)
        for (int i = 0; i < shape.ElementCount; i++)
        {
            ptr[i] = 1.0f;
        }
        byte[] rgbWhite = ImagePostProcessor.TensorToRgbBytes(img);
        for (int i = 0; i < rgbWhite.Length; i++)
        {
            Assert.Equal(255, rgbWhite[i]);
        }

        img.Dispose();
    }

    [Fact]
    public void ImagePostProcessor_SaveBmp_CreatesValidFile()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid():N}.bmp");
        try
        {
            byte[] rgb = new byte[8 * 8 * 3];
            // Fill with red
            for (int i = 0; i < rgb.Length; i += 3)
            {
                rgb[i] = 255;     // R
                rgb[i + 1] = 0;   // G
                rgb[i + 2] = 0;   // B
            }

            ImagePostProcessor.SaveBmp(tempPath, rgb, 8, 8);

            Assert.True(File.Exists(tempPath));
            byte[] bmpData = File.ReadAllBytes(tempPath);
            // BMP magic bytes
            Assert.Equal((byte)'B', bmpData[0]);
            Assert.Equal((byte)'M', bmpData[1]);
            // File size should be 54 (header) + padded pixel data
            Assert.True(bmpData.Length >= 54);
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void TimestepEmbedding_Forward_ProducesCorrectShape()
    {
        // Tiny config: embeddingDim=16, timeDim=32
        int embDim = 16;
        int timeDim = 32;
        int batch = 1;

        TimestepEmbedding te = new TimestepEmbedding(embDim, timeDim);

        // Create fake weights
        Dictionary<string, Tensor> weights = new()
        {
            ["te.linear_1.weight"] = CreateRandomTensor(timeDim, embDim),
            ["te.linear_1.bias"] = CreateRandomTensor(timeDim),
            ["te.linear_2.weight"] = CreateRandomTensor(timeDim, timeDim),
            ["te.linear_2.bias"] = CreateRandomTensor(timeDim),
        };
        te.LoadWeights(weights, "te");

        Span<float> timesteps = stackalloc float[] { 500.0f };
        Tensor result = te.Forward(_backend, timesteps, batch);

        Assert.Equal(2, result.Shape.Rank);
        Assert.Equal(batch, result.Shape[0]);
        Assert.Equal(timeDim, result.Shape[1]);

        result.Dispose();
        DisposeWeights(weights);
    }

    [Fact]
    public void UNetResNetBlock_Forward_ProducesCorrectShape()
    {
        int inCh = 16;
        int outCh = 32;
        int timeDim = 64;
        int batch = 1;
        int h = 4, w = 4;

        UNetResNetBlock block = new UNetResNetBlock(inCh, outCh, timeDim);

        Dictionary<string, Tensor> weights = BuildResNetWeights("r", inCh, outCh, timeDim);
        block.LoadWeights(weights, "r");

        Tensor input = CreateRandomTensor(batch, inCh, h, w);
        Tensor temb = CreateRandomTensor(batch, timeDim);

        Tensor output = block.Forward(_backend, input, temb);

        Assert.Equal(4, output.Shape.Rank);
        Assert.Equal(batch, output.Shape[0]);
        Assert.Equal(outCh, output.Shape[1]);
        Assert.Equal(h, output.Shape[2]);
        Assert.Equal(w, output.Shape[3]);

        output.Dispose();
        input.Dispose();
        temb.Dispose();
        DisposeWeights(weights);
    }

    [Fact]
    public void ClipTextEncoder_Forward_ProducesCorrectShape()
    {
        // Tiny CLIP config for testing
        ClipTextEncoderConfig config = new ClipTextEncoderConfig
        {
            HiddenSize = 16,
            IntermediateSize = 32,
            NumLayers = 1,
            NumHeads = 2,
            MaxPositionEmbeddings = 8,
            VocabSize = 100,
            UseQuickGelu = true,
        };

        ClipTextEncoder encoder = new ClipTextEncoder(config);

        Dictionary<string, Tensor> weights = BuildClipWeights("t", config);
        encoder.LoadWeights(weights, "t");

        int[][] batchTokenIds = [[0, 1, 2, 3, 4, 5, 6, 7]];
        Tensor result = encoder.Encode(_backend, batchTokenIds);

        Assert.Equal(3, result.Shape.Rank);
        Assert.Equal(1, result.Shape[0]); // batch
        Assert.Equal(8, result.Shape[1]); // seqLen
        Assert.Equal(16, result.Shape[2]); // hiddenSize

        result.Dispose();
        DisposeWeights(weights);
    }

    [Fact]
    public void VaeDecoder_Forward_ProducesCorrectShape()
    {
        // Use SD1.5 config with real architecture but random weights
        VaeConfig config = VaeConfig.Sd15;
        VaeDecoder decoder = new VaeDecoder(config);

        Dictionary<string, Tensor> weights = BuildVaeDecoderWeights(config);
        decoder.LoadWeights(weights);

        // Input: [1, 4, 4, 4] latent (32x32 pixel output)
        Tensor latent = CreateRandomTensor(1, 4, 4, 4);
        Tensor output = decoder.Decode(_backend, latent);

        Assert.Equal(4, output.Shape.Rank);
        Assert.Equal(1, output.Shape[0]); // batch
        Assert.Equal(3, output.Shape[1]); // RGB
        Assert.Equal(32, output.Shape[2]); // H * 8
        Assert.Equal(32, output.Shape[3]); // W * 8

        output.Dispose();
        latent.Dispose();
        DisposeWeights(weights);
    }

    // ─── Helper methods ──────────────────────────────────────────────

    private static unsafe Tensor CreateRandomTensor(params long[] dims)
    {
        TensorShape shape = dims.Length switch
        {
            1 => new TensorShape(dims[0]),
            2 => new TensorShape(dims[0], dims[1]),
            3 => new TensorShape(dims[0], dims[1], dims[2]),
            4 => new TensorShape(dims[0], dims[1], dims[2], dims[3]),
            _ => new TensorShape(dims),
        };

        Tensor t = new Tensor(shape, DType.F32);
        Random rng = new Random(42);
        float* ptr = (float*)t.DataPointer;
        int count = (int)shape.ElementCount;

        // Small random values to prevent numerical explosion
        for (int i = 0; i < count; i++)
        {
            ptr[i] = (float)(rng.NextDouble() * 0.02 - 0.01);
        }

        return t;
    }

    private static Dictionary<string, Tensor> BuildResNetWeights(string prefix, int inCh, int outCh, int timeDim)
    {
        Dictionary<string, Tensor> w = new()
        {
            [$"{prefix}.norm1.weight"] = CreateRandomTensor(inCh),
            [$"{prefix}.norm1.bias"] = CreateRandomTensor(inCh),
            [$"{prefix}.conv1.weight"] = CreateRandomTensor(outCh, inCh, 3, 3),
            [$"{prefix}.conv1.bias"] = CreateRandomTensor(outCh),
            [$"{prefix}.time_emb_proj.weight"] = CreateRandomTensor(outCh, timeDim),
            [$"{prefix}.time_emb_proj.bias"] = CreateRandomTensor(outCh),
            [$"{prefix}.norm2.weight"] = CreateRandomTensor(outCh),
            [$"{prefix}.norm2.bias"] = CreateRandomTensor(outCh),
            [$"{prefix}.conv2.weight"] = CreateRandomTensor(outCh, outCh, 3, 3),
            [$"{prefix}.conv2.bias"] = CreateRandomTensor(outCh),
        };

        if (inCh != outCh)
        {
            w[$"{prefix}.conv_shortcut.weight"] = CreateRandomTensor(outCh, inCh, 1, 1);
            w[$"{prefix}.conv_shortcut.bias"] = CreateRandomTensor(outCh);
        }

        return w;
    }

    private static Dictionary<string, Tensor> BuildClipWeights(string prefix, ClipTextEncoderConfig config)
    {
        int h = config.HiddenSize;
        int inter = config.IntermediateSize;
        Dictionary<string, Tensor> w = new()
        {
            [$"{prefix}.embeddings.token_embedding.weight"] = CreateRandomTensor(config.VocabSize, h),
            [$"{prefix}.embeddings.position_embedding.weight"] = CreateRandomTensor(config.MaxPositionEmbeddings, h),
            [$"{prefix}.final_layer_norm.weight"] = CreateRandomTensor(h),
            [$"{prefix}.final_layer_norm.bias"] = CreateRandomTensor(h),
        };

        for (int i = 0; i < config.NumLayers; i++)
        {
            string lp = $"{prefix}.encoder.layers.{i}";
            w[$"{lp}.layer_norm1.weight"] = CreateRandomTensor(h);
            w[$"{lp}.layer_norm1.bias"] = CreateRandomTensor(h);
            w[$"{lp}.self_attn.q_proj.weight"] = CreateRandomTensor(h, h);
            w[$"{lp}.self_attn.q_proj.bias"] = CreateRandomTensor(h);
            w[$"{lp}.self_attn.k_proj.weight"] = CreateRandomTensor(h, h);
            w[$"{lp}.self_attn.k_proj.bias"] = CreateRandomTensor(h);
            w[$"{lp}.self_attn.v_proj.weight"] = CreateRandomTensor(h, h);
            w[$"{lp}.self_attn.v_proj.bias"] = CreateRandomTensor(h);
            w[$"{lp}.self_attn.out_proj.weight"] = CreateRandomTensor(h, h);
            w[$"{lp}.self_attn.out_proj.bias"] = CreateRandomTensor(h);
            w[$"{lp}.layer_norm2.weight"] = CreateRandomTensor(h);
            w[$"{lp}.layer_norm2.bias"] = CreateRandomTensor(h);
            w[$"{lp}.mlp.fc1.weight"] = CreateRandomTensor(inter, h);
            w[$"{lp}.mlp.fc1.bias"] = CreateRandomTensor(inter);
            w[$"{lp}.mlp.fc2.weight"] = CreateRandomTensor(h, inter);
            w[$"{lp}.mlp.fc2.bias"] = CreateRandomTensor(h);
        }

        return w;
    }

    private static Dictionary<string, Tensor> BuildVaeDecoderWeights(VaeConfig config)
    {
        int[] blockCh = config.BlockOutChannels;
        int midCh = blockCh[^1];
        int latentCh = config.LatentChannels;
        int numBlocks = blockCh.Length;
        int resnetsPerBlock = config.LayersPerBlock + 1;

        Dictionary<string, Tensor> w = new();

        // post_quant_conv
        if (config.UsePostQuantConv)
        {
            w["post_quant_conv.weight"] = CreateRandomTensor(latentCh, latentCh, 1, 1);
            w["post_quant_conv.bias"] = CreateRandomTensor(latentCh);
        }

        // conv_in
        w["decoder.conv_in.weight"] = CreateRandomTensor(midCh, latentCh, 3, 3);
        w["decoder.conv_in.bias"] = CreateRandomTensor(midCh);

        // mid_block
        AddResNetWeights(w, "decoder.mid_block.resnets.0", midCh, midCh, config.NormNumGroups);
        AddVaeAttentionWeights(w, "decoder.mid_block.attentions.0", midCh, config.NormNumGroups);
        AddResNetWeights(w, "decoder.mid_block.resnets.1", midCh, midCh, config.NormNumGroups);

        // up_blocks (reversed channel order)
        int[] reversedCh = new int[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            reversedCh[i] = blockCh[numBlocks - 1 - i];
        }

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int outCh = reversedCh[blockIdx];

            for (int resIdx = 0; resIdx < resnetsPerBlock; resIdx++)
            {
                int inCh;
                if (resIdx == 0)
                {
                    inCh = blockIdx == 0 ? midCh : reversedCh[blockIdx - 1];
                }
                else
                {
                    inCh = outCh;
                }
                AddResNetWeights(w, $"decoder.up_blocks.{blockIdx}.resnets.{resIdx}", inCh, outCh, config.NormNumGroups);
            }

            if (blockIdx < numBlocks - 1)
            {
                int ch = reversedCh[blockIdx];
                w[$"decoder.up_blocks.{blockIdx}.upsamplers.0.conv.weight"] = CreateRandomTensor(ch, ch, 3, 3);
                w[$"decoder.up_blocks.{blockIdx}.upsamplers.0.conv.bias"] = CreateRandomTensor(ch);
            }
        }

        // conv_norm_out + conv_out
        int finalCh = blockCh[0];
        w["decoder.conv_norm_out.weight"] = CreateRandomTensor(finalCh);
        w["decoder.conv_norm_out.bias"] = CreateRandomTensor(finalCh);
        w["decoder.conv_out.weight"] = CreateRandomTensor(3, finalCh, 3, 3);
        w["decoder.conv_out.bias"] = CreateRandomTensor(3);

        return w;
    }

    private static void AddResNetWeights(Dictionary<string, Tensor> w, string prefix, int inCh, int outCh, int normGroups)
    {
        w[$"{prefix}.norm1.weight"] = CreateRandomTensor(inCh);
        w[$"{prefix}.norm1.bias"] = CreateRandomTensor(inCh);
        w[$"{prefix}.conv1.weight"] = CreateRandomTensor(outCh, inCh, 3, 3);
        w[$"{prefix}.conv1.bias"] = CreateRandomTensor(outCh);
        w[$"{prefix}.norm2.weight"] = CreateRandomTensor(outCh);
        w[$"{prefix}.norm2.bias"] = CreateRandomTensor(outCh);
        w[$"{prefix}.conv2.weight"] = CreateRandomTensor(outCh, outCh, 3, 3);
        w[$"{prefix}.conv2.bias"] = CreateRandomTensor(outCh);

        if (inCh != outCh)
        {
            w[$"{prefix}.conv_shortcut.weight"] = CreateRandomTensor(outCh, inCh, 1, 1);
            w[$"{prefix}.conv_shortcut.bias"] = CreateRandomTensor(outCh);
        }
    }

    private static void AddVaeAttentionWeights(Dictionary<string, Tensor> w, string prefix, int channels, int normGroups)
    {
        w[$"{prefix}.group_norm.weight"] = CreateRandomTensor(channels);
        w[$"{prefix}.group_norm.bias"] = CreateRandomTensor(channels);
        w[$"{prefix}.to_q.weight"] = CreateRandomTensor(channels, channels);
        w[$"{prefix}.to_q.bias"] = CreateRandomTensor(channels);
        w[$"{prefix}.to_k.weight"] = CreateRandomTensor(channels, channels);
        w[$"{prefix}.to_k.bias"] = CreateRandomTensor(channels);
        w[$"{prefix}.to_v.weight"] = CreateRandomTensor(channels, channels);
        w[$"{prefix}.to_v.bias"] = CreateRandomTensor(channels);
        w[$"{prefix}.to_out.0.weight"] = CreateRandomTensor(channels, channels);
        w[$"{prefix}.to_out.0.bias"] = CreateRandomTensor(channels);
    }

    private static void DisposeWeights(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor t in weights.Values)
        {
            t.Dispose();
        }
    }

    public void Dispose()
    {
        _backend.Dispose();
    }
}
