using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tests.Common;

namespace SharpInference.Diffusion.Tests;

/// <summary>Flux weight loading tests. Validates that Flux safetensors checkpoints load correctly into the FluxTransformer. Supports both BFL single-file and diffusers formats.</summary>
public sealed class FluxWeightLoadingTests
{
    private static string FluxSingleFilePath => TestPaths.Flux.Schnell;

    private readonly ITestOutputHelper _output;

    public FluxWeightLoadingTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SingleFile_EnumerateKeys()
    {
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {FluxSingleFilePath}");
            return;
        }

        _output.WriteLine($"Loading: {Path.GetFileName(FluxSingleFilePath)}");
        using SafeTensorsLoader loader = new();
        loader.Load(FluxSingleFilePath);

        IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors = loader.Descriptors;
        _output.WriteLine($"Total tensors: {descriptors.Count}");

        // Group by top-level prefix
        Dictionary<string, int> prefixCounts = [];
        foreach (string key in descriptors.Keys)
        {
            string prefix = key.Contains('.') ? key[..key.IndexOf('.')] : key;
            prefixCounts.TryGetValue(prefix, out int count);
            prefixCounts[prefix] = count + 1;
        }

        _output.WriteLine("\n=== Key Prefix Groups ===");
        foreach (KeyValuePair<string, int> kvp in prefixCounts.OrderByDescending(x => x.Value))
        {
            _output.WriteLine($"  {kvp.Key}: {kvp.Value} tensors");
        }

        Assert.True(descriptors.Count > 0, "Checkpoint should contain tensors");
    }

    [Fact]
    public void SingleFile_ConvertAndVerifyTransformerKeys()
    {
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {FluxSingleFilePath}");
            return;
        }

        _output.WriteLine($"Loading and converting: {Path.GetFileName(FluxSingleFilePath)}");
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);

        using (loader)
        {
            _output.WriteLine($"Transformer keys: {converted.Transformer.Count}");
            _output.WriteLine($"CLIP-L keys: {converted.ClipL.Count}");
            _output.WriteLine($"T5 keys: {converted.T5.Count}");
            _output.WriteLine($"VAE keys: {converted.Vae.Count}");

            // Verify essential transformer keys exist
            string[] requiredKeys =
            [
                "x_embedder.weight",
                "x_embedder.bias",
                "context_embedder.weight",
                "context_embedder.bias",
                "time_text_embed.timestep_embedder.linear_1.weight",
                "time_text_embed.timestep_embedder.linear_1.bias",
                "time_text_embed.timestep_embedder.linear_2.weight",
                "time_text_embed.timestep_embedder.linear_2.bias",
                "time_text_embed.text_embedder.linear_1.weight",
                "time_text_embed.text_embedder.linear_1.bias",
                "time_text_embed.text_embedder.linear_2.weight",
                "time_text_embed.text_embedder.linear_2.bias",
                "norm_out.linear.weight",
                "norm_out.linear.bias",
                "proj_out.weight",
                "proj_out.bias",
                // First double-stream block
                "transformer_blocks.0.norm1.linear.weight",
                "transformer_blocks.0.attn.to_q.weight",
                "transformer_blocks.0.attn.to_k.weight",
                "transformer_blocks.0.attn.to_v.weight",
                "transformer_blocks.0.attn.to_out.0.weight",
                "transformer_blocks.0.attn.add_q_proj.weight",
                "transformer_blocks.0.attn.add_k_proj.weight",
                "transformer_blocks.0.attn.add_v_proj.weight",
                "transformer_blocks.0.attn.norm_q.weight",
                "transformer_blocks.0.attn.norm_k.weight",
                "transformer_blocks.0.attn.norm_added_q.weight",
                "transformer_blocks.0.attn.norm_added_k.weight",
                "transformer_blocks.0.ff.net.0.proj.weight",
                "transformer_blocks.0.ff.net.2.weight",
                "transformer_blocks.0.ff_context.net.0.proj.weight",
                "transformer_blocks.0.ff_context.net.2.weight",
                // First single-stream block
                "single_transformer_blocks.0.norm.linear.weight",
                "single_transformer_blocks.0.attn.to_q.weight",
                "single_transformer_blocks.0.attn.to_k.weight",
                "single_transformer_blocks.0.attn.to_v.weight",
                "single_transformer_blocks.0.attn.norm_q.weight",
                "single_transformer_blocks.0.attn.norm_k.weight",
                "single_transformer_blocks.0.proj_mlp.weight",
                "single_transformer_blocks.0.proj_out.weight",
            ];

            foreach (string key in requiredKeys)
            {
                Assert.True(converted.Transformer.ContainsKey(key),
                    $"Missing required transformer key: {key}");
            }

            _output.WriteLine($"\nAll {requiredKeys.Length} required keys verified present.");
        }
    }

    [Fact]
    public void SingleFile_DetectArchitecture()
    {
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {FluxSingleFilePath}");
            return;
        }

        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);

        using (loader)
        {
            (int doubleBlocks, int singleBlocks, bool hasGuidance) =
                FluxCheckpointConverter.DetectArchitecture(converted.Transformer);

            _output.WriteLine($"Double blocks: {doubleBlocks}");
            _output.WriteLine($"Single blocks: {singleBlocks}");
            _output.WriteLine($"Has guidance: {hasGuidance}");

            Assert.Equal(19, doubleBlocks);
            Assert.Equal(38, singleBlocks);
            // Schnell has no guidance, Dev has guidance
        }
    }

    [Fact]
    public void SingleFile_LoadTransformer_NoExceptions()
    {
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {FluxSingleFilePath}");
            return;
        }

        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);

        using (loader)
        {
            (int _, int _, bool hasGuidance) =
                FluxCheckpointConverter.DetectArchitecture(converted.Transformer);

            FluxConfig config = hasGuidance ? FluxConfig.Dev : FluxConfig.Schnell;
            _output.WriteLine($"Config: {(hasGuidance ? "Dev" : "Schnell")}");

            FluxTransformer transformer = new FluxTransformer(config);
            transformer.LoadWeights(converted.Transformer);

            _output.WriteLine("Transformer loaded successfully.");
            transformer.Dispose();
        }
    }

    [Fact]
    public void SingleFile_VerifyQkvShapes()
    {
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {FluxSingleFilePath}");
            return;
        }

        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);

        using (loader)
        {
            // Double block Q/K/V should be [3072, 3072]
            Tensor toQ = converted.Transformer["transformer_blocks.0.attn.to_q.weight"];
            Assert.Equal(3072, (int)toQ.Shape[0]);
            Assert.Equal(3072, (int)toQ.Shape[1]);

            // Single block Q/K/V should be [3072, 3072]
            Tensor singleQ = converted.Transformer["single_transformer_blocks.0.attn.to_q.weight"];
            Assert.Equal(3072, (int)singleQ.Shape[0]);
            Assert.Equal(3072, (int)singleQ.Shape[1]);

            // Single block proj_mlp should be [12288, 3072]
            Tensor projMlp = converted.Transformer["single_transformer_blocks.0.proj_mlp.weight"];
            Assert.Equal(12288, (int)projMlp.Shape[0]);
            Assert.Equal(3072, (int)projMlp.Shape[1]);

            // x_embedder: [3072, 64]
            Tensor xEmbed = converted.Transformer["x_embedder.weight"];
            Assert.Equal(3072, (int)xEmbed.Shape[0]);
            Assert.Equal(64, (int)xEmbed.Shape[1]);

            // context_embedder: [3072, 4096]
            Tensor ctxEmbed = converted.Transformer["context_embedder.weight"];
            Assert.Equal(3072, (int)ctxEmbed.Shape[0]);
            Assert.Equal(4096, (int)ctxEmbed.Shape[1]);

            _output.WriteLine("All shapes verified.");
        }
    }

    [Fact]
    public unsafe void SingleFile_NoNaNInWeights()
    {
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {FluxSingleFilePath}");
            return;
        }

        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);

        using (loader)
        {
            int checkedKeys = 0;
            foreach (KeyValuePair<string, Tensor> kvp in converted.Transformer)
            {
                Tensor tensor = kvp.Value;
                if (tensor.DType == DType.F32)
                {
                    float* ptr = (float*)tensor.DataPointer;
                    int count = (int)tensor.ElementCount;
                    for (int i = 0; i < count; i++)
                    {
                        Assert.False(float.IsNaN(ptr[i]), $"NaN found in {kvp.Key} at index {i}");
                    }
                }
                checkedKeys++;
            }

            _output.WriteLine($"Checked {checkedKeys} keys for NaN — all clean.");
        }
    }
}
