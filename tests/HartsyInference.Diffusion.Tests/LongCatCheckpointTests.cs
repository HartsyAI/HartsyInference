using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The sharded diffusers-named index the base and avatar repositories ship, written tiny and read back through the loader.</summary>
public sealed class LongCatCheckpointTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "longcat-ckpt-" + Guid.NewGuid().ToString("N"));
    private readonly LongCatConfig _config = LongCatTestData.Tiny;

    public LongCatCheckpointTests() => Directory.CreateDirectory(_directory);

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private Dictionary<string, Tensor> WriteCheckpoint(bool writeConfig, Func<string, bool>? skip = null, DType? storage = null)
    {
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(_config, 9);
        string[] keys = weights.Keys.Order(StringComparer.Ordinal).Where(key => skip?.Invoke(key) != true).ToArray();
        string[] shardNames = ["diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"];
        Dictionary<string, string> map = [];
        long total = 0;
        for (int shard = 0; shard < 2; shard++)
        {
            Dictionary<string, Tensor> part = [];
            foreach (string key in keys.Where((_, i) => i % 2 == shard))
            {
                part[key] = storage is null ? weights[key] : weights[key].CastTo(storage.Value);
                map[key] = shardNames[shard];
                total += part[key].DType.ComputeByteCount(part[key].ElementCount);
            }
            SafeTensorsWriter.Save(Path.Combine(_directory, shardNames[shard]), part);
        }
        File.WriteAllText(Path.Combine(_directory, LongCatCheckpoint.IndexFileName),
            JsonSerializer.Serialize(new Dictionary<string, object> { ["metadata"] = new Dictionary<string, long> { ["total_size"] = total }, ["weight_map"] = map }));
        if (writeConfig)
        {
            File.WriteAllText(Path.Combine(_directory, "config.json"), $$"""
                {"in_channels":4,"out_channels":4,"hidden_size":128,"depth":2,"num_heads":4,"caption_channels":48,"mlp_ratio":4,
                 "adaln_tembed_dim":64,"frequency_embedding_size":64,"patch_size":[1,2,2],"text_tokens_zero_pad":true}
                """);
        }
        return weights;
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OpenedCheckpointRunsLikeTheSourceWeights(bool withConfigFile)
    {
        Dictionary<string, Tensor> source = WriteCheckpoint(withConfigFile);
        try
        {
            using CpuBackend backend = new();
            using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(_directory);
            Assert.Equal(_config, checkpoint.Config);
            Assert.Equal(source.Count, checkpoint.Weights.Count);

            using Tensor latent = LongCatTestData.Latent(_config, 5, 4, 4, 2);
            using Tensor text = LongCatTestData.Encoder(_config, 5, 3);
            LongCatConditioning conditioning = new(NumCondLatents: 1);
            float[] fromDisk = Forward(backend, checkpoint.Weights, latent, text, conditioning);
            float[] fromMemory = Forward(backend, source, latent, text, conditioning);

            Assert.Equal(fromMemory, fromDisk);
        }
        finally
        {
            LongCatTestData.DisposeAll(source);
        }
    }

    [Fact]
    public void Bf16ShardsLoadAsBf16AndWidenToTheSameForward()
    {
        Dictionary<string, Tensor> source = WriteCheckpoint(writeConfig: true, storage: DType.BF16);
        Dictionary<string, Tensor> widened = [];
        try
        {
            using CpuBackend backend = new();
            using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(_directory);
            foreach ((string key, Tensor tensor) in checkpoint.Weights)
            {
                Assert.Equal(DType.BF16, tensor.DType);
                widened[key] = tensor.CastTo(DType.F32);
            }

            using Tensor latent = LongCatTestData.Latent(_config, 5, 4, 4, 2);
            using Tensor text = LongCatTestData.Encoder(_config, 5, 3);
            LongCatConditioning conditioning = new(NumCondLatents: 1);
            float[] fromDisk = Forward(backend, widened, latent, text, conditioning);
            float[] fromSource = Forward(backend, source, latent, text, conditioning);

            Assert.True(LongCatTestData.RelRms(fromDisk, fromSource) < 5e-2, "bf16 storage should only round the weights");
        }
        finally
        {
            LongCatTestData.DisposeAll(source);
            LongCatTestData.DisposeAll(widened);
        }
    }

    [Fact]
    public void MissingTensorIsNamedBeforeAnythingIsMapped()
    {
        Dictionary<string, Tensor> source = WriteCheckpoint(writeConfig: true, skip: key => key == "blocks.1.cross_attn.kv_linear.bias");
        try
        {
            InvalidDataException error = Assert.Throws<InvalidDataException>(() => LongCatCheckpoint.Open(_directory));

            Assert.Contains("blocks.1.cross_attn.kv_linear.bias", error.Message);
        }
        finally
        {
            LongCatTestData.DisposeAll(source);
        }
    }

    [Fact]
    public void Int8LayoutIsRefusedWithAPointerToItsLoader()
    {
        Dictionary<string, Tensor> source = WriteCheckpoint(writeConfig: true);
        try
        {
            string index = Path.Combine(_directory, LongCatCheckpoint.IndexFileName);
            File.WriteAllText(index, File.ReadAllText(index).Replace("blocks.0.ffn.w1.weight", "blocks.0.ffn.w1.weight_int8"));

            Assert.ThrowsAny<Exception>(() => LongCatCheckpoint.Open(_directory));
        }
        finally
        {
            LongCatTestData.DisposeAll(source);
        }
    }

    private float[] Forward(CpuBackend backend, IReadOnlyDictionary<string, Tensor> weights, Tensor latent, Tensor text, LongCatConditioning conditioning)
    {
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(weights);
        using Tensor velocity = transformer.Forward(backend, latent, text, conditioning.FrameTimesteps(300f, 5), in conditioning);
        return velocity.AsReadOnlySpan<float>().ToArray();
    }
}
