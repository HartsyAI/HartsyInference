using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the container that makes quantized-checkpoint support architecture-independent: a GGUF file and the
/// safetensors build it was converted from must reach a converter as the same keys, the same shapes and the same folded
/// quantization companions, or every recipe needs its own format branch again.</summary>
public sealed unsafe class CheckpointSourceTests : IDisposable
{
    private readonly string _tempDir;

    public CheckpointSourceTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"checkpoint_source_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); }
        catch (IOException) { /* a mmap the OS has not released yet is not a test failure */ }
    }

    private static Tensor F32(TensorShape shape, float start = 0f)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = start + i * 0.25f;
        return tensor;
    }

    /// <summary>Writes the same weights as safetensors and as GGUF, so a test can assert the two arrive identically.</summary>
    private (string SafeTensorsPath, string GgufPath) WriteBothContainers(Dictionary<string, Tensor> weights,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        string safetensorsPath = Path.Combine(_tempDir, "model.safetensors");
        SafeTensorsWriter.Save(safetensorsPath, weights, metadata);

        string ggufPath = Path.Combine(_tempDir, "model.gguf");
        using (GgufWriter writer = new GgufWriter(ggufPath))
        {
            // An architecture with no registered mapper falls through to the identity passthrough, which is what every
            // diffusion GGUF mapper is anyway.
            writer.SetMetadata("general.architecture", "checkpointsourcetest");
            foreach (KeyValuePair<string, Tensor> entry in weights) writer.AddTensor(entry.Key, entry.Value);
            writer.Flush();
        }
        return (safetensorsPath, ggufPath);
    }

    [Fact]
    public void Sniff_ReadsTheMagicBytesRatherThanTheExtension()
    {
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn.to_q.weight"] = F32(new TensorShape(4, 8)) };
        try
        {
            (string safetensorsPath, string ggufPath) = WriteBothContainers(weights);
            Assert.Equal(ModelFormat.SafeTensors, CheckpointSource.Sniff(safetensorsPath));
            Assert.Equal(ModelFormat.Gguf, CheckpointSource.Sniff(ggufPath));

            // Repacks are routinely published under the wrong extension; trusting it fails deep inside a parser.
            string misnamed = Path.Combine(_tempDir, "actually-gguf.safetensors");
            File.Copy(ggufPath, misnamed);
            Assert.Equal(ModelFormat.Gguf, CheckpointSource.Sniff(misnamed));
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Sniff_RefusesSomethingThatIsNeitherContainer()
    {
        string path = Path.Combine(_tempDir, "not-a-checkpoint.safetensors");
        File.WriteAllBytes(path, Encoding.ASCII.GetBytes("version https://git-lfs.github.com/spec/v1\n"));
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(() => CheckpointSource.Sniff(path));
        Assert.Contains("neither safetensors nor GGUF", error.Message);
    }

    [Fact]
    public void Sniff_RefusesAFileTooSmallToHoldAHeader()
    {
        string path = Path.Combine(_tempDir, "truncated.safetensors");
        File.WriteAllBytes(path, [1, 2, 3, 4]);
        Assert.Throws<UnsupportedModelException>(() => CheckpointSource.Sniff(path));
    }

    [Fact]
    public void Open_PresentsAGgufAndItsSafetensorsTwinIdentically()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.attn.to_q.weight"] = F32(new TensorShape(4, 8)),
            ["blocks.0.attn.to_q.bias"] = F32(new TensorShape(4), 100f),
            ["blocks.0.norm.weight"] = F32(new TensorShape(8), 200f),
        };
        try
        {
            (string safetensorsPath, string ggufPath) = WriteBothContainers(weights);

            using CheckpointSource fromSafeTensors = CheckpointSource.Open(safetensorsPath);
            using CheckpointSource fromGguf = CheckpointSource.Open(ggufPath);

            Assert.Equal(ModelFormat.SafeTensors, fromSafeTensors.Format);
            Assert.Equal(ModelFormat.Gguf, fromGguf.Format);
            Assert.Equal(weights.Keys.Order(), fromGguf.Weights.Keys.Order());

            foreach (string key in weights.Keys)
            {
                Tensor expected = fromSafeTensors.Weights[key];
                Tensor actual = fromGguf.Weights[key];
                Assert.Equal(expected.DType, actual.DType);
                // The relabel is what puts a GGUF matrix back in [out, in]; without it every Linear is transposed.
                Assert.Equal(expected.Shape, actual.Shape);
                Assert.True(expected.AsReadOnlySpan<float>().SequenceEqual(actual.AsReadOnlySpan<float>()),
                    $"'{key}' differs between the two containers.");
            }
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Header_ReportsTheSameInventoryForBothContainers()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.attn.to_q.weight"] = F32(new TensorShape(4, 8)),
            ["blocks.0.norm.weight"] = F32(new TensorShape(8)),
        };
        try
        {
            (string safetensorsPath, string ggufPath) = WriteBothContainers(weights,
                new Dictionary<string, string> { ["format"] = "pt" });

            CheckpointHeader safetensorsHeader = CheckpointHeader.Read(safetensorsPath);
            CheckpointHeader ggufHeader = CheckpointHeader.Read(ggufPath);

            Assert.Equal(ModelFormat.SafeTensors, safetensorsHeader.Format);
            Assert.Equal(ModelFormat.Gguf, ggufHeader.Format);
            Assert.Equal(safetensorsHeader.Descriptors.Keys.Order(), ggufHeader.Descriptors.Keys.Order());
            foreach (string key in weights.Keys)
            {
                SafeTensorDescriptor expected = safetensorsHeader.Descriptors[key];
                SafeTensorDescriptor actual = ggufHeader.Descriptors[key];
                Assert.Equal(expected.DType, actual.DType);
                // A planner validating a checkpoint's structure by shape must see the same shape from either container.
                Assert.Equal(expected.Shape, actual.Shape);
                Assert.Equal(expected.ByteLength, actual.ByteLength);
            }
            Assert.Equal("pt", safetensorsHeader.Metadata["format"]);
            Assert.Equal("checkpointsourcetest", ggufHeader.Metadata["general.architecture"]);
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Open_ReversesEveryGgufAxis_NotOnlyMatrices()
    {
        // ggml ne order reverses every axis, so a convolution kernel the engine calls [out, in, kh, kw] is stored
        // [kw, kh, in, out]. Un-reversing only rank 2 leaves every conv in an SD1.5 UNet declaring its kernel width
        // as its output channel count.
        Dictionary<string, Tensor> weights = new()
        {
            ["conv_in.weight"] = F32(new TensorShape(8, 4, 3, 3)),
            ["blocks.0.attn.to_q.weight"] = F32(new TensorShape(4, 8)),
            ["blocks.0.norm.weight"] = F32(new TensorShape(8)),
        };
        try
        {
            (string safetensorsPath, string ggufPath) = WriteBothContainers(weights);
            using CheckpointSource fromSafeTensors = CheckpointSource.Open(safetensorsPath);
            using CheckpointSource fromGguf = CheckpointSource.Open(ggufPath);

            foreach (string key in weights.Keys)
            {
                Assert.Equal(fromSafeTensors.Weights[key].Shape, fromGguf.Weights[key].Shape);
                Assert.Equal(fromSafeTensors.Header.Descriptors[key].Shape, fromGguf.Header.Descriptors[key].Shape);
            }
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void AConverterGivenAnUnfoldedDictionaryRefusesByName()
    {
        // The fold has to precede the rename, so a caller that skips the container and hands over a raw loader
        // dictionary must fail loudly — the alternative is a model whose weights are quietly real/scale, which has no
        // symptom at load and renders as noise at the end of a generation.
        Dictionary<string, Tensor> raw = new()
        {
            ["transformer.transformer_blocks.0.attn.to_q.weight"] = new Tensor(new TensorShape(4, 8), DType.F8E4M3),
            ["transformer.transformer_blocks.0.attn.to_q.weight_scale"] = F32(new TensorShape(1)),
        };
        try
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(
                () => CheckpointConverters.QwenImageCheckpointConverter.Convert(raw));
            Assert.Contains("weight_scale", error.Message);
            Assert.Contains("CheckpointSource.Open", error.Message);
        }
        finally
        {
            foreach (Tensor tensor in raw.Values) tensor.Dispose();
        }
    }

    /// <summary>A weight and its companion split across shards still pair up.</summary>
    /// <remarks>Safetensors sharding makes no promise that related tensors share a file. Folding each shard on its own
    /// therefore splits pairs that belong together: an I8 weight whose scale is in the next shard refuses outright,
    /// and the fp8 case is worse because it is silent — the weight keeps a factor of 1.0 while the shard holding its
    /// scale drops it as an unclaimed companion, leaving a weight running at 1/scale.</remarks>
    [Fact]
    public void OpenShards_PairsAWeightWithACompanionInAnotherShard()
    {
        Dictionary<string, Tensor> first = new()
        {
            ["blocks.0.attn.to_q.weight"] = new Tensor(new TensorShape(4, 8), DType.F8E4M3),
        };
        Dictionary<string, Tensor> second = new()
        {
            ["blocks.0.attn.to_q.scale_weight"] = F32(new TensorShape(1)),
            ["blocks.0.norm.weight"] = F32(new TensorShape(8)),
        };
        second["blocks.0.attn.to_q.scale_weight"].AsSpan<float>()[0] = 0.0195f;
        try
        {
            string a = Path.Combine(_tempDir, "model-00001-of-00002.safetensors");
            string b = Path.Combine(_tempDir, "model-00002-of-00002.safetensors");
            SafeTensorsWriter.Save(a, first);
            SafeTensorsWriter.Save(b, second);

            using CheckpointSource source = CheckpointSource.OpenShards([a, b]);

            Assert.Equal(0.0195f, source.Weights["blocks.0.attn.to_q.weight"].Fp8ScaleFactor);
            Assert.DoesNotContain("blocks.0.attn.to_q.scale_weight", source.Weights.Keys);
            // Both shards' inventories are visible through the one header.
            Assert.Contains("blocks.0.attn.to_q.weight", source.Header.Descriptors.Keys);
            Assert.Contains("blocks.0.norm.weight", source.Header.Descriptors.Keys);
        }
        finally
        {
            foreach (Tensor tensor in first.Values) tensor.Dispose();
            foreach (Tensor tensor in second.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Header_NamesTheDominantQuantByBytesNotByTensorCount()
    {
        // A real GGUF is mostly quantized matrices and a long tail of F32 norms, so a count would report "not
        // quantized" for a file that is overwhelmingly Q8_0.
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.attn.to_q.weight"] = new Tensor(new TensorShape(64, 32), DType.Q8_0),
            ["blocks.0.norm.weight"] = F32(new TensorShape(8)),
            ["blocks.1.norm.weight"] = F32(new TensorShape(8)),
            ["blocks.2.norm.weight"] = F32(new TensorShape(8)),
        };
        try
        {
            string path = Path.Combine(_tempDir, "quantized.gguf");
            using (GgufWriter writer = new GgufWriter(path))
            {
                writer.SetMetadata("general.architecture", "checkpointsourcetest");
                foreach (KeyValuePair<string, Tensor> entry in weights) writer.AddTensor(entry.Key, entry.Value);
                writer.Flush();
            }

            Assert.Equal(DType.Q8_0.Name, CheckpointHeader.Read(path).DominantQuantName());
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Header_ReportsNoDominantQuantForADenseCheckpoint()
    {
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn.to_q.weight"] = F32(new TensorShape(4, 8)) };
        try
        {
            (string safetensorsPath, _) = WriteBothContainers(weights);
            Assert.Null(CheckpointHeader.Read(safetensorsPath).DominantQuantName());
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }
}
