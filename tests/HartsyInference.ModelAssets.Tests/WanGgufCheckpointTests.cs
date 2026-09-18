using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Wan through <see cref="CheckpointSource"/>: a GGUF repack and the safetensors build it was made from must
/// reach <see cref="WanVideoCheckpointConverter"/> as the same keys and the same shapes.</summary>
/// <remarks>Wan is the first family whose checkpoint is not all matrices, which is what these pin. Its
/// <c>blocks.N.modulation</c> is rank 3 and its <c>patch_embedding.weight</c> is a rank-5 Conv3d kernel, and ggml
/// stores both with every axis reversed. Under the rank-2-only relabel the rest of the engine used before the
/// container, they would have arrived as <c>[dim, 6, 1]</c> and <c>[2, 2, 1, in, dim]</c> — shapes that make
/// <c>WanConfigDetector</c> read the patch embed's kernel width as its output channel count.</remarks>
public sealed class WanGgufCheckpointTests(ITestOutputHelper output) : IDisposable
{
    private readonly ITestOutputHelper _output = output;
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), $"wan_gguf_{Guid.NewGuid():N}");

    /// <summary>A Wan T2V key set at toy width, keeping every rank the real checkpoint has.</summary>
    private static Dictionary<string, Tensor> BuildWanWeights()
    {
        const int dim = 8;
        const int inChannels = 16;
        return new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            // Conv3d [out, in, kt, kh, kw] — the only rank-5 tensor in the family.
            ["patch_embedding.weight"] = Filled(new TensorShape([dim, inChannels, 1, 2, 2])),
            ["patch_embedding.bias"] = Filled(new TensorShape(dim)),
            // Rank-3 AdaLN table: [1, 6, dim].
            ["blocks.0.modulation"] = Filled(new TensorShape(1, 6, dim)),
            ["head.modulation"] = Filled(new TensorShape(1, 2, dim)),
            ["blocks.0.self_attn.q.weight"] = Filled(new TensorShape(dim, dim)),
            ["blocks.0.cross_attn.k.weight"] = Filled(new TensorShape(dim, dim)),
            ["blocks.0.norm3.weight"] = Filled(new TensorShape(dim)),
            ["time_embedding.0.weight"] = Filled(new TensorShape(dim, dim)),
            ["head.head.weight"] = Filled(new TensorShape(dim, dim)),
        };
    }

    private static Tensor Filled(TensorShape shape)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = i * 0.5f;
        return tensor;
    }

    [Fact]
    public void AWanGgufAndItsSafetensorsTwinConvertIdentically()
    {
        Directory.CreateDirectory(_tempDir);
        Dictionary<string, Tensor> weights = BuildWanWeights();
        try
        {
            string safetensorsPath = Path.Combine(_tempDir, "wan.safetensors");
            SafeTensorsWriter.Save(safetensorsPath, weights);
            string ggufPath = Path.Combine(_tempDir, "wan.gguf");
            using (GgufWriter writer = new GgufWriter(ggufPath))
            {
                writer.SetMetadata("general.architecture", "wan");
                foreach (KeyValuePair<string, Tensor> entry in weights) writer.AddTensor(entry.Key, entry.Value);
                writer.Flush();
            }

            using CheckpointSource fromSafeTensors = CheckpointSource.Open(safetensorsPath);
            using CheckpointSource fromGguf = CheckpointSource.Open(ggufPath);
            Assert.Equal(ModelFormat.Gguf, fromGguf.Format);
            Assert.Equal("wan", fromGguf.Architecture);

            WanVideoCheckpointConverter.ConvertedWeights safetensorsConv =
                WanVideoCheckpointConverter.Convert(fromSafeTensors.Weights, fromSafeTensors.Header.Metadata);
            WanVideoCheckpointConverter.ConvertedWeights ggufConv =
                WanVideoCheckpointConverter.Convert(fromGguf.Weights, fromGguf.Header.Metadata);

            Assert.Equal(safetensorsConv.Transformer.Keys.Order(), ggufConv.Transformer.Keys.Order());
            foreach (KeyValuePair<string, Tensor> entry in safetensorsConv.Transformer)
            {
                Tensor actual = ggufConv.Transformer[entry.Key];
                Assert.Equal(entry.Value.Shape, actual.Shape);
                Assert.True(entry.Value.AsReadOnlySpan<float>().SequenceEqual(actual.AsReadOnlySpan<float>()),
                    $"'{entry.Key}' differs between the two containers.");
            }

            // Stated on its own because this is the value the recipe branches T2V vs concat-I2V on.
            Assert.Equal(5, ggufConv.Transformer["patch_embedding.weight"].Shape.Rank);
            Assert.Equal(16, ggufConv.Transformer["patch_embedding.weight"].Shape[1]);
            Assert.Equal(3, ggufConv.Transformer["blocks.0.scale_shift_table"].Shape.Rank);
            Assert.Equal(6, ggufConv.Transformer["blocks.0.scale_shift_table"].Shape[1]);
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    /// <summary>A GGUF repack cannot carry the Animate-2 declaration: the quantizer writes its own KV block and the
    /// safetensors <c>__metadata__</c> it read is not copied. The detection has to stay honest about that rather than
    /// guessing from a filename — an Animate-2 GGUF is key-for-key a Wan2.1 I2V-14B one.</summary>
    [Fact]
    public void AGgufRepackCarriesNoAnimate2Declaration()
    {
        Directory.CreateDirectory(_tempDir);
        Dictionary<string, Tensor> weights = BuildWanWeights();
        try
        {
            Dictionary<string, string> animate2 = new(StringComparer.Ordinal)
            {
                ["config"] = "{\"transformer\": {\"model_type\": \"animate2\"}}",
            };
            string safetensorsPath = Path.Combine(_tempDir, "animate2.safetensors");
            SafeTensorsWriter.Save(safetensorsPath, weights, animate2);
            string ggufPath = Path.Combine(_tempDir, "animate2.gguf");
            using (GgufWriter writer = new GgufWriter(ggufPath))
            {
                writer.SetMetadata("general.architecture", "wan");
                foreach (KeyValuePair<string, Tensor> entry in weights) writer.AddTensor(entry.Key, entry.Value);
                writer.Flush();
            }

            using CheckpointSource fromSafeTensors = CheckpointSource.Open(safetensorsPath);
            using CheckpointSource fromGguf = CheckpointSource.Open(ggufPath);
            // The safetensors side proves the declaration survives the container; the GGUF side is the limitation.
            Assert.True(WanVideoCheckpointConverter.IsAnimate2Metadata(fromSafeTensors.Header.Metadata));
            Assert.False(WanVideoCheckpointConverter.IsAnimate2Metadata(fromGguf.Header.Metadata));
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    /// <summary>The published city96/QuantStack repack, read for real. The synthetic twin above proves the axis
    /// reversal; only a shipped file proves that the family row recognizes what those tools actually write.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    [Trait("Category", "RealWeights")]
    public void ARealWanGguf_DeclaresWan_AndConvertsToTheDiffusersNames()
    {
        string path = TestPaths.WanVideo.T2V1_3BGguf;
        if (!RealWeightGate.Require(_output.WriteLine, path)) return;

        using CheckpointSource source = CheckpointSource.Open(path);
        Assert.Equal(ModelFormat.Gguf, source.Format);
        Assert.Equal("wan", source.Architecture);
        _output.WriteLine($"tensors: {source.Weights.Count}  dominant quant: {source.Header.DominantQuantName()}");

        WanVideoCheckpointConverter.ConvertedWeights converted =
            WanVideoCheckpointConverter.Convert(source.Weights, source.Header.Metadata);
        Assert.Equal(source.Weights.Count, converted.Transformer.Count);
        Assert.False(converted.IsAnimate2);

        Tensor patchEmbed = converted.Transformer["patch_embedding.weight"];
        Assert.Equal(5, patchEmbed.Shape.Rank);
        Assert.Equal(16, patchEmbed.Shape[1]);
        Tensor modulation = converted.Transformer["blocks.0.scale_shift_table"];
        Assert.Equal(3, modulation.Shape.Rank);
        Assert.Equal(6, modulation.Shape[1]);
        // The rename pass ran: original names would still be here otherwise.
        Assert.Contains("blocks.0.attn1.to_q.weight", converted.Transformer.Keys);
        Assert.Contains("blocks.0.attn2.to_k.weight", converted.Transformer.Keys);
        Assert.DoesNotContain("blocks.0.self_attn.q.weight", converted.Transformer.Keys);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true); }
        catch (IOException) { /* a mmap the OS has not released yet is not a test failure */ }
    }
}
