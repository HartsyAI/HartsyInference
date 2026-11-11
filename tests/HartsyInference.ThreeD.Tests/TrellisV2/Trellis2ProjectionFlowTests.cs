using System.Buffers.Binary;
using System.Text;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>Pixal3D's projection mode of the TRELLIS.2 flows: sidecar parsing, key layout, forward wiring and the loader's handling of the complex RoPE buffer.</summary>
public sealed class Trellis2ProjectionFlowTests
{
    private const double Tolerance = 2e-4;
    private const int ProjChannels = 6;

    private const string TinyProjectionSidecar = """
        {"name": "ElasticSLatFlowModel", "args": {"resolution": 8, "in_channels": 4, "out_channels": 4, "model_channels": 16,
        "cond_channels": 8, "num_blocks": 2, "num_heads": 2, "mlp_ratio": 2.0, "pe_mode": "rope", "share_mod": true,
        "qk_rms_norm": true, "qk_rms_norm_cross": true, "image_attn_mode": "proj", "proj_in_channels": 6}}
        """;

    [Fact]
    public void ShippedPixal3DSidecars_ParseToProjectionFlows()
    {
        Trellis2FlowConfig structure = Trellis2FlowConfig.FromJson(Trellis2TestData.Pixal3DStructureFlowJson);
        Trellis2FlowConfig shape = Trellis2FlowConfig.FromJson(Trellis2TestData.Pixal3DShapeFlow512Json);

        Assert.True(structure.Dense);
        Assert.True(structure.HasProjection);
        // proj_in_channels is absent from the structure sidecar, so it follows cond_channels.
        Assert.Equal(1024, structure.ProjInChannels);
        Assert.False(shape.Dense);
        Assert.True(shape.HasProjection);
        Assert.Equal(2048, shape.ProjInChannels);
        Assert.Equal("cross_attn.cross_attn_block", shape.CrossAttentionScope);
    }

    [Fact]
    public void Trellis2Sidecars_StayPlainCrossAttention()
    {
        Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(Trellis2TestData.ShippedShapeFlow512Json);

        Assert.False(config.HasProjection);
        Assert.Equal("cross_attn", config.CrossAttentionScope);
    }

    [Theory]
    [InlineData("\"image_attn_mode\": \"proj\"", "\"image_attn_mode\": \"gated_proj\"")]
    [InlineData("\"image_attn_mode\": \"proj\"", "\"image_attn_mode\": \"other\"")]
    public void UnimplementedImageAttentionModesAreRejected(string from, string to)
    {
        string json = Trellis2TestData.Pixal3DShapeFlow512Json.Replace(from, to, StringComparison.Ordinal);

        Assert.Throws<NotSupportedException>(() => Trellis2FlowConfig.FromJson(json));
    }

    [Fact]
    public void Pixal3DKeyMap_ListsSevenHundredTensorsWithTheNestedCrossAttentionNames()
    {
        // 10 model-level and 23 per block: the 21 plain ones, now under cross_attn_block, plus proj_linear weight and bias.
        Trellis2FlowConfig shape = Trellis2FlowConfig.FromJson(Trellis2TestData.Pixal3DShapeFlow512Json);
        Trellis2FlowConfig structure = Trellis2FlowConfig.FromJson(Trellis2TestData.Pixal3DStructureFlowJson);

        Dictionary<string, long[]> shapes = Trellis2FlowKeyMap.Specs(shape).ToDictionary(s => s.CheckpointKey, s => s.Shape);

        Assert.Equal(10 + 23 * 30, shapes.Count);
        Assert.Equal(700, shapes.Count);
        Assert.Equal(new long[] { 1536, 2048 }, shapes["blocks.7.cross_attn.proj_linear.weight"]);
        Assert.Equal(new long[] { 1536 }, shapes["blocks.7.cross_attn.proj_linear.bias"]);
        Assert.Equal(new long[] { 3072, 1024 }, shapes["blocks.7.cross_attn.cross_attn_block.to_kv.weight"]);
        Assert.Equal(new long[] { 12, 128 }, shapes["blocks.7.cross_attn.cross_attn_block.q_rms_norm.gamma"]);
        Assert.DoesNotContain("blocks.7.cross_attn.to_kv.weight", shapes.Keys);
        Assert.Equal(700, Trellis2FlowKeyMap.Specs(structure).Count);
        Assert.Equal(new long[] { 1536, 1024 }, Trellis2FlowKeyMap.Specs(structure).Single(s => s.CheckpointKey == "blocks.0.cross_attn.proj_linear.weight").Shape);
    }

    [Fact]
    public void DenseProjectionFlow_MatchesTheScalarReference()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: true, ProjChannels);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 71);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2SparseStructureFlow flow = new(weights);
        Random rng = new(72);
        int r = config.Resolution, tokens = config.DenseTokens;
        using Tensor latent = Trellis2TestData.Random(rng, 1, config.InChannels, r, r, r);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 5, config.CondChannels);
        using Tensor proj = Trellis2TestData.Random(rng, 1, tokens, ProjChannels);
        using CpuBackend backend = new();

        using Tensor velocity = flow.Forward(backend, latent, 640f, cond, proj);

        float[] latentValues = Trellis2TestData.ToArray(latent);
        double[][] input = new double[tokens][];
        double[][] positions = new double[tokens][];
        for (int t = 0; t < tokens; t++)
        {
            input[t] = Enumerable.Range(0, config.InChannels).Select(c => (double)latentValues[c * tokens + t]).ToArray();
            positions[t] = [t / (r * r), t / r % r, t % r];
        }

        double[][] expected = new Trellis2Reference(config, tensors).Forward(input, positions, 640.0, ToRows(cond), ToRows(proj));

        float[] actual = Trellis2TestData.ToArray(velocity);
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < config.OutChannels; c++)
                Assert.Equal(expected[t][c], actual[c * tokens + t], Tolerance);
        }

        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void SparseProjectionFlow_MatchesTheScalarReference()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false, ProjChannels);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 81);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2SlatFlow flow = new(weights);
        Random rng = new(82);
        int[] coords = [0, 1, 2, 3, 0, 7, 0, 4, 0, 3, 3, 3, 0, 5, 6, 1];
        int n = coords.Length / 4;
        using Tensor x = Trellis2TestData.Random(rng, 1, n, config.InChannels);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 5, config.CondChannels);
        using Tensor proj = Trellis2TestData.Random(rng, 1, n, ProjChannels);
        using CpuBackend backend = new();

        using Tensor velocity = flow.Forward(backend, x, coords, 310f, cond, proj);

        double[][] positions = Enumerable.Range(0, n).Select(i => new double[] { coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3] }).ToArray();
        double[][] expected = new Trellis2Reference(config, tensors).Forward(ToRows(x), positions, 310.0, ToRows(cond), ToRows(proj));
        float[] actual = Trellis2TestData.ToArray(velocity);
        for (int t = 0; t < n; t++)
        {
            for (int c = 0; c < config.OutChannels; c++)
                Assert.Equal(expected[t][c], actual[t * config.OutChannels + c], Tolerance);
        }

        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void ProjectionBranchChangesTheVelocity()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false, ProjChannels);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 91);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2SlatFlow flow = new(weights);
        Random rng = new(92);
        int[] coords = [0, 1, 2, 3, 0, 4, 4, 4];
        using Tensor x = Trellis2TestData.Random(rng, 1, 2, config.InChannels);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 3, config.CondChannels);
        using Tensor proj = Trellis2TestData.Random(rng, 1, 2, ProjChannels);
        using Tensor zeros = new(new TensorShape(1, 2, ProjChannels), DType.F32);
        using CpuBackend backend = new();

        using Tensor withFeatures = flow.Forward(backend, x, coords, 500f, cond, proj);
        using Tensor withZeros = flow.Forward(backend, x, coords, 500f, cond, zeros);

        // A zero feature row still adds proj_linear's bias, the negative branch of classifier-free guidance.
        Assert.NotEqual(Trellis2TestData.ToArray(withFeatures), Trellis2TestData.ToArray(withZeros));
        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void ZeroedProjectionLinearReproducesThePlainFlowExactly()
    {
        // With proj_linear zero the projection flow is the plain flow: pins that the proj branch adds nothing else to TRELLIS.2's math.
        Trellis2FlowConfig plain = Trellis2TestData.Tiny(dense: false);
        Trellis2FlowConfig projection = Trellis2TestData.Tiny(dense: false, ProjChannels);
        Dictionary<string, Tensor> plainTensors = Trellis2TestData.RandomWeights(plain, 101);
        Dictionary<string, Tensor> projectionTensors = [];
        foreach ((string key, Tensor tensor) in plainTensors)
            projectionTensors[key.Replace(".cross_attn.", ".cross_attn.cross_attn_block.", StringComparison.Ordinal)] = tensor;
        foreach (ShapeVaeKeyMap.TensorSpec spec in Trellis2FlowKeyMap.Specs(projection).Where(s => s.CheckpointKey.Contains("proj_linear", StringComparison.Ordinal)))
            projectionTensors[spec.CheckpointKey] = new Tensor(new TensorShape(spec.Shape), DType.F32);
        using Trellis2FlowWeights plainWeights = Trellis2FlowWeights.FromTensors(plain, plainTensors);
        using Trellis2FlowWeights projectionWeights = Trellis2FlowWeights.FromTensors(projection, projectionTensors);
        using Trellis2SlatFlow plainFlow = new(plainWeights);
        using Trellis2SlatFlow projectionFlow = new(projectionWeights);
        Random rng = new(102);
        int[] coords = [0, 1, 2, 3, 0, 4, 4, 4, 0, 6, 1, 0];
        using Tensor x = Trellis2TestData.Random(rng, 1, 3, plain.InChannels);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 4, plain.CondChannels);
        using Tensor proj = Trellis2TestData.Random(rng, 1, 3, ProjChannels);
        using CpuBackend backend = new();

        using Tensor expected = plainFlow.Forward(backend, x, coords, 420f, cond);
        using Tensor actual = projectionFlow.Forward(backend, x, coords, 420f, cond, proj);

        Assert.Equal(Trellis2TestData.ToArray(expected), Trellis2TestData.ToArray(actual));
        Trellis2TestData.DisposeAll(plainTensors.Values);
        foreach (KeyValuePair<string, Tensor> entry in projectionTensors.Where(e => e.Key.Contains("proj_linear", StringComparison.Ordinal)))
            entry.Value.Dispose();
    }

    [Fact]
    public void FeaturesAreRequiredByProjectionFlowsAndRefusedByPlainOnes()
    {
        Trellis2FlowConfig projection = Trellis2TestData.Tiny(dense: false, ProjChannels);
        Trellis2FlowConfig plain = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> projectionTensors = Trellis2TestData.RandomWeights(projection, 111);
        Dictionary<string, Tensor> plainTensors = Trellis2TestData.RandomWeights(plain, 112);
        using Trellis2FlowWeights projectionWeights = Trellis2FlowWeights.FromTensors(projection, projectionTensors);
        using Trellis2FlowWeights plainWeights = Trellis2FlowWeights.FromTensors(plain, plainTensors);
        using Trellis2SlatFlow projectionFlow = new(projectionWeights);
        using Trellis2SlatFlow plainFlow = new(plainWeights);
        Random rng = new(113);
        int[] coords = [0, 1, 2, 3, 0, 4, 4, 4];
        using Tensor x = Trellis2TestData.Random(rng, 1, 2, 4);
        using Tensor cond = Trellis2TestData.Random(rng, 1, 3, 8);
        using Tensor wrongRows = Trellis2TestData.Random(rng, 1, 3, ProjChannels);
        using Tensor wrongWidth = Trellis2TestData.Random(rng, 1, 2, ProjChannels + 1);
        using Tensor proj = Trellis2TestData.Random(rng, 1, 2, ProjChannels);
        using CpuBackend backend = new();

        Assert.Throws<ArgumentException>(() => projectionFlow.Forward(backend, x, coords, 1f, cond));
        Assert.Throws<ArgumentException>(() => projectionFlow.Forward(backend, x, coords, 1f, cond, wrongRows));
        Assert.Throws<ArgumentException>(() => projectionFlow.Forward(backend, x, coords, 1f, cond, wrongWidth));
        Assert.Throws<ArgumentException>(() => plainFlow.Forward(backend, x, coords, 1f, cond, proj));
        Trellis2TestData.DisposeAll(projectionTensors.Values);
        Trellis2TestData.DisposeAll(plainTensors.Values);
    }

    [Fact]
    public void Load_LeavesTheComplexRopeBufferUnread()
    {
        string directory = Directory.CreateTempSubdirectory("pixal3d-flow-").FullName;
        try
        {
            Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(TinyProjectionSidecar);
            string path = WriteCheckpoint(directory, config, "rope_phases", "C64");

            using Trellis2FlowWeights weights = Trellis2FlowWeights.Load(path);

            Assert.True(weights.Config.HasProjection);
            Assert.Equal(Trellis2FlowKeyMap.Specs(config).Count, weights.Tensors.Count);
            Assert.DoesNotContain("rope_phases", weights.Tensors.Keys);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Theory]
    [InlineData("rope_phases", "F32")]
    [InlineData("blocks.0.extra_phases", "C64")]
    public void Load_RefusesAnyOtherUnexpectedTensor(string name, string dtype)
    {
        string directory = Directory.CreateTempSubdirectory("pixal3d-flow-").FullName;
        try
        {
            Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(TinyProjectionSidecar);
            string path = WriteCheckpoint(directory, config, name, dtype);

            Assert.ThrowsAny<HartsyInferenceException>(() => Trellis2FlowWeights.Load(path));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    // Writes the tiny checkpoint's sidecar and a safetensors file holding its F32 tensors plus one extra 8-byte-element tensor.
    private static string WriteCheckpoint(string directory, Trellis2FlowConfig config, string extraName, string extraDType)
    {
        string path = Path.Combine(directory, "flow.safetensors");
        File.WriteAllText(Path.ChangeExtension(path, ".json"), TinyProjectionSidecar);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 121);
        StringBuilder header = new("{");
        using MemoryStream data = new();
        foreach ((string key, Tensor tensor) in tensors)
        {
            long start = data.Length;
            data.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tensor.AsReadOnlySpan<float>()));
            header.Append(CultureInvariantEntry(key, "F32", Enumerable.Range(0, tensor.Shape.Rank).Select(i => tensor.Shape[i]).ToArray(), start, data.Length)).Append(',');
        }

        long extraStart = data.Length;
        int extraBytes = extraDType == "C64" ? 8 * 4 : 4 * 4;
        data.Write(new byte[extraBytes]);
        header.Append(CultureInvariantEntry(extraName, extraDType, [4], extraStart, data.Length)).Append('}');
        byte[] json = Encoding.UTF8.GetBytes(header.ToString().PadRight(header.Length + (8 - header.Length % 8) % 8));
        byte[] prefix = new byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)json.Length);
        using (FileStream file = File.Create(path))
        {
            file.Write(prefix);
            file.Write(json);
            data.Position = 0;
            data.CopyTo(file);
        }

        Trellis2TestData.DisposeAll(tensors.Values);
        return path;
    }

    private static string CultureInvariantEntry(string name, string dtype, long[] shape, long start, long end)
        => $"\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{string.Join(",", shape)}],\"data_offsets\":[{start},{end}]}}";

    private static double[][] ToRows(Tensor t)
    {
        int width = (int)t.Shape[t.Shape.Rank - 1];
        float[] values = Trellis2TestData.ToArray(t);
        return Enumerable.Range(0, values.Length / width)
            .Select(r => Enumerable.Range(0, width).Select(c => (double)values[r * width + c]).ToArray()).ToArray();
    }
}
