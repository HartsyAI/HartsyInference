using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Runtime LoRA on the LongCat DiT: the mapper for the reference's own state-dict layout (split fused projections,
/// stored alpha scale), and the gated
/// terms against a merged-weight forward computed from the reference's chunked LoRA arithmetic.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatLoraTests : IDisposable
{
    private const int Frames = 5, Height = 4, Width = 4, TextRows = 5;
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "longcat-lora-" + Guid.NewGuid().ToString("N"));
    private readonly CpuBackend _backend = new();
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _dense;

    public LongCatLoraTests()
    {
        Directory.CreateDirectory(_directory);
        _dense = LongCatTestData.Weights(_config, 41);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_dense);
        _backend.Dispose();
        Directory.Delete(_directory, recursive: true);
    }

    private string PathOf(string name) => Path.Combine(_directory, name);

    [Fact]
    public void FileIsRecognizedAndEveryModuleBecomesOneLayerOnItsDitWeight()
    {
        using LongCatLoraFixture fixture = new(PathOf("a.safetensors"), _config, 1);

        using LoraFile file = LoraFile.Load(fixture.Path);

        Assert.Equal(LoraFormat.LongCat, file.Format);
        Assert.Equal(7 * _config.Depth, file.Layers.Count);
        Assert.Equal(fixture.Modules.Select(static m => m + ".weight").Order(), file.Layers.Select(static l => l.TargetKey).Order());
        Assert.All(file.Layers, static l => Assert.Equal(LoraTarget.Transformer, l.Target));
        Assert.Empty(file.FullWeightDiffs);
    }

    [Fact]
    public void SplitProjectionsBecomeOneBlockDiagonalPairOfTheirFullRank()
    {
        using LongCatLoraFixture fixture = new(PathOf("a.safetensors"), _config, 1);
        using LoraFile file = LoraFile.Load(fixture.Path);
        int h = _config.HiddenSize;

        LoraLayer qkv = file.Layers.Single(static l => l.TargetKey == "blocks.0.attn.qkv.weight");
        LoraLayer kv = file.Layers.Single(static l => l.TargetKey == "blocks.1.cross_attn.kv_linear.weight");
        LoraLayer proj = file.Layers.Single(static l => l.TargetKey == "blocks.0.attn.proj.weight");

        Assert.Equal(new TensorShape(3 * h, 3 * LongCatLoraFixture.Rank), qkv.LoraUp.Shape);
        Assert.Equal(new TensorShape(3 * LongCatLoraFixture.Rank, h), qkv.LoraDown.Shape);
        Assert.Equal(new TensorShape(2 * h, 2 * LongCatLoraFixture.Rank), kv.LoraUp.Shape);
        Assert.Equal(new TensorShape(h, LongCatLoraFixture.Rank), proj.LoraUp.Shape);
        ReadOnlySpan<float> up = qkv.LoraUp.AsReadOnlySpan<float>();
        int cols = 3 * LongCatLoraFixture.Rank;
        for (int r = 0; r < 3 * h; r++)
            for (int c = 0; c < cols; c++)
                if (c / LongCatLoraFixture.Rank != r / h)
                    Assert.Equal(0f, up[r * cols + c]);
        Assert.Equal(fixture.AlphaScale, qkv.Delta.Scale, 6);
        Assert.Equal(LongCatLoraFixture.Rank * 3, qkv.Rank);
    }

    [Fact]
    public void EveryLayersDeltaIsTheReferencesChunkedProduct()
    {
        using LongCatLoraFixture fixture = new(PathOf("a.safetensors"), _config, 2);
        using LoraFile file = LoraFile.Load(fixture.Path);

        foreach (LoraLayer layer in file.Layers)
        {
            using Tensor full = layer.Delta.ComputeF32(_backend);
            float[] expected = fixture.Delta(layer.TargetKey[..^".weight".Length], 1f);
            ReadOnlySpan<float> got = full.AsReadOnlySpan<float>();
            float scale = layer.Delta.Scale;
            Assert.Equal(expected.Length, got.Length);
            double diff = 0, reference = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                diff += Math.Pow(scale * got[i] - expected[i], 2);
                reference += Math.Pow(expected[i], 2);
            }
            Assert.True(Math.Sqrt(diff / reference) < 1e-5, layer.TargetKey);
        }
    }

    [Fact]
    public void ANegativeStoredScaleIsFoldedIntoTheUpMatrix()
    {
        using LongCatLoraFixture fixture = new(PathOf("a.safetensors"), _config, 3, firstBlockAlphaScale: -0.3f);
        using LoraFile file = LoraFile.Load(fixture.Path);

        LoraLayer negative = file.Layers.Single(static l => l.TargetKey == "blocks.0.attn.proj.weight");
        using Tensor full = negative.Delta.ComputeF32(_backend);

        Assert.Equal(0.3f, negative.Delta.Scale, 6);
        float[] expected = fixture.Delta("blocks.0.attn.proj", 1f);
        ReadOnlySpan<float> got = full.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.Length; i += 97)
            Assert.Equal(expected[i], 0.3f * got[i], 5);
    }

    [Fact]
    public void MalformedModulesAreRefusedByName()
    {
        const string Module = "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___proj";
        Tensor Mat(long rows, long cols) => LongCatTestData.Random(new TensorShape(rows, cols), 5);
        void Write(string name, Dictionary<string, Tensor> tensors) => SafeTensorsWriter.Save(PathOf(name), tensors);

        Write("nodown.safetensors", new() { [Module + ".lora_up.weight"] = Mat(8, 2) });
        Write("both.safetensors", new()
        {
            [Module + ".lora_down.weight"] = Mat(2, 8), [Module + ".lora_up.weight"] = Mat(8, 2), [Module + ".lora_up.blocks.0.weight"] = Mat(8, 2),
        });
        Write("hole.safetensors", new()
        {
            [Module + ".lora_down.weight"] = Mat(4, 8), [Module + ".lora_up.blocks.0.weight"] = Mat(8, 2),
            [Module + ".lora_up.blocks.2.weight"] = Mat(8, 2),
        });
        Write("rank.safetensors", new() { [Module + ".lora_down.weight"] = Mat(2, 8), [Module + ".lora_up.weight"] = Mat(8, 3) });
        Write("foreign.safetensors", new()
        {
            [Module + ".lora_down.weight"] = Mat(2, 8), [Module + ".lora_up.weight"] = Mat(8, 2), ["blocks.0.attn.proj.weight"] = Mat(8, 8),
        });

        Assert.Contains("no lora_down", Assert.ThrowsAny<Exception>(() => LoraFile.Load(PathOf("nodown.safetensors"))).Message);
        Assert.Contains("not both", Assert.ThrowsAny<Exception>(() => LoraFile.Load(PathOf("both.safetensors"))).Message);
        Assert.Contains("no block 1", Assert.ThrowsAny<Exception>(() => LoraFile.Load(PathOf("hole.safetensors"))).Message);
        Assert.Contains("does not pair", Assert.ThrowsAny<Exception>(() => LoraFile.Load(PathOf("rank.safetensors"))).Message);
        Assert.ThrowsAny<Exception>(() => LoraFile.Load(PathOf("foreign.safetensors")));
    }

    [Fact]
    public void TargetKeyUndoesTheReferencesModuleSpelling()
    {
        Assert.Equal("blocks.12.cross_attn.kv_linear.weight",
            LongCatLoraMapper.TargetKeyOf("lora___lorahyphen___blocks___lorahyphen___12___lorahyphen___cross_attn___lorahyphen___kv_linear"));
    }

    [Fact]
    public void AGateOffForwardIsTheBaseAndAGateOnForwardIsTheMergedWeight()
    {
        using LongCatLoraFixture fixture = new(PathOf("a.safetensors"), _config, 7);
        using LoraFile file = LoraFile.Load(fixture.Path);
        using LoraStack stack = new();
        AdjunctGate gate = new();
        stack.AddGated(file, gate);
        Dictionary<string, Tensor> patched = new(_dense, StringComparer.Ordinal);
        stack.ApplyTo(patched, LoraTarget.Transformer, _backend);

        float[] baseline = Forward(_dense);
        float[] off = Forward(patched);
        float[] on, half;
        using (gate.Enable(1f))
            on = Forward(patched);
        using (gate.Enable(0.5f))
            half = Forward(patched);

        Assert.Equal(baseline, off);
        Dictionary<string, Tensor> merged = fixture.MergedInto(_dense, 1f), mergedHalf = fixture.MergedInto(_dense, 0.5f);
        Assert.True(LongCatTestData.RelRms(on, Forward(merged)) < 2e-5, "gate on at 1 is the merged weight");
        Assert.True(LongCatTestData.RelRms(half, Forward(mergedHalf)) < 2e-5, "gate on at 0.5 is the half-strength merged weight");
        Assert.True(LongCatTestData.RelRms(on, baseline) > 1e-3, "the LoRA must change the output");
        Dispose(merged, _dense);
        Dispose(mergedHalf, _dense);
    }

    [Fact]
    public void PerStepSwitchingRunsTheFirstLoraThenTheSecond()
    {
        using LongCatLoraFixture first = new(PathOf("a.safetensors"), _config, 11);
        using LongCatLoraFixture second = new(PathOf("b.safetensors"), _config, 12);
        using LoraFile fileA = LoraFile.Load(first.Path);
        using LoraFile fileB = LoraFile.Load(second.Path);
        LongCatLoraMix mix = new(mixSteps: 2);
        using LoraStack stack = new();
        stack.AddGated(fileA, mix.First);
        stack.AddGated(fileB, mix.Second);
        Dictionary<string, Tensor> patched = new(_dense, StringComparer.Ordinal);
        stack.ApplyTo(patched, LoraTarget.Transformer, _backend);

        float[][] steps = new float[4][];
        for (int step = 0; step < 4; step++)
            using (mix.ForStep(step))
                steps[step] = Forward(patched);
        float[] after = Forward(patched);

        Dictionary<string, Tensor> mergedA = first.MergedInto(_dense, 1f), mergedB = second.MergedInto(_dense, 1f);
        float[] expectedA = Forward(mergedA), expectedB = Forward(mergedB);
        Assert.True(LongCatTestData.RelRms(steps[0], expectedA) < 2e-5);
        Assert.True(LongCatTestData.RelRms(steps[1], expectedA) < 2e-5);
        Assert.True(LongCatTestData.RelRms(steps[2], expectedB) < 2e-5);
        Assert.True(LongCatTestData.RelRms(steps[3], expectedB) < 2e-5);
        Assert.True(LongCatTestData.RelRms(expectedA, expectedB) > 1e-3, "the two LoRAs must differ");
        Assert.Equal(Forward(_dense), after);
        Dispose(mergedA, _dense);
        Dispose(mergedB, _dense);
    }

    [Fact]
    public void GateStrengthMustBeNonNegative()
    {
        LongCatLoraMix mix = new(2);

        Assert.Throws<ArgumentOutOfRangeException>(() => mix.ForStep(0, -0.1f));
        Assert.Same(mix.First, mix.GateFor(1));
        Assert.Same(mix.Second, mix.GateFor(2));
    }

    [Fact]
    public void AnInt8BaseCarriesTheGatedLoraToo()
    {
        Dictionary<string, Tensor> quantized = LongCatTestData.QuantizePerRow(_dense);
        string ckpt = PathOf("int8");
        Directory.CreateDirectory(ckpt);
        LongCatTestData.WriteCheckpoint(ckpt, quantized, indexFileName: LongCatCheckpoint.Int8IndexFileName);
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(ckpt);
        Dictionary<string, Tensor> widened = new(checkpoint.Weights, StringComparer.Ordinal);
        using QuantizedWeightPolicy.PreparedWeights prepared = QuantizedWeightPolicy.PrepareForBackends(widened, [_backend], DType.F32);
        using LongCatLoraFixture fixture = new(PathOf("a.safetensors"), _config, 17);
        using LoraFile file = LoraFile.Load(fixture.Path);
        using LoraStack stack = new();
        AdjunctGate gate = new();
        stack.AddGated(file, gate);
        Dictionary<string, Tensor> patched = new(widened, StringComparer.Ordinal);
        stack.ApplyTo(patched, LoraTarget.Transformer, _backend);

        float[] off = Forward(patched);
        float[] on;
        using (gate.Enable(1f))
            on = Forward(patched);

        Dictionary<string, Tensor> merged = fixture.MergedInto(widened, 1f);
        Assert.Equal(Forward(widened), off);
        Assert.True(LongCatTestData.RelRms(on, Forward(merged)) < 2e-5, "INT8 base plus gated LoRA is the merged weight");
        Assert.True(LongCatTestData.RelRms(on, off) > 1e-3);
        Dispose(merged, widened);
        foreach ((string key, Tensor t) in quantized)
            if (!_dense.ContainsKey(key))
                t.Dispose();
    }

    private float[] Forward(IReadOnlyDictionary<string, Tensor> weights)
    {
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(weights);
        LongCatConditioning conditioning = new(NumCondLatents: 1);
        using Tensor velocity = transformer.Forward(_backend, latent, text, conditioning.FrameTimesteps(500f, Frames), in conditioning);
        return velocity.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>Disposes the tensors of <paramref name="merged"/> that are not in <paramref name="original"/>.</summary>
    private static void Dispose(Dictionary<string, Tensor> merged, Dictionary<string, Tensor> original)
    {
        foreach ((string key, Tensor t) in merged)
            if (!original.TryGetValue(key, out Tensor? o) || !ReferenceEquals(o, t))
                t.Dispose();
    }
}
