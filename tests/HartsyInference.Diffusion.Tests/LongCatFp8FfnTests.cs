using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary><c>numerics.longcatFp8Ffn</c>: the FFN input projections the reference runs in FP8 are held as FP8 E4M3 weights, on a tiny DiT of the
/// release's 48 blocks.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatFp8FfnTests : IDisposable
{
    private const int Frames = 3, Height = 8, Width = 8, TextRows = 6;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "longcat-fp8-" + Guid.NewGuid().ToString("N"));
    private readonly LongCatConfig _config = LongCatTestData.Tiny with { Depth = 48 };
    private readonly Dictionary<string, Tensor> _dense;
    private readonly string _checkpoint;

    public LongCatFp8FfnTests()
    {
        _dense = LongCatTestData.Weights(_config, 97);
        _checkpoint = Write("dense", _config, _dense);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_dense);
        Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void TheSelectionIsTheReferencesFortyBlocks()
    {
        Assert.Equal(40, LongCatFp8FfnConverter.Blocks.Count);
        Assert.Equal([6, 9, 10], LongCatFp8FfnConverter.Blocks.Take(3));
        Assert.Equal(47, LongCatFp8FfnConverter.Blocks[^1]);
        Assert.Equal(80, LongCatFp8FfnConverter.Keys().Distinct().Count());
    }

    [Fact]
    public void OffTheWeightsStayBf16()
    {
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(_checkpoint);

        Assert.DoesNotContain(checkpoint.Weights.Values, static t => t.DType == DType.F8E4M3);
    }

    [Fact]
    public void OnExactlyTheSelectionIsFp8WithinItsBoundOfItsBf16Weight()
    {
        using LongCatCheckpoint bf16 = LongCatCheckpoint.Open(_checkpoint);
        using IDisposable on = Fp8On();
        using LongCatCheckpoint fp8 = LongCatCheckpoint.Open(_checkpoint);

        Assert.Equal(LongCatFp8FfnConverter.Keys().Order(StringComparer.Ordinal),
            fp8.Weights.Where(entry => entry.Value.DType == DType.F8E4M3).Select(entry => entry.Key).Order(StringComparer.Ordinal));
        foreach (string key in LongCatFp8FfnConverter.Keys())
        {
            using Tensor reference = bf16.Weights[key].CastTo(DType.F32);
            LongCatFp8FfnCheck.Report report = LongCatFp8FfnCheck.Compare(reference, fp8.Weights[key]);
            Assert.True(report.Violations == 0 && report.ScaleIsAbsMaxOver448, $"{key}: {report}");
        }
    }

    [Fact]
    public void TheForwardAndTheLoraAreThoseOfTheDecodedWeights()
    {
        using LongCatCheckpoint bf16 = LongCatCheckpoint.Open(_checkpoint);
        using IDisposable on = Fp8On();
        using LongCatCheckpoint fp8 = LongCatCheckpoint.Open(_checkpoint);
        using CpuBackend backend = new();
        Dictionary<string, Tensor> loaded = new(fp8.Weights, StringComparer.Ordinal);
        using QuantizedWeightPolicy.PreparedWeights prepared = QuantizedWeightPolicy.PrepareForBackends(loaded, [backend], DType.F32);
        Dictionary<string, Tensor> decoded = new(loaded, StringComparer.Ordinal);
        List<Tensor> made = [];
        foreach (string key in LongCatFp8FfnConverter.Keys())
            made.Add(decoded[key] = loaded[key].CastTo(DType.F32));
        using LongCatLoraFixture fixture = new(Path.Combine(_root, "lora.safetensors"), _config, 7);
        using LoraFile file = LoraFile.Load(fixture.Path);
        using LoraStack stack = new();
        AdjunctGate gate = new();
        stack.AddGated(file, gate);
        Dictionary<string, Tensor> patched = new(loaded, StringComparer.Ordinal);
        stack.ApplyTo(patched, LoraTarget.Transformer, backend);
        // The fixture merges into F32 weights, so the merge starts from an F32 copy of everything the LoRA adapts.
        Dictionary<string, Tensor> wide = new(decoded, StringComparer.Ordinal);
        foreach (string module in fixture.Modules)
            if (wide[module + ".weight"].DType != DType.F32)
                made.Add(wide[module + ".weight"] = wide[module + ".weight"].CastTo(DType.F32));
        Dictionary<string, Tensor> merged = fixture.MergedInto(wide, 1f);
        try
        {
            float[] off = Forward(backend, patched);
            float[] gated;
            using (gate.Enable(1f))
                gated = Forward(backend, patched);

            LongCatRecipeSupport.RequireFullyApplied(stack.Report, "lora.safetensors");
            Assert.True(LongCatTestData.RelRms(off, Forward(backend, decoded)) < 1e-5, "the FP8 forward is the decoded weights' forward");
            Assert.True(LongCatTestData.RelRms(off, Forward(backend, bf16.Weights)) > 1e-5, "the FP8 weights are not the BF16 ones");
            Assert.True(LongCatTestData.RelRms(gated, Forward(backend, merged)) < 2e-5, "gate on at 1 is the merged weight");
            Assert.True(LongCatTestData.RelRms(gated, off) > 1e-3, "the LoRA must change the output");
        }
        finally
        {
            foreach ((string key, Tensor tensor) in merged)
                if (!wide.TryGetValue(key, out Tensor? kept) || !ReferenceEquals(kept, tensor))
                    tensor.Dispose();
            foreach (Tensor tensor in made) tensor.Dispose();
        }
    }

    [Fact]
    public void TheDeviceFitCountsTheSelectionAtOneByte()
    {
        using IDisposable on = Fp8On();
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(_checkpoint);

        long held = checkpoint.Weights.Values.Sum(static t => t.DType.ComputeByteCount(t.ElementCount));

        Assert.Equal(held, LongCatWeightFit.HeldBytes(_checkpoint, static _ => false).Bytes);
    }

    [Fact]
    public void AnInt8ReleaseARepackAndAShallowDitAreRefused()
    {
        LongCatConfig shallow = LongCatTestData.Tiny;
        Dictionary<string, Tensor> two = LongCatTestData.Weights(shallow, 5);
        Dictionary<string, Tensor> int8 = LongCatTestData.QuantizePerRow(two);
        try
        {
            string shallowFolder = Write("shallow", shallow, two);
            string int8Folder = Path.Combine(_root, "int8");
            Directory.CreateDirectory(int8Folder);
            LongCatTestData.WriteCheckpoint(int8Folder, int8, indexFileName: LongCatCheckpoint.Int8IndexFileName);
            string repack = Path.Combine(_root, "repack");
            Directory.CreateDirectory(repack);
            LongCatTestData.DisposeAll(LongCatGgufTestData.Write(repack, two, "Q8_0", "Q8_0"));
            using IDisposable on = Fp8On();

            Assert.Contains("blocks.6.ffn.w1.weight", Assert.Throws<InvalidDataException>(() => LongCatCheckpoint.Open(shallowFolder)).Message);
            Assert.Contains("INT8", Assert.Throws<NotSupportedException>(() => LongCatCheckpoint.Open(int8Folder)).Message);
            Assert.Contains("GGUF", Assert.Throws<NotSupportedException>(() => LongCatCheckpoint.Open(repack)).Message);
        }
        finally
        {
            foreach ((string key, Tensor tensor) in int8)
                if (!two.TryGetValue(key, out Tensor? kept) || !ReferenceEquals(kept, tensor))
                    tensor.Dispose();
            LongCatTestData.DisposeAll(two);
        }
    }

    private static IDisposable Fp8On() => KnobProfile.Create("longcat-fp8-ffn").With(EngineKnobs.LongCatFp8Ffn, true).Push();

    private string Write(string name, LongCatConfig config, Dictionary<string, Tensor> weights)
    {
        string folder = Path.Combine(_root, name);
        Directory.CreateDirectory(folder);
        LongCatTestData.WriteCheckpoint(folder, weights, writeConfig: false, storage: DType.BF16);
        File.WriteAllText(Path.Combine(folder, "config.json"), LongCatTinyAvatarRelease.ConfigJson(config));
        return folder;
    }

    private float[] Forward(CpuBackend backend, IReadOnlyDictionary<string, Tensor> weights)
    {
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(weights);
        LongCatConditioning conditioning = new(NumCondLatents: 1);
        using Tensor velocity = transformer.Forward(backend, latent, text, conditioning.FrameTimesteps(500f, Frames), in conditioning);
        return velocity.AsReadOnlySpan<float>().ToArray();
    }
}
