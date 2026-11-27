using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A LongCat DiT from a GGUF repack in the ComfyUI Wan layout: the split query, key and value weights join into the engine's fused ones,
/// and the forward, the release LoRA and the device-fit count agree with the dense weights the file decodes to.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatGgufTests : IDisposable
{
    private const int Frames = 3, Height = 8, Width = 8, TextRows = 6;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "longcat-gguf-" + Guid.NewGuid().ToString("N"));
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _dense;

    public LongCatGgufTests()
    {
        Directory.CreateDirectory(_root);
        _dense = LongCatTestData.Weights(_config, 61);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_dense);
        Directory.Delete(_root, recursive: true);
    }

    [Theory]
    [InlineData("Q8_0", "Q8_0", 0)]
    [InlineData("Q4_0", "Q4_0", 0)]
    [InlineData("Q4_1", "Q4_1", 0)]
    [InlineData("Q5_0", "Q5_0", 0)]
    [InlineData("Q5_1", "Q5_1", 0)]
    // The "_M" and "_S" mixes: v in a wider type than q and k, so each block's two joined weights become BF16.
    [InlineData("Q4_0", "Q8_0", 4)]
    public void TheJoinedWeightsDecodeToTheirPartsInOrder(string quant, string valueQuant, int widened)
    {
        using Repack repack = WriteRepack(quant, valueQuant);
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(repack.Directory);

        Assert.False(checkpoint.IsInt8);
        Assert.Equal(LongCatCheckpoint.ExpectedShapes(_config).Keys.Order(), checkpoint.Weights.Keys.Order());
        // The file carries no config.json, so the geometry is the one its shapes give.
        LongCatConfig derived = checkpoint.Config;
        Assert.Equal((_config.HiddenSize, _config.Depth, _config.NumHeads), (derived.HiddenSize, derived.Depth, derived.NumHeads));
        DType joined = widened > 0 ? DType.BF16 : DTypeOf(quant);
        for (int i = 0; i < _config.Depth; i++)
        {
            Assert.Equal(joined, checkpoint.Weights[$"blocks.{i}.attn.qkv.weight"].DType);
            Assert.Equal(joined, checkpoint.Weights[$"blocks.{i}.cross_attn.kv_linear.weight"].DType);
            Assert.Equal(DTypeOf(quant), checkpoint.Weights[$"blocks.{i}.ffn.w1.weight"].DType);
        }
        foreach ((string key, Tensor held) in checkpoint.Weights)
        {
            float[] expected = repack.Decoded[key].AsReadOnlySpan<float>().ToArray();
            // A widened join, and a dense F32 tensor held as BF16, store each value rounded to nearest.
            if (held.DType == DType.BF16)
                for (int i = 0; i < expected.Length; i++)
                    expected[i] = BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(expected[i]));
            Assert.Equal(expected, Decode(held));
        }
    }

    [Theory]
    [InlineData("Q8_0", "Q8_0")]
    [InlineData("Q4_0", "Q8_0")]
    public void TheForwardIsTheDecodedDenseWeightsForward(string quant, string valueQuant)
    {
        using IDisposable f32 = LongCatTestData.PinF32Dit();
        using Repack repack = WriteRepack(quant, valueQuant);
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(repack.Directory);
        using CpuBackend backend = new();
        Dictionary<string, Tensor> loaded = new(checkpoint.Weights, StringComparer.Ordinal);
        using QuantizedWeightPolicy.PreparedWeights prepared =
            QuantizedWeightPolicy.PrepareForBackends(loaded, [backend], LongCatRecipeSupport.WidenedDType(checkpoint));
        Dictionary<string, Tensor> expected = ExpectedDense(repack, checkpoint);
        try
        {
            float[] got = Forward(backend, loaded), want = Forward(backend, expected), original = Forward(backend, _dense);

            Assert.True(LongCatTestData.RelRms(got, want) < 1e-6, $"relRMS {LongCatTestData.RelRms(got, want)}");
            Assert.True(LongCatTestData.RelRms(got, original) > 1e-4, "the repack is quantized, so it is not the dense forward");
        }
        finally
        {
            LongCatTestData.DisposeAll(expected);
        }
    }

    [Fact]
    public void TheReleaseLoraAttachesToTheJoinedWeights()
    {
        using IDisposable f32 = LongCatTestData.PinF32Dit();
        using Repack repack = WriteRepack("Q8_0", "Q8_0");
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(repack.Directory);
        using CpuBackend backend = new();
        Dictionary<string, Tensor> loaded = new(checkpoint.Weights, StringComparer.Ordinal);
        using QuantizedWeightPolicy.PreparedWeights prepared =
            QuantizedWeightPolicy.PrepareForBackends(loaded, [backend], LongCatRecipeSupport.WidenedDType(checkpoint));
        // The LoRA's fused modules (attn.qkv, cross_attn.kv_linear) are the ones the loader builds from the split parts.
        using LongCatLoraFixture fixture = new(Path.Combine(_root, "dmd.safetensors"), _config, 7);
        using LoraFile file = LoraFile.Load(fixture.Path);
        using LoraStack stack = new();
        AdjunctGate gate = new();
        stack.AddGated(file, gate);
        Dictionary<string, Tensor> patched = new(loaded, StringComparer.Ordinal);
        stack.ApplyTo(patched, LoraTarget.Transformer, backend);
        Dictionary<string, Tensor> dense = ExpectedDense(repack, checkpoint);
        Dictionary<string, Tensor> merged = fixture.MergedInto(dense, 1f);
        try
        {
            float[] off = Forward(backend, patched);
            float[] on;
            using (gate.Enable(1f))
                on = Forward(backend, patched);

            LongCatRecipeSupport.RequireFullyApplied(stack.Report, "dmd.safetensors");
            Assert.Equal(Forward(backend, loaded), off);
            Assert.True(LongCatTestData.RelRms(on, Forward(backend, merged)) < 2e-5, "gate on at 1 is the merged weight");
            Assert.True(LongCatTestData.RelRms(on, off) > 1e-3, "the LoRA must change the output");
        }
        finally
        {
            foreach ((string key, Tensor tensor) in merged)
                if (!dense.TryGetValue(key, out Tensor? kept) || !ReferenceEquals(kept, tensor))
                    tensor.Dispose();
            LongCatTestData.DisposeAll(dense);
        }
    }

    [Theory]
    [InlineData("Q8_0", "Q8_0", false)]
    [InlineData("Q8_0", "Q8_0", true)]
    [InlineData("Q4_0", "Q8_0", false)]
    [InlineData("Q4_0", "Q8_0", true)]
    public void TheDeviceFitCountsTheWeightsAsTheyAreHeld(string quant, string valueQuant, bool packed)
    {
        using Repack repack = WriteRepack(quant, valueQuant);
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(repack.Directory);
        Dictionary<string, Tensor> weights = new(checkpoint.Weights, StringComparer.Ordinal);
        using QuantizedWeightPolicy.PreparedWeights prepared =
            QuantizedWeightPolicy.PrepareFor(weights, _ => packed, "stand-in", LongCatRecipeSupport.WidenedDType(checkpoint));

        long held = weights.Values.Sum(static t => t.DType.ComputeByteCount(t.ElementCount));
        (long bytes, long widened) = LongCatWeightFit.HeldBytes(repack.Directory, _ => packed);

        Assert.Equal(held, bytes);
        Assert.Equal(prepared.WidenedCount, widened);
    }

    [Theory]
    [InlineData("Q4_1")]
    [InlineData("Q5_1")]
    public void ATypeTheBackendCannotDecodeIsHeldAsItsBf16ValuesNotF32(string quant)
    {
        // Neither CUDA nor Vulkan decodes these two; widened to F32 the released files would hold about 57 GiB of host weights.
        using Repack repack = WriteRepack(quant, quant);
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(repack.Directory);
        Dictionary<string, Tensor> weights = new(checkpoint.Weights, StringComparer.Ordinal);

        using QuantizedWeightPolicy.PreparedWeights prepared = QuantizedWeightPolicy.PrepareFor(weights,
            weight => weight.DType != DTypeOf(quant), "stand-in", LongCatRecipeSupport.WidenedDType(checkpoint));

        Assert.True(prepared.WidenedCount > 0);
        Assert.Equal(prepared.WidenedCount, LongCatWeightFit.HeldBytes(repack.Directory, weight => weight.DType != DTypeOf(quant)).Widened);
        foreach ((string key, Tensor held) in weights)
        {
            Assert.False(held.DType.IsQuantized);
            if (!checkpoint.Weights[key].DType.IsQuantized)
                continue;
            Assert.Equal(DType.BF16, held.DType);
            float[] expected = repack.Decoded[key].AsReadOnlySpan<float>().ToArray();
            for (int i = 0; i < expected.Length; i++)
                expected[i] = BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(expected[i]));
            Assert.Equal(expected, Decode(held));
        }
    }

    [Fact]
    public void OnlyARepackWidensToBf16()
    {
        using Repack repack = WriteRepack("Q8_0", "Q8_0");
        string dense = Path.Combine(_root, "dense");
        Directory.CreateDirectory(dense);
        LongCatTestData.WriteCheckpoint(dense, _dense);
        using LongCatCheckpoint gguf = LongCatCheckpoint.Open(repack.Directory);
        using LongCatCheckpoint safetensors = LongCatCheckpoint.Open(dense);

        Assert.Equal((DType.BF16, DType.F32),
            (LongCatRecipeSupport.WidenedDType(gguf), LongCatRecipeSupport.WidenedDType(safetensors)));
    }

    [Fact]
    public void AFolderWithTwoRepacksIsRefusedByName()
    {
        using Repack repack = WriteRepack("Q8_0", "Q8_0");
        File.Copy(Path.Combine(repack.Directory, "model.gguf"), Path.Combine(repack.Directory, "other.gguf"));

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => LongCatCheckpoint.Open(repack.Directory));

        Assert.Contains("model.gguf, other.gguf", error.Message);
    }

    [Fact]
    public void ARepackBesideASafetensorsIndexIsRefused()
    {
        using Repack repack = WriteRepack("Q8_0", "Q8_0");
        LongCatTestData.WriteCheckpoint(repack.Directory, _dense);

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => LongCatCheckpoint.Open(repack.Directory));

        Assert.Contains("both a safetensors index and model.gguf", error.Message);
    }

    [Fact]
    public void ARepackMissingOneSplitPartIsRefused()
    {
        using Repack repack = WriteRepack("Q8_0", "Q8_0", skip: "blocks.1.self_attn.k.weight");

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => LongCatCheckpoint.Open(repack.Directory));

        Assert.Contains("blocks.1.attn.qkv.weight", error.Message);
    }

    [Fact]
    public void ARepackInAnotherLayoutIsRefused()
    {
        string directory = Path.Combine(_root, "other");
        Directory.CreateDirectory(directory);
        using (GgufWriter writer = new(Path.Combine(directory, "model.gguf")))
        {
            writer.SetMetadata("general.architecture", "wan");
            writer.AddTensor("blocks.0.self_attn.q.weight", _dense["blocks.0.attn.proj.weight"]);
            writer.Flush();
        }

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => LongCatCheckpoint.Open(directory));

        Assert.Contains("ComfyUI Wan layout", error.Message);
    }

    /// <summary>A written repack's folder and decoded weights.</summary>
    private sealed record Repack(string Directory, Dictionary<string, Tensor> Decoded) : IDisposable
    {
        public void Dispose() => LongCatTestData.DisposeAll(Decoded);
    }

    /// <summary>The written repack: its folder, and each engine weight as the file decodes it (F32).</summary>
    private Repack WriteRepack(string quant, string valueQuant, string? skip = null)
    {
        string directory = Path.Combine(_root, Guid.NewGuid().ToString("N"));
        System.IO.Directory.CreateDirectory(directory);
        return new Repack(directory, LongCatGgufTestData.Write(directory, _dense, quant, valueQuant, skip));
    }

    /// <summary>The dense weights the checkpoint should behave as: the file's decoded values, BF16-rounded wherever the engine holds BF16 (a
    /// widened join) or widens to it (a block quant the CPU cannot hold packed).</summary>
    private static Dictionary<string, Tensor> ExpectedDense(Repack repack, LongCatCheckpoint checkpoint)
    {
        Dictionary<string, Tensor> result = new(StringComparer.Ordinal);
        foreach ((string key, Tensor decoded) in repack.Decoded)
        {
            Tensor copy = new(decoded.Shape, DType.F32);
            Span<float> values = copy.AsSpan<float>();
            decoded.AsReadOnlySpan<float>().CopyTo(values);
            if (checkpoint.Weights[key].DType == DType.BF16 || checkpoint.Weights[key].DType.IsQuantized)
                for (int i = 0; i < values.Length; i++)
                    values[i] = BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(values[i]));
            result[key] = copy;
        }
        return result;
    }

    private static DType DTypeOf(string name) => LongCatGgufTestData.DTypeOf(name);

    private static float[] Decode(Tensor tensor) => LongCatGgufTestData.Decode(tensor);

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
