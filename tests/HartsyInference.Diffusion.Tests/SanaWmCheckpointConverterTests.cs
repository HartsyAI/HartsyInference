using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SANA-WM converter on a tiny synthetic safetensors file: round trip, key validation, and the dtype policy.</summary>
public sealed class SanaWmCheckpointConverterTests : IDisposable
{
    private const string GdnBlockKey = "blocks.0.attn.qkv.weight";
    private const string SoftmaxBlockKey = "blocks.1.attn.proj.weight";

    private readonly string _directory = Path.Combine(Path.GetTempPath(), "sanawm-" + Guid.NewGuid().ToString("N"));

    public SanaWmCheckpointConverterTests()
    {
        Directory.CreateDirectory(_directory);
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory))
            Directory.Delete(_directory, recursive: true);
    }

    [Fact]
    public void Load_SyntheticFile_DerivesConfigAndCopiesEveryTensorUnderItsUpstreamName()
    {
        string path = FileFor("round-trip.safetensors");
        SanaWmSyntheticLayout.WriteFile(path);

        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(path);

        SanaWmConfig config = checkpoint.Config;
        Assert.Equal(2, config.Depth);
        Assert.Equal(224, config.HiddenSize);
        Assert.Equal(2, config.NumHeads);
        Assert.Equal(112, config.HeadDim);
        Assert.Equal(8, config.InChannels);
        Assert.Equal(448, config.FfnHidden);
        Assert.Equal(16, config.TextWidth);
        Assert.Equal(6, config.NullCaptionLength);
        Assert.Equal(16, config.TimestepWidth);
        Assert.Equal(4, config.ConvKernelSize);
        Assert.Equal(5, config.PosEmbedTokens);
        Assert.Equal(6, config.ModulationRows);
        Assert.Equal(new[] { 1 }, config.SoftmaxBlocks);
        Assert.True(config.HasCameraBranch);
        Assert.True(config.HasPluckerBranch);
        Assert.Equal(12, config.PluckerChannels);
        Assert.True(config.HasRaymapEmbedder);
        Assert.Equal(3, config.RaymapChannels);

        IReadOnlyList<SanaWmTensorSpec> expected = SanaWmCheckpointConverter.ExpectedTensors(config);
        Assert.Equal(106, expected.Count);
        Assert.Equal(expected.Count, checkpoint.Weights.Count);
        foreach (SanaWmTensorSpec spec in expected)
        {
            Assert.True(checkpoint.Weights.TryGetValue(spec.Name, out Tensor? tensor), $"missing {spec.Name}");
            Assert.Equal(spec.DType, tensor!.DType);
            Assert.Equal(spec.Shape, tensor.Shape);
        }
    }

    [Fact]
    public void Load_DtypePolicy_StoresWeightsAsBf16AndStatesAsF32()
    {
        string path = FileFor("dtype.safetensors");
        SanaWmSyntheticLayout.WriteFile(path);

        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(path);

        AssertDtype(checkpoint, "blocks.0.attn.qkv.weight", DType.BF16);
        AssertDtype(checkpoint, "blocks.0.attn.output_gate.weight", DType.BF16);
        AssertDtype(checkpoint, "blocks.0.attn.conv_k.weight", DType.BF16);
        AssertDtype(checkpoint, "blocks.0.attn.conv_k_cam.weight", DType.BF16);
        AssertDtype(checkpoint, "blocks.0.mlp.point_conv.conv.weight", DType.BF16);
        AssertDtype(checkpoint, "blocks.1.attn.qkv.weight", DType.BF16);
        AssertDtype(checkpoint, "y_embedder.y_embedding", DType.BF16);
        AssertDtype(checkpoint, "blocks.0.attn.A_log", DType.F32);
        AssertDtype(checkpoint, "blocks.0.attn.beta_proj.weight", DType.F32);
        AssertDtype(checkpoint, "blocks.0.attn.gate_proj.weight", DType.F32);
        AssertDtype(checkpoint, "blocks.0.attn.recall_gate", DType.F32);
        AssertDtype(checkpoint, "blocks.0.attn.q_norm.weight", DType.F32);
        AssertDtype(checkpoint, "blocks.0.attn.k_norm_cam.weight", DType.F32);
        AssertDtype(checkpoint, "blocks.0.attn.proj.bias", DType.F32);
        AssertDtype(checkpoint, "blocks.0.scale_shift_table", DType.F32);
        AssertDtype(checkpoint, "t_block.1.weight", DType.F32);
        AssertDtype(checkpoint, "t_embedder.mlp.2.weight", DType.F32);
        AssertDtype(checkpoint, "pos_embed", DType.F32);
    }

    [Fact]
    public void Load_RoundsF32ToBf16ToNearestEven_AndCopiesF32Exactly()
    {
        string path = FileFor("rounding.safetensors");
        SanaWmSyntheticLayout.WriteFile(path);

        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(path);

        // 1 + 3·2^-8 sits between BF16 0x3F81 (truncation) and 0x3F82 (round-to-nearest-even).
        Assert.Equal((ushort)0x3F82, checkpoint.Weights[GdnBlockKey].AsSpan<ushort>()[0]);
        Assert.Equal(SanaWmSyntheticLayout.RoundingTie, checkpoint.Weights["blocks.0.attn.A_log"].AsSpan<float>()[0]);
    }

    [Fact]
    public void Load_BfSourceTensor_KeepsItsStoredBits()
    {
        string path = FileFor("bf16-passthrough.safetensors");
        SanaWmSyntheticLayout.WriteFile(path);

        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(path);

        Span<ushort> embedding = checkpoint.Weights["y_embedder.y_embedding"].AsSpan<ushort>();
        Assert.Equal(SafeTensorsMerger.ToBf16(SanaWmSyntheticLayout.PatternValue(1)), embedding[1]);
        Assert.Equal(SafeTensorsMerger.ToBf16(SanaWmSyntheticLayout.PatternValue(embedding.Length - 1)), embedding[^1]);
    }

    [Fact]
    public void Load_MissingKey_ThrowsNamingTheKey()
    {
        string path = FileFor("missing.safetensors");
        SanaWmSyntheticLayout.WriteFile(path, specs => specs.RemoveAll(spec => spec.Name == SoftmaxBlockKey));

        HartsyInferenceException error = LoadFailure(path);

        Assert.Contains($"missing {SoftmaxBlockKey}", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_UnexpectedKey_ThrowsNamingTheKey()
    {
        string path = FileFor("unexpected.safetensors");
        SanaWmSyntheticLayout.WriteFile(path, specs =>
            specs.Add(new SanaWmTensorSpec("blocks.0.attn.bogus.weight", DType.F32, new TensorShape(4))));

        HartsyInferenceException error = LoadFailure(path);

        Assert.Contains("unexpected blocks.0.attn.bogus.weight", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_ShapeMismatch_ThrowsWithExpectedAndActualShape()
    {
        string path = FileFor("shape.safetensors");
        SanaWmSyntheticLayout.WriteFile(path, specs =>
        {
            int index = specs.FindIndex(spec => spec.Name == "blocks.0.attn.proj.bias");
            specs[index] = specs[index] with { Shape = new TensorShape(7) };
        });

        HartsyInferenceException error = LoadFailure(path);

        Assert.Contains("blocks.0.attn.proj.bias: shape [7], expected [224]", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_SeveralMismatches_ListsEveryOneInOneError()
    {
        string path = FileFor("several.safetensors");
        SanaWmSyntheticLayout.WriteFile(path, specs =>
        {
            specs.RemoveAll(spec => spec.Name == SoftmaxBlockKey);
            specs.RemoveAll(spec => spec.Name == "final_layer.linear.bias");
            specs.Add(new SanaWmTensorSpec("extra.weight", DType.F32, new TensorShape(2)));
        });

        HartsyInferenceException error = LoadFailure(path);

        Assert.Contains("(3 mismatches)", error.Message, StringComparison.Ordinal);
        Assert.Contains($"missing {SoftmaxBlockKey}", error.Message, StringComparison.Ordinal);
        Assert.Contains("missing final_layer.linear.bias", error.Message, StringComparison.Ordinal);
        Assert.Contains("unexpected extra.weight", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Dispose_CalledTwice_IsIdempotent()
    {
        string path = FileFor("dispose.safetensors");
        SanaWmSyntheticLayout.WriteFile(path);
        SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(path);

        checkpoint.Dispose();
        checkpoint.Dispose();
    }

    private string FileFor(string name) => Path.Combine(_directory, name);

    private static HartsyInferenceException LoadFailure(string path) =>
        Assert.Throws<HartsyInferenceException>(() => SanaWmCheckpointConverter.Load(path).Dispose());

    private static void AssertDtype(SanaWmCheckpoint checkpoint, string name, DType expected)
    {
        Assert.True(checkpoint.Weights.TryGetValue(name, out Tensor? tensor), $"missing converted tensor {name}");
        Assert.Equal(expected, tensor!.DType);
    }
}
