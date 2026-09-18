using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the fold that has to happen <b>before</b> an architecture converter runs. A converter renames
/// <c>blocks.0.attn.qkv.weight</c> and has no rule for <c>blocks.0.attn.qkv.weight_scale</c>, so a fold that ran
/// afterwards would find no pairing and drop the scale — the Krea2 failure, where the weights came out hundreds of
/// times too large and the output was noise rather than an error. Folding at the container makes that unreachable.</summary>
public sealed unsafe class QuantCompanionFoldTests : IDisposable
{
    private readonly string _tempDir;

    public QuantCompanionFoldTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"companion_fold_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); }
        catch (IOException) { /* a mmap the OS has not released yet is not a test failure */ }
    }

    private static Tensor Scalar(float value)
    {
        Tensor tensor = new Tensor(new TensorShape(1), DType.F32);
        tensor.AsSpan<float>()[0] = value;
        return tensor;
    }

    private static Tensor RowScale(int rows)
    {
        Tensor tensor = new Tensor(new TensorShape(rows, 1), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < rows; i++) values[i] = 0.01f * (i + 1);
        return tensor;
    }

    private static Tensor Blob(string json)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        Tensor tensor = new Tensor(new TensorShape(bytes.Length), DType.U8);
        bytes.CopyTo(tensor.AsSpan<byte>());
        return tensor;
    }

    private string Write(Dictionary<string, Tensor> weights, string name = "model.safetensors")
    {
        string path = Path.Combine(_tempDir, name);
        SafeTensorsWriter.Save(path, weights);
        return path;
    }

    [Fact]
    public void Open_FoldsAnFp8ScaledCompanionAndDropsIt()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.attn.qkv.weight"] = new Tensor(new TensorShape(4, 8), DType.F8E4M3),
            ["blocks.0.attn.qkv.scale_weight"] = Scalar(0.0195f),
            ["blocks.0.attn.qkv.scale_input"] = Scalar(0.0625f),
        };
        try
        {
            using CheckpointSource source = CheckpointSource.Open(Write(weights));

            Tensor folded = source.Weights["blocks.0.attn.qkv.weight"];
            Assert.Equal(0.0195f, folded.Fp8ScaleFactor);
            Assert.Equal(0.0625f, folded.Fp8InputScaleFactor);
            Assert.DoesNotContain("blocks.0.attn.qkv.scale_weight", source.Weights.Keys);
            Assert.DoesNotContain("blocks.0.attn.qkv.scale_input", source.Weights.Keys);
            // The header still shows the file as written — a planner inspecting companions reads it, not the fold.
            Assert.Contains("blocks.0.attn.qkv.scale_weight", source.Header.Descriptors.Keys);
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Open_FoldsInt8ConvRotCompanionsOntoQuantInfoAndLeavesTheWeightPacked()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.attn.qkv.weight"] = new Tensor(new TensorShape(4, 256), DType.I8),
            ["blocks.0.attn.qkv.weight_scale"] = RowScale(4),
            ["blocks.0.attn.qkv.comfy_quant"] =
                Blob("{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": 256, \"per_row\": true}"),
        };
        try
        {
            using CheckpointSource source = CheckpointSource.Open(Write(weights));

            Tensor folded = source.Weights["blocks.0.attn.qkv.weight"];
            // Dequantizing here would undo the entire point of the format: LTX 2.5's 21.5 GB DiT becomes 42 GB.
            Assert.Equal(DType.I8, folded.DType);
            QuantWeightInfo info = Assert.IsType<QuantWeightInfo>(folded.QuantInfo);
            Assert.Equal("int8_tensorwise", info.Format);
            Assert.Equal(256, info.ConvRotGroupSize);
            Assert.Equal(4L, info.RowScale!.Shape[0]);
            Assert.DoesNotContain("blocks.0.attn.qkv.weight_scale", source.Weights.Keys);
            Assert.DoesNotContain("blocks.0.attn.qkv.comfy_quant", source.Weights.Keys);
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Open_WithFoldingOff_LeavesTheCompanionsWhereTheFileWroteThem()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.attn.qkv.weight"] = new Tensor(new TensorShape(4, 8), DType.F8E4M3),
            ["blocks.0.attn.qkv.scale_weight"] = Scalar(0.0195f),
        };
        try
        {
            using CheckpointSource source = CheckpointSource.Open(Write(weights),
                new CheckpointOpenOptions { FoldQuantCompanions = false });

            Assert.Contains("blocks.0.attn.qkv.scale_weight", source.Weights.Keys);
            Assert.Equal(1.0f, source.Weights["blocks.0.attn.qkv.weight"].Fp8ScaleFactor);
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void FoldThenSplit_GivesEachProjectionItsOwnScales()
    {
        // The end-to-end shape of the fix: the container folds the fused weight's companions, and the converter's
        // QKV split then carries them per projection. Folding after the split instead finds nothing to pair.
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.attn.qkv.weight"] = new Tensor(new TensorShape(12, 256), DType.I8),
            ["blocks.0.attn.qkv.weight_scale"] = RowScale(12),
            ["blocks.0.attn.qkv.comfy_quant"] = Blob("{\"format\": \"int8_tensorwise\", \"per_row\": true}"),
        };
        try
        {
            using CheckpointSource source = CheckpointSource.Open(Write(weights));

            Dictionary<string, Tensor> converted = new();
            CheckpointConverters.Utils.CheckpointConvertUtils.SplitQkvWeight(
                source.Weights["blocks.0.attn.qkv.weight"], 4, "blocks.0.attn", "to_q", "to_k", "to_v", converted);
            try
            {
                Assert.Equal(0.01f, converted["blocks.0.attn.to_q.weight"].QuantInfo!.RowScale!.AsReadOnlySpan<float>()[0], 6);
                Assert.Equal(0.05f, converted["blocks.0.attn.to_k.weight"].QuantInfo!.RowScale!.AsReadOnlySpan<float>()[0], 6);
                Assert.Equal(0.09f, converted["blocks.0.attn.to_v.weight"].QuantInfo!.RowScale!.AsReadOnlySpan<float>()[0], 6);
            }
            finally
            {
                foreach (Tensor tensor in converted.Values) tensor.Dispose();
            }
        }
        finally
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
        }
    }
}
