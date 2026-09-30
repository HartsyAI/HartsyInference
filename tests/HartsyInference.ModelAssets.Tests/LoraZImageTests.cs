using System.Text;
using System.Text.Json;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Z-Image (Tongyi Lumina2/NextDiT) LoRA support: detection of the Comfy <c>diffusion_model.</c> wrapper
/// over the three block roots, the pure body→canonical mapping, and the numeric merge that proves the split Q/K/V
/// targets land in the right rows of the checkpoint's FUSED <c>attention.qkv.weight</c>.</summary>
public sealed unsafe class LoraZImageTests : IDisposable
{
    private readonly string _tempDir;
    private readonly ITestOutputHelper _output;

    public LoraZImageTests(ITestOutputHelper output)
    {
        _output = output;
        _tempDir = Path.Combine(Path.GetTempPath(), $"lora_zimage_tests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    private static SafeTensorDescriptor Desc(string name) => new()
    {
        Name = name,
        DType = DType.BF16,
        Shape = new TensorShape(32, 3840),
        DataOffset = 0,
        ByteLength = 32 * 3840 * 2,
    };

    private static Dictionary<string, SafeTensorDescriptor> Descriptors(params string[] keys)
    {
        Dictionary<string, SafeTensorDescriptor> d = new();
        foreach (string k in keys) d[k] = Desc(k);
        return d;
    }

    [Fact]
    public void ComfyOrgTurboDistillKeys_DetectedAsComfyZImageDit()
    {
        // Every distinct shape the published file carries: three block roots × attention/feed_forward.
        Dictionary<string, SafeTensorDescriptor> d = Descriptors(
            "diffusion_model.layers.0.attention.to_q.lora_A.default.weight",
            "diffusion_model.layers.0.attention.to_q.lora_B.default.weight",
            "diffusion_model.layers.29.attention.to_out.0.lora_A.default.weight",
            "diffusion_model.layers.29.attention.to_out.0.lora_B.default.weight",
            "diffusion_model.layers.7.feed_forward.w2.lora_A.default.weight",
            "diffusion_model.layers.7.feed_forward.w2.lora_B.default.weight",
            "diffusion_model.context_refiner.1.attention.to_k.lora_A.default.weight",
            "diffusion_model.context_refiner.1.attention.to_k.lora_B.default.weight",
            "diffusion_model.noise_refiner.0.feed_forward.w3.lora_A.default.weight",
            "diffusion_model.noise_refiner.0.feed_forward.w3.lora_B.default.weight");
        Assert.Equal(LoraFormat.ComfyZImageDit, LoraFormatDetector.Detect(d));
    }

    [Fact]
    public void EachBlockRoot_DetectsOnItsOwn()
    {
        // A LoRA touching only the main stack, or only one refiner, must still be recognized.
        foreach (string root in new[] { "layers", "context_refiner", "noise_refiner" })
        {
            Dictionary<string, SafeTensorDescriptor> d = Descriptors(
                $"diffusion_model.{root}.0.attention.to_v.lora_A.default.weight",
                $"diffusion_model.{root}.0.attention.to_v.lora_B.default.weight");
            Assert.Equal(LoraFormat.ComfyZImageDit, LoraFormatDetector.Detect(d));
        }
    }

    [Fact]
    public void OtherFamilies_KeepTheirOwnFormat()
    {
        // Negative controls: the new arm must not capture any neighbouring family's keys.
        Assert.Equal(LoraFormat.ComfyBflDit, LoraFormatDetector.Detect(Descriptors(
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight",
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight")));
        Assert.Equal(LoraFormat.ComfyBflDit, LoraFormatDetector.Detect(Descriptors(
            "diffusion_model.single_blocks.3.linear1.lora_down.weight",
            "diffusion_model.single_blocks.3.linear1.lora_up.weight")));
        Assert.Equal(LoraFormat.DiffusersWan, LoraFormatDetector.Detect(Descriptors(
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight",
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight")));
        // The `transformer.` passthrough owns its wrapper even over identical Z-Image bodies.
        Assert.Equal(LoraFormat.DiffusersFlux, LoraFormatDetector.Detect(Descriptors(
            "transformer.layers.0.attention.to_q.lora_A.weight",
            "transformer.layers.0.attention.to_q.lora_B.weight")));
        // No wrapper at all stays the bare-root reading.
        Assert.Equal(LoraFormat.DiffusersBareDit, LoraFormatDetector.Detect(Descriptors(
            "layers.0.attention.qkv.lora_A.weight",
            "layers.0.attention.qkv.lora_B.weight")));
    }

    [Theory]
    // Q/K/V keep their split names — FusedProjectionLayouts resolves them into attention.qkv at merge time.
    [InlineData("layers.0.attention.to_q", "layers.0.attention.to_q.weight")]
    [InlineData("layers.29.attention.to_k", "layers.29.attention.to_k.weight")]
    [InlineData("context_refiner.1.attention.to_v", "context_refiner.1.attention.to_v.weight")]
    // The one rename: diffusers' ModuleList spelling → the Tongyi module ZImageBlock.LoadWeights reads.
    [InlineData("layers.0.attention.to_out.0", "layers.0.attention.out.weight")]
    [InlineData("noise_refiner.0.attention.to_out.0", "noise_refiner.0.attention.out.weight")]
    [InlineData("context_refiner.1.attention.to_out.0", "context_refiner.1.attention.out.weight")]
    // SwiGLU projections are already the checkpoint's own names.
    [InlineData("layers.7.feed_forward.w1", "layers.7.feed_forward.w1.weight")]
    [InlineData("noise_refiner.0.feed_forward.w2", "noise_refiner.0.feed_forward.w2.weight")]
    [InlineData("context_refiner.0.feed_forward.w3", "context_refiner.0.feed_forward.w3.weight")]
    // The rename is scoped to the to_out.0 tail: a body already naming an output projection passes through.
    [InlineData("layers.0.attention.o", "layers.0.attention.o.weight")]
    [InlineData("layers.0.attention.out", "layers.0.attention.out.weight")]
    public void MapBodyToCanonical_ProducesCheckpointKeys(string body, string expected)
    {
        Assert.Equal(expected, ZImageLoraMapper.MapBodyToCanonical(body));
    }

    [Fact]
    public void SplitQkvTargets_ResolveToTheFusedCheckpointWeight()
    {
        // The Q/K/V canonical keys do not exist in the checkpoint — ZImageBlock loads one fused
        // `attention.qkv.weight` and carves it Q|K|V by contiguous row thirds (SplitQkv). This pins that the
        // existing fused-projection table maps each split key onto the matching third.
        const string Fused = "layers.0.attention.qkv.weight";
        bool Present(string k) => k == Fused;

        foreach ((string split, int expectedSlice) in new[]
                 {
                     ("layers.0.attention.to_q.weight", 0),
                     ("layers.0.attention.to_k.weight", 1),
                     ("layers.0.attention.to_v.weight", 2),
                 })
        {
            Assert.True(FusedProjectionLayouts.TryResolve(split, Present,
                out string fused, out int sliceIndex, out int sliceCount), split);
            Assert.Equal(Fused, fused);
            Assert.Equal(expectedSlice, sliceIndex);
            Assert.Equal(3, sliceCount);
        }

        // Negative control: the directly-present targets must never be rerouted into a fused sibling.
        Assert.False(FusedProjectionLayouts.TryResolve("layers.0.attention.out.weight", Present, out _, out _, out _));
        Assert.False(FusedProjectionLayouts.TryResolve("layers.0.feed_forward.w1.weight", Present, out _, out _, out _));
    }

    [Fact]
    public void Load_ComfyZImageFile_ParsesEveryModuleToACanonicalKey()
    {
        const int rank = 2, hidden = 4, ffn = 8;
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new();
        foreach (string module in new[] { "attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0" })
        {
            tensors[$"diffusion_model.layers.0.{module}.lora_A.default.weight"] = (DType.F32, [rank, hidden], Fill(rank * hidden, 0.5f));
            tensors[$"diffusion_model.layers.0.{module}.lora_B.default.weight"] = (DType.F32, [hidden, rank], Fill(hidden * rank, 0.25f));
        }
        tensors["diffusion_model.context_refiner.1.feed_forward.w1.lora_A.default.weight"] = (DType.F32, [rank, hidden], Fill(rank * hidden, 0.5f));
        tensors["diffusion_model.context_refiner.1.feed_forward.w1.lora_B.default.weight"] = (DType.F32, [ffn, rank], Fill(ffn * rank, 0.25f));
        tensors["diffusion_model.noise_refiner.0.feed_forward.w2.lora_A.default.weight"] = (DType.F32, [rank, ffn], Fill(rank * ffn, 0.5f));
        tensors["diffusion_model.noise_refiner.0.feed_forward.w2.lora_B.default.weight"] = (DType.F32, [hidden, rank], Fill(hidden * rank, 0.25f));
        string path = CreateSafeTensorsFile(_tempDir, "zimage_turbo_patch", tensors);

        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.ComfyZImageDit, file.Format);
        Assert.Equal(6, file.Layers.Count);
        Assert.All(file.Layers, l => Assert.Equal(LoraTarget.Transformer, l.Target));
        Assert.All(file.Layers, l => Assert.EndsWith(".weight", l.TargetKey, StringComparison.Ordinal));
        // The wrapper must be gone and the ModuleList index renamed away, or the merge matches nothing.
        Assert.DoesNotContain(file.Layers, l => l.TargetKey.StartsWith("diffusion_model.", StringComparison.Ordinal));
        Assert.DoesNotContain(file.Layers, l => l.TargetKey.Contains("to_out", StringComparison.Ordinal));

        Assert.Contains(file.Layers, l => l.TargetKey == "layers.0.attention.to_q.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "layers.0.attention.out.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "context_refiner.1.feed_forward.w1.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "noise_refiner.0.feed_forward.w2.weight");

        // No .alpha companions in this format, so alpha must default to rank (scale = 1).
        LoraLayer q = file.Layers.Single(l => l.TargetKey == "layers.0.attention.to_q.weight");
        Assert.Equal(rank, q.Rank);
        Assert.Equal(rank, q.Alpha);
    }

    [Fact]
    public void LoraStack_MergesSplitKIntoTheFusedQkvSliceOnly()
    {
        // alpha defaults to rank so scale = 1; A = 0.5 [2,4], B = 0.25 [4,2] → delta = 2·(0.25·0.5) = 0.25.
        const int rank = 2, hidden = 4, ffn = 8;
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["diffusion_model.layers.0.attention.to_k.lora_A.default.weight"] = (DType.F32, [rank, hidden], Fill(rank * hidden, 0.5f)),
            ["diffusion_model.layers.0.attention.to_k.lora_B.default.weight"] = (DType.F32, [hidden, rank], Fill(hidden * rank, 0.25f)),
            ["diffusion_model.layers.0.attention.to_out.0.lora_A.default.weight"] = (DType.F32, [rank, hidden], Fill(rank * hidden, 0.5f)),
            ["diffusion_model.layers.0.attention.to_out.0.lora_B.default.weight"] = (DType.F32, [hidden, rank], Fill(hidden * rank, 0.25f)),
            ["diffusion_model.layers.0.feed_forward.w1.lora_A.default.weight"] = (DType.F32, [rank, hidden], Fill(rank * hidden, 0.5f)),
            ["diffusion_model.layers.0.feed_forward.w1.lora_B.default.weight"] = (DType.F32, [ffn, rank], Fill(ffn * rank, 0.25f)),
        };
        string path = CreateSafeTensorsFile(_tempDir, "merge_zimage", tensors);

        // A checkpoint-shaped dict: fused QKV plus the two directly-named projections.
        Tensor qkv = Ones(3 * hidden, hidden);
        Tensor attnOut = Ones(hidden, hidden);
        Tensor w1 = Ones(ffn, hidden);
        Dictionary<string, Tensor> weights = new()
        {
            ["layers.0.attention.qkv.weight"] = qkv,
            ["layers.0.attention.out.weight"] = attnOut,
            ["layers.0.feed_forward.w1.weight"] = w1,
        };

        using CpuBackend backend = new();
        using LoraStack stack = new();
        stack.AddFromPath(path, strength: 1.0f);
        int merged = stack.ApplyTo(weights, LoraTarget.Transformer, backend);
        Assert.Equal(3, merged);

        // K occupies rows [hidden, 2·hidden) of the fused weight: only those may move.
        float* fused = (float*)weights["layers.0.attention.qkv.weight"].DataPointer;
        for (int row = 0; row < 3 * hidden; row++)
        {
            float expected = row >= hidden && row < 2 * hidden ? 1.25f : 1.0f;
            for (int col = 0; col < hidden; col++)
            {
                int i = row * hidden + col;
                Assert.True(MathF.Abs(fused[i] - expected) < 1e-5f,
                    $"qkv[{row},{col}] = {fused[i]}, expected {expected}");
            }
        }

        float* outP = (float*)weights["layers.0.attention.out.weight"].DataPointer;
        for (int i = 0; i < hidden * hidden; i++)
            Assert.True(MathF.Abs(outP[i] - 1.25f) < 1e-5f, $"out[{i}] = {outP[i]}");
        float* w1P = (float*)weights["layers.0.feed_forward.w1.weight"].DataPointer;
        for (int i = 0; i < ffn * hidden; i++)
            Assert.True(MathF.Abs(w1P[i] - 1.25f) < 1e-5f, $"w1[{i}] = {w1P[i]}");

        qkv.Dispose();
        attnOut.Dispose();
        w1.Dispose();
    }

    /// <summary>The real published Comfy-Org Turbo distill patch LoRA — the synthetic tests pin the rules, this pins that the file the world ships actually loads.</summary>
    [Trait("Category", "Integration")]
    [Trait("Category", "RealWeights")]
    [Fact]
    public void RealTurboDistillLora_LoadsWithEveryModuleMapped()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.ZImage.TurboDistillLora)) return;

        using LoraFile file = LoraFile.Load(TestPaths.ZImage.TurboDistillLora);
        Assert.Equal(LoraFormat.ComfyZImageDit, file.Format);
        Assert.Equal(238, file.Layers.Count);
        Assert.All(file.Layers, l => Assert.Equal(LoraTarget.Transformer, l.Target));
        Assert.All(file.Layers, l => Assert.EndsWith(".weight", l.TargetKey, StringComparison.Ordinal));
        Assert.DoesNotContain(file.Layers, l => l.TargetKey.StartsWith("diffusion_model.", StringComparison.Ordinal));
        Assert.DoesNotContain(file.Layers, l => l.TargetKey.Contains("to_out", StringComparison.Ordinal));

        Assert.Contains(file.Layers, l => l.TargetKey == "layers.0.attention.to_q.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "layers.29.feed_forward.w2.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "context_refiner.1.attention.out.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "noise_refiner.0.attention.to_q.weight");
        _output.WriteLine($"{file.Layers.Count} modules, format {file.Format}.");
    }

    private static Tensor Ones(int rows, int cols)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < rows * cols; i++) p[i] = 1.0f;
        return t;
    }

    private static float[] Fill(int count, float value)
    {
        float[] data = new float[count];
        Array.Fill(data, value);
        return data;
    }

    private static string CreateSafeTensorsFile(string dir, string name,
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors)
    {
        using MemoryStream dataStream = new();
        Dictionary<string, (long start, long end)> offsets = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            long start = dataStream.Position;
            foreach (float val in kvp.Value.data)
            {
                dataStream.Write(BitConverter.GetBytes(val), 0, sizeof(float));
            }
            offsets[kvp.Key] = (start, dataStream.Position);
        }
        byte[] dataBlob = dataStream.ToArray();

        Dictionary<string, object> headerDict = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            (long start, long end) = offsets[kvp.Key];
            headerDict[kvp.Key] = new Dictionary<string, object>
            {
                ["dtype"] = kvp.Value.dtype.Name,
                ["shape"] = kvp.Value.shape,
                ["data_offsets"] = new long[] { start, end },
            };
        }

        byte[] headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(headerDict));
        string filePath = Path.Combine(dir, $"{name}.safetensors");
        using FileStream fs = new(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new(fs);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        writer.Write(dataBlob);
        return filePath;
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, true);
            }
        }
        catch
        {
            // best-effort cleanup
        }
    }
}
