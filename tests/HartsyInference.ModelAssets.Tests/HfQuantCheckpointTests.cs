using System.Text.Json;
using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Nf4;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>A Hugging Face checkpoint read as logical weights whatever its storage — plain, block FP8 with an F32 scale grid, GPTQ
/// int4, fused expert stacks — and the streamed disk cache those weights go through when a load converts them.</summary>
public sealed class HfQuantCheckpointTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"hartsy-hfquant-{Guid.NewGuid():N}");

    public HfQuantCheckpointTests() => Directory.CreateDirectory(_dir);

    [Fact]
    public void Fp8BlockDequantize_AppliesEachBlocksScale_OnRaggedEdges()
    {
        const int rows = 130, cols = 200, block = 128;
        using Tensor weight = Fp8(rows, cols, seed: 1);
        using Tensor scales = new(new TensorShape(2, 2), DType.F32);
        float[] grid = [0.5f, 2f, -1.5f, 0.25f];
        grid.CopyTo(scales.AsSpan<float>());
        using Tensor dequantized = Fp8BlockF32Codec.Dequantize(weight, scales, block, block);
        ReadOnlySpan<byte> q = weight.AsReadOnlySpan<byte>();
        ReadOnlySpan<float> got = dequantized.AsReadOnlySpan<float>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                Assert.Equal(CheckpointConvertUtils.E4M3Table[q[r * cols + c]] * grid[(r / block) * 2 + c / block], got[r * cols + c]);
            }
        }
    }

    [Fact]
    public void Index_ReadsEveryStorage_AsLogicalWeights()
    {
        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([WriteCheckpoint()]);
        IReadOnlyDictionary<string, HfQuantWeight> w = checkpoint.Weights;
        Assert.Equal(HfWeightStorage.Plain, w["model.norm.weight"].Storage);
        Assert.Equal(HfWeightStorage.Fp8Block, w["model.layers.0.mlp.experts.0.gate_proj.weight"].Storage);
        Assert.Equal(HfWeightStorage.GptqInt4, w["model.layers.0.self_attn.q_proj.weight"].Storage);
        Assert.Equal(new TensorShape(16, 256), w["model.layers.0.self_attn.q_proj.weight"].Shape);
        Assert.Equal(new TensorShape(4, 32), w["model.layers.1.mlp.experts.1.up_proj.weight"].Shape);
        Assert.Equal(new TensorShape(32, 4), w["model.layers.1.mlp.experts.1.down_proj.weight"].Shape);
        Assert.DoesNotContain("model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv", w.Keys);
        Assert.DoesNotContain("model.layers.1.mlp.experts.gate_up_proj", w.Keys);
    }

    [Fact]
    public void Nf4_IndexesEachWeightOnce_AndDecodesWithinTheCodecBound()
    {
        const int rows = 8, cols = 40;
        const string weightKey = "model.layers.0.mlp.experts.0.down_proj.weight";
        float[] values = new float[rows * cols];
        Random random = new(4);
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        Dictionary<string, Tensor> tensors = Nf4Codec.QuantizeToCompanions(weightKey, values, rows, cols, "float32")
            .ToDictionary(entry => entry.Key, entry => entry.Value, StringComparer.Ordinal);
        Tensor norm = new(new TensorShape(32), DType.F32);
        norm.AsSpan<float>().Fill(0.5f);
        tensors["model.norm.weight"] = norm;
        string path = Path.Combine(_dir, "model.safetensors");
        try
        {
            SafeTensorsWriter.Save(path, tensors);
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }

        (byte[] packed, float[] absmax) = Nf4Codec.Quantize(values);
        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([path]);
        Assert.Equal(2, checkpoint.Weights.Count);
        HfQuantWeight weight = checkpoint.Weights[weightKey];
        Assert.Equal(HfWeightStorage.Nf4, weight.Storage);
        Assert.Equal(new TensorShape(rows, cols), weight.Shape);
        Assert.Equal(HfWeightStorage.Plain, checkpoint.Weights["model.norm.weight"].Storage);

        using Tensor decoded = checkpoint.DequantizeF32(weightKey);
        ReadOnlySpan<float> got = decoded.AsReadOnlySpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            int code = (i & 1) == 0 ? packed[i >> 1] >> 4 : packed[i >> 1] & 0x0F;
            float scale = absmax[i / Nf4Codec.DefaultBlockSize];
            Assert.Equal(Nf4Codec.Nf4Lut[code] * scale, got[i], 6);
            Assert.True(MathF.Abs(got[i] - values[i]) <= scale * Nf4Codec.MaxBlockError + 1e-6f);
        }
    }

    [Fact]
    public void Nf4_BlobShorterThanItsQuantState_FailsAtOpen()
    {
        const int rows = 8, cols = 40;
        const string weightKey = "model.layers.0.mlp.experts.0.down_proj.weight";
        float[] values = new float[rows * cols];
        for (int i = 0; i < values.Length; i++) values[i] = (i % 5) - 2f;
        Dictionary<string, Tensor> tensors = Nf4Codec.QuantizeToCompanions(weightKey, values, rows, cols, "float32")
            .ToDictionary(entry => entry.Key, entry => entry.Value, StringComparer.Ordinal);
        tensors[weightKey].Dispose();
        tensors[weightKey] = new Tensor(new TensorShape(10), DType.U8);
        string path = Path.Combine(_dir, "truncated_nf4.safetensors");
        try
        {
            SafeTensorsWriter.Save(path, tensors);
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => HfQuantCheckpoint.OpenFiles([path]));
        Assert.Contains("holds 10 NF4 bytes", error.Message);
    }

    [Fact]
    public void WriteFromFp8_StreamsEachExpertToNf4_WithinTheBlockBound()
    {
        string source = WriteFp8Source(Path.Combine(_dir, "fp8.safetensors"));
        string output = Path.Combine(_dir, "nf4.safetensors");
        using (HfQuantCheckpoint src = HfQuantCheckpoint.OpenFiles([source]))
        {
            string sha = Nf4Codec.WriteFromFp8(src, output, new Dictionary<string, string> { ["hartsy.quant"] = "nf4" });
            Assert.Equal(64, sha.Length);
        }

        // bitsandbytes' layout: packed nibbles as [n/2, 1], and a quant state its loader reads the type from.
        using (ShardedSafeTensorSet written = ShardedSafeTensorSet.OpenFiles([output]))
        using (Tensor state = written.GetTensor(Expert(0) + ".quant_state.bitsandbytes__nf4"))
        {
            Assert.Equal(new TensorShape(64 * 128 / 2, 1), written.Inventory[Expert(0)].Shape);
            using JsonDocument json = JsonDocument.Parse(state.AsReadOnlySpan<byte>().ToArray());
            Assert.Equal("nf4", json.RootElement.GetProperty("quant_type").GetString());
        }
        using HfQuantCheckpoint dst = HfQuantCheckpoint.OpenFiles([output]);
        using HfQuantCheckpoint reference = HfQuantCheckpoint.OpenFiles([source]);
        Assert.Equal(4, dst.Weights.Count);
        Assert.Equal(HfWeightStorage.Plain, dst.Weights["model.norm.weight"].Storage);
        Assert.Equal(DType.I64, dst.Weights["model.position_ids"].StoredDType);
        using (Tensor storedIds = reference.Borrow("model.position_ids"))
        using (Tensor copiedIds = dst.Borrow("model.position_ids"))
        {
            Assert.Equal(storedIds.AsReadOnlySpan<byte>().ToArray(), copiedIds.AsReadOnlySpan<byte>().ToArray());
        }
        for (int e = 0; e < 2; e++)
        {
            Assert.Equal(HfWeightStorage.Nf4, dst.Weights[Expert(e)].Storage);
            using Tensor want = reference.DequantizeF32(Expert(e));
            using Tensor got = dst.DequantizeF32(Expert(e));
            ReadOnlySpan<float> w = want.AsReadOnlySpan<float>();
            ReadOnlySpan<float> g = got.AsReadOnlySpan<float>();
            Assert.Equal(w.Length, g.Length);
            for (int b = 0; b < w.Length; b += Nf4Codec.DefaultBlockSize)
            {
                float blockMax = 0f;
                int end = Math.Min(w.Length, b + Nf4Codec.DefaultBlockSize);
                for (int i = b; i < end; i++) blockMax = MathF.Max(blockMax, MathF.Abs(w[i]));
                for (int i = b; i < end; i++)
                    Assert.True(MathF.Abs(g[i] - w[i]) <= blockMax * Nf4Codec.MaxBlockError + 1e-6f, $"element {i}: nf4 {g[i]} vs fp8 {w[i]}");
            }
        }
        Assert.False(File.Exists(output + ".partial"));
    }

    [Fact]
    public void WriteFromFp8_RefusesEveryPathToItsSource()
    {
        string real = Directory.CreateDirectory(Path.Combine(_dir, "real")).FullName;
        string source = WriteFp8Source(Path.Combine(real, "fp8.safetensors"));
        List<string> outputs = [source, Path.Combine(real, "..", "real", "fp8.safetensors")];
        // Windows runners do not consistently grant the symbolic-link privilege.
        if (!OperatingSystem.IsWindows())
        {
            outputs.Add(File.CreateSymbolicLink(Path.Combine(_dir, "fp8-link.safetensors"), source).FullName);
            outputs.Add(Path.Combine(Directory.CreateSymbolicLink(Path.Combine(_dir, "alias"), real).FullName, "fp8.safetensors"));
        }
        byte[] before = File.ReadAllBytes(source);
        using (HfQuantCheckpoint src = HfQuantCheckpoint.OpenFiles([source]))
        {
            foreach (string output in outputs)
                Assert.Throws<ArgumentException>(() => Nf4Codec.WriteFromFp8(src, output, new Dictionary<string, string>()));
        }
        Assert.Equal(before, File.ReadAllBytes(source));

        // A source the output's .partial file would land on.
        string partialNamed = WriteFp8Source(Path.Combine(_dir, "nf4.safetensors.partial"));
        byte[] partialBefore = File.ReadAllBytes(partialNamed);
        using (HfQuantCheckpoint src = HfQuantCheckpoint.OpenFiles([partialNamed]))
        {
            string output = Path.Combine(_dir, "nf4.safetensors");
            Assert.Throws<ArgumentException>(() => Nf4Codec.WriteFromFp8(src, output, new Dictionary<string, string>()));
        }
        Assert.Equal(partialBefore, File.ReadAllBytes(partialNamed));
    }

    [Fact]
    public void WriteFromFp8_Cancelled_KeepsThePreviousFileAndLeavesNoPartial()
    {
        string source = WriteFp8Source(Path.Combine(_dir, "fp8.safetensors"));
        string output = Path.Combine(_dir, "nf4.safetensors");
        File.WriteAllText(output, "previous checkpoint");
        using CancellationTokenSource stop = new();
        stop.Cancel();
        using HfQuantCheckpoint src = HfQuantCheckpoint.OpenFiles([source]);
        Assert.Throws<OperationCanceledException>(() =>
            Nf4Codec.WriteFromFp8(src, output, new Dictionary<string, string>(), cancellationToken: stop.Token));
        Assert.Equal("previous checkpoint", File.ReadAllText(output));
        Assert.False(File.Exists(output + ".partial"));
    }

    [Fact]
    public void WriteFromFp8_LeavesAPartialFileItCouldNotOpen()
    {
        string source = WriteFp8Source(Path.Combine(_dir, "fp8.safetensors"));
        string output = Path.Combine(_dir, "nf4.safetensors");
        using HfQuantCheckpoint src = HfQuantCheckpoint.OpenFiles([source]);
        // A second run writing the same output still holds its partial file.
        using FileStream held = new(output + ".partial", FileMode.Create, FileAccess.Write, FileShare.None);
        held.Write("in progress"u8);
        held.Flush();
        Assert.ThrowsAny<IOException>(() => Nf4Codec.WriteFromFp8(src, output, new Dictionary<string, string>()));
        Assert.Equal(held.Length, new FileInfo(output + ".partial").Length);
    }

    [Fact]
    public void WriteFromFp8_FailingMidStream_KeepsThePreviousFileAndLeavesNoPartial()
    {
        // 500 quantized weights: the progress callback fires after the 500th is written, with the output still open.
        string source = Path.Combine(_dir, "fp8_many.safetensors");
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        for (int i = 0; i < 500; i++)
        {
            string name = $"model.layers.{i}.mlp.experts.0.down_proj.weight";
            tensors[name] = Fp8(8, 8, seed: i);
            Tensor scale = new(new TensorShape(1, 1), DType.F32);
            scale.AsSpan<float>()[0] = 1f;
            tensors[name + "_scale_inv"] = scale;
        }
        try
        {
            SafeTensorsWriter.Save(source, tensors);
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }

        string output = Path.Combine(_dir, "nf4_previous.safetensors");
        File.WriteAllText(output, "previous checkpoint");
        using HfQuantCheckpoint src = HfQuantCheckpoint.OpenFiles([source]);
        InvalidOperationException injected = Assert.Throws<InvalidOperationException>(() =>
            Nf4Codec.WriteFromFp8(src, output, new Dictionary<string, string>(), _ => throw new InvalidOperationException("injected")));

        Assert.Equal("injected", injected.Message);
        Assert.Equal("previous checkpoint", File.ReadAllText(output));
        Assert.False(File.Exists(output + ".partial"));
    }

    [Fact]
    public void Nf4Expert_TargetsQ4_0AtNative_AndKeepsTheOtherPrecisions()
    {
        HfQuantWeight expert = new("model.layers.0.mlp.experts.0.down_proj.weight", new TensorShape(2048, 512),
            HfWeightStorage.Nf4, DType.U8, "model.layers.0.mlp.experts.0.down_proj.weight");
        Assert.Equal(DType.Q4_0, FishAudioDrama3Weights.TargetDType(expert, FishAudioDrama3Precision.Native));
        Assert.Equal(DType.Q8_0, FishAudioDrama3Weights.TargetDType(expert, FishAudioDrama3Precision.Q8_0));
        // A row that is not whole Q4_0 blocks loads as BF16 rather than reaching the quantizer.
        HfQuantWeight ragged = expert with { Shape = new TensorShape(8, 40) };
        Assert.Equal(DType.BF16, FishAudioDrama3Weights.TargetDType(ragged, FishAudioDrama3Precision.Native));
    }

    [Fact]
    public void Produce_RefusesABlockQuantWhoseBlocksWouldSpanRows()
    {
        const string weightKey = "model.layers.0.mlp.experts.0.down_proj.weight";
        string path = Path.Combine(_dir, "nf4.safetensors");
        Dictionary<string, Tensor> tensors = Nf4Codec.QuantizeToCompanions(weightKey, new float[8 * 40], 8, 40, "float32")
            .ToDictionary(entry => entry.Key, entry => entry.Value, StringComparer.Ordinal);
        try
        {
            SafeTensorsWriter.Save(path, tensors);
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }

        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([path]);
        ArgumentException error = Assert.Throws<ArgumentException>(() => checkpoint.Produce(weightKey, DType.Q4_0));
        Assert.Contains("rows of 40 values", error.Message);
    }

    [Fact]
    public void Nf4_DoubleQuantWithoutItsNestedCodebook_FailsAtOpen()
    {
        const string weightKey = "model.layers.0.mlp.experts.0.down_proj.weight";
        Dictionary<string, Tensor> tensors = Nf4Codec.QuantizeToCompanions(weightKey, new float[8 * 64], 8, 64, "float32")
            .ToDictionary(entry => entry.Key, entry => entry.Value, StringComparer.Ordinal);
        // A U8 absmax is double quant: give it the nested absmax but not the nested codebook.
        long blocks = tensors[weightKey + ".absmax"].Shape.ElementCount;
        tensors[weightKey + ".absmax"].Dispose();
        tensors[weightKey + ".absmax"] = new Tensor(new TensorShape(blocks), DType.U8);
        tensors[weightKey + ".nested_absmax"] = new Tensor(new TensorShape(1), DType.F32);
        string path = Path.Combine(_dir, "double_quant_nf4.safetensors");
        try
        {
            SafeTensorsWriter.Save(path, tensors);
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => HfQuantCheckpoint.OpenFiles([path]));
        Assert.Contains($"'{weightKey}.nested_quant_map'", error.Message);
    }

    [Fact]
    public void Produce_TakesTheExactRoute_ForEachStorage()
    {
        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([WriteCheckpoint()]);
        // GPTQ to Q4_0 is the bit-exact repack.
        using (Tensor q4 = checkpoint.Produce("model.layers.0.self_attn.q_proj.weight", DType.Q4_0))
        using (Tensor viaDequant = checkpoint.DequantizeF32("model.layers.0.self_attn.q_proj.weight"))
        {
            Assert.Equal(DType.Q4_0, q4.DType);
            using Tensor decoded = GgufDequantizer.Dequantize(q4, DType.F32);
            Assert.Equal(viaDequant.AsReadOnlySpan<float>().ToArray(), decoded.AsReadOnlySpan<float>().ToArray());
        }
        // An expert of a fused stack is its own rows of the stack: gate first, then up.
        using (Tensor up = checkpoint.Borrow("model.layers.1.mlp.experts.1.up_proj.weight"))
        {
            ReadOnlySpan<float> values = up.AsReadOnlySpan<float>();
            Assert.Equal(FusedValue(1, 4, 0), values[0]);
            Assert.Equal(FusedValue(1, 7, 31), values[^1]);
        }
        // FP8 to Q8_0 reports the error the requantization adds.
        GgufQuantError error = new();
        using (Tensor q8 = checkpoint.Produce("model.layers.0.mlp.experts.0.gate_proj.weight", DType.Q8_0, error))
        {
            Assert.Equal(DType.Q8_0, q8.DType);
        }
        Assert.Equal(64L * 128, error.Count);
        Assert.True(error.RelativeRms is > 0 and < 0.02, $"relative RMS {error.RelativeRms}");
    }

    [Fact]
    public void QuantCache_ProducesEachTensorOnlyWhenWritten_AndReadsBackEngineShapes()
    {
        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([WriteCheckpoint()]);
        List<string> produced = [];
        List<GgufCacheEntry> entries = [];
        foreach ((string name, HfQuantWeight weight) in checkpoint.Weights.OrderBy(static w => w.Key, StringComparer.Ordinal))
        {
            DType target = weight.Storage == HfWeightStorage.GptqInt4 ? DType.Q4_0 : weight.Storage == HfWeightStorage.Fp8Block ? DType.Q8_0 : weight.StoredDType;
            entries.Add(new GgufCacheEntry(name, weight.Shape, target, () =>
            {
                produced.Add(name);
                return checkpoint.Produce(name, target);
            }));
        }
        string cache = Path.Combine(_dir, "cache.gguf");
        int enumerations = 0;
        using (GgufLoader loader = GgufQuantCache.Open(cache, () => { enumerations++; return entries; }, "test", "[test]"))
        {
            Assert.Equal(entries.Select(static e => e.Name), produced);
            Dictionary<string, Tensor> weights = GgufQuantCache.ReadBack(loader, entries);
            Assert.Equal(new TensorShape(16, 256), weights["model.layers.0.self_attn.q_proj.weight"].Shape);
            Assert.Equal(DType.Q4_0, weights["model.layers.0.self_attn.q_proj.weight"].DType);
            Assert.Equal(new TensorShape(32, 4), weights["model.layers.1.mlp.experts.1.down_proj.weight"].Shape);
            using Tensor up = checkpoint.Borrow("model.layers.1.mlp.experts.1.up_proj.weight");
            Assert.Equal(up.AsReadOnlySpan<float>().ToArray(), weights["model.layers.1.mlp.experts.1.up_proj.weight"].AsReadOnlySpan<float>().ToArray());
        }
        using (GgufLoader reopened = GgufQuantCache.Open(cache, () => { enumerations++; return entries; }, "test", "[test]"))
        {
            Assert.Equal(entries.Count, reopened.Descriptors.Count);
        }
        // The reopen reads the entries to check the cache against them, but produces none of them again.
        Assert.Equal(2, enumerations);
        Assert.Equal(entries.Count, produced.Count);
    }

    // A plain norm, a block-FP8 expert, a GPTQ attention projection and a fused BF16-style (F32 here) expert stack.
    private string WriteCheckpoint()
    {
        string path = Path.Combine(_dir, "model.safetensors");
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        try
        {
            Tensor norm = new(new TensorShape(32), DType.F32);
            norm.AsSpan<float>().Fill(0.5f);
            tensors["model.norm.weight"] = norm;
            tensors["model.layers.0.mlp.experts.0.gate_proj.weight"] = Fp8(64, 128, seed: 2);
            Tensor scale = new(new TensorShape(1, 1), DType.F32);
            scale.AsSpan<float>()[0] = 0.02f;
            tensors["model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"] = scale;
            Random random = new(9);
            Tensor qweight = new(new TensorShape(256 / 8, 16), DType.I32);
            foreach (ref int word in qweight.AsSpan<int>()) word = random.Next();
            Tensor scales = new(new TensorShape(2, 16), DType.F16);
            foreach (ref Half s in scales.AsSpan<Half>()) s = (Half)(0.01f + random.NextSingle() * 0.02f);
            Tensor qzeros = new(new TensorShape(2, 2), DType.I32);
            qzeros.AsSpan<uint>().Fill(0x77777777u);
            tensors["model.layers.0.self_attn.q_proj.qweight"] = qweight;
            tensors["model.layers.0.self_attn.q_proj.scales"] = scales;
            tensors["model.layers.0.self_attn.q_proj.qzeros"] = qzeros;
            Tensor gateUp = new(new TensorShape(2, 8, 32), DType.F32);
            Span<float> gu = gateUp.AsSpan<float>();
            for (int e = 0; e < 2; e++)
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 32; c++) gu[(e * 8 + r) * 32 + c] = FusedValue(e, r, c);
            tensors["model.layers.1.mlp.experts.gate_up_proj"] = gateUp;
            tensors["model.layers.1.mlp.experts.down_proj"] = new Tensor(new TensorShape(2, 32, 4), DType.F32);
            SafeTensorsWriter.Save(path, tensors);
            return path;
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    private static float FusedValue(int expert, int row, int col) => expert * 1000 + row * 40 + col;

    private static Tensor Fp8(int rows, int cols, int seed)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F8E4M3);
        Random random = new(seed);
        Span<byte> bytes = t.AsSpan<byte>();
        // Avoid the two NaN encodings so the comparison stays exact.
        for (int i = 0; i < bytes.Length; i++)
        {
            byte b;
            do b = (byte)random.Next(256); while ((b & 0x7F) == 0x7F);
            bytes[i] = b;
        }
        return t;
    }

    private static string Expert(int index) => $"model.layers.0.mlp.experts.{index}.down_proj.weight";

    // Two block-FP8 experts, a plain F32 norm and a plain I64 buffer.
    private static string WriteFp8Source(string path)
    {
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        try
        {
            for (int e = 0; e < 2; e++)
            {
                tensors[Expert(e)] = Fp8(64, 128, seed: 6 + e);
                Tensor scale = new(new TensorShape(1, 1), DType.F32);
                scale.AsSpan<float>()[0] = 0.03f;
                tensors[Expert(e) + "_scale_inv"] = scale;
            }
            Tensor norm = new(new TensorShape(16), DType.F32);
            norm.AsSpan<float>().Fill(1.5f);
            tensors["model.norm.weight"] = norm;
            Tensor ids = new(new TensorShape(5), DType.I64);
            Span<long> idValues = ids.AsSpan<long>();
            for (int i = 0; i < idValues.Length; i++) idValues[i] = (long)i << 40;
            tensors["model.position_ids"] = ids;
            SafeTensorsWriter.Save(path, tensors);
            return path;
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }
}
