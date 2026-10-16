using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
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
        Assert.Equal(1, enumerations);
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

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }
}
