using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers YuE2's conversion on a <b>quantized</b> checkpoint — the official <c>int8_convrot</c> repack's
/// shape, and a block-quantized one.</summary>
/// <remarks><para>Every failure this pins is invisible on the BF16 build. Splitting the fused QKV and gate/up
/// projections drops the per-row dequant scale unless each part narrows it with the rows it takes, and sizes the copy
/// from <c>DType.SizeInBytes</c> — which is 0 for every block quant, so the copy moves nothing and the split is a
/// matrix of zeros. Three entries YuE2 reads on the host are quantized in the published file (the embedding table,
/// <c>llm2vae</c>, the timestep MLP), and raw int8 bytes read as numbers are not a wrong magnitude, they are a
/// different weight.</para>
/// <para>The reference for a split is the fused weight's own decoded rows, not a re-quantization: the invariant is
/// that splitting and decoding commute, which holds whatever the quantizer did.</para></remarks>
public sealed unsafe class Yue2QuantizedConversionTests : IDisposable
{
    private const int Group = 16;
    private const int Vocab = 64;
    private static readonly Yue2Geometry _geometry = new(NumHiddenLayers: 1, NumAttentionHeads: 2,
        NumKeyValueHeads: 1, HeadDim: 16, IntermediateSize: 32);

    private readonly string _tempDir;
    private readonly List<Tensor> _tensors = new();

    public Yue2QuantizedConversionTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"yue2_quant_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        foreach (Tensor tensor in _tensors) tensor.Dispose();
        try { Directory.Delete(_tempDir, recursive: true); }
        catch (IOException) { /* an mmap the OS has not released yet is not a test failure */ }
    }

    private int Hidden => _geometry.NumAttentionHeads * _geometry.HeadDim;

    private Tensor Track(Tensor tensor)
    {
        _tensors.Add(tensor);
        return tensor;
    }

    /// <summary>A deterministic BF16 weight — distinct per key so a mis-routed tensor cannot pass by coincidence.</summary>
    private Tensor Bf16(int seed, params long[] dims)
    {
        Tensor tensor = new Tensor(new TensorShape(dims), DType.BF16);
        Span<ushort> values = tensor.AsSpan<ushort>();
        for (int i = 0; i < values.Length; i++)
            values[i] = TensorCasts.F32ToBf16Bits(0.01f * ((seed * 7 + i * 3) % 41 - 20));
        return Track(tensor);
    }

    /// <summary>An untracked F32 matrix — the caller disposes it, so it must not also be tracked.</summary>
    private static Tensor Dense(long rows, long columns)
    {
        Tensor tensor = new Tensor(new TensorShape(rows, columns), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = 0.01f * ((i * 11) % 37 - 18);
        return tensor;
    }

    private Tensor Int8(int seed, long rows, long columns)
    {
        Tensor tensor = new Tensor(new TensorShape(rows, columns), DType.I8);
        Span<sbyte> values = tensor.AsSpan<sbyte>();
        for (int i = 0; i < values.Length; i++) values[i] = (sbyte)((seed * 13 + i * 29) % 251 - 125);
        return Track(tensor);
    }

    private Tensor RowScale(int seed, long rows)
    {
        Tensor tensor = new Tensor(new TensorShape(rows, 1), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < rows; i++) values[i] = 0.001f * (seed + i + 1);
        return Track(tensor);
    }

    private Tensor Descriptor() => Track(Blob($"{{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": {Group}}}"));

    private static Tensor Blob(string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        Tensor tensor = new Tensor(new TensorShape(bytes.Length), DType.U8);
        bytes.CopyTo(tensor.AsSpan<byte>());
        return tensor;
    }

    /// <summary>A complete YuE2 checkpoint whose Linear weights are int8_convrot with their published companions, matching the key set and per-tensor dtypes of <c>Comfy-Org/YuE2/checkpoints/yue2_3b_int8_convrot.safetensors</c>.</summary>
    private Dictionary<string, Tensor> BuildInt8Checkpoint()
    {
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal)
        {
            ["text_encoders.yue2_tokenizer_json"] = Track(Blob("{\"model\":{\"type\":\"BPE\"}}")),
        };
        int seed = 1;
        foreach (string prefix in (string[])["text_encoders.", "model.diffusion_model."])
        {
            AddInt8(weights, $"{prefix}model.layers.0.self_attn.qkv_proj", seed++, QkvRows, Hidden);
            AddInt8(weights, $"{prefix}model.layers.0.self_attn.o_proj", seed++, Hidden, Hidden);
            AddInt8(weights, $"{prefix}model.layers.0.mlp.gate_up_proj", seed++, 2L * _geometry.IntermediateSize, Hidden);
            AddInt8(weights, $"{prefix}model.layers.0.mlp.down_proj", seed++, Hidden, _geometry.IntermediateSize);
            weights[$"{prefix}model.layers.0.self_attn.q_norm.weight"] = Bf16(seed++, _geometry.HeadDim);
            weights[$"{prefix}model.layers.0.self_attn.k_norm.weight"] = Bf16(seed++, _geometry.HeadDim);
            weights[$"{prefix}model.layers.0.input_layernorm.weight"] = Bf16(seed++, Hidden);
            weights[$"{prefix}model.layers.0.post_attention_layernorm.weight"] = Bf16(seed++, Hidden);
            weights[$"{prefix}model.norm.weight"] = Bf16(seed++, Hidden);
        }
        AddInt8(weights, "text_encoders.model.embed_tokens", seed++, Vocab, Hidden);
        AddInt8(weights, "text_encoders.model.lm_head", seed++, Vocab, Hidden);
        AddInt8(weights, "model.diffusion_model.llm2vae", seed++, 8, Hidden);
        AddInt8(weights, "model.diffusion_model.time_embedder.mlp.0", seed++, Hidden, Group);
        AddInt8(weights, "model.diffusion_model.time_embedder.mlp.2", seed++, Hidden, Hidden);
        weights["model.diffusion_model.vae2llm.weight"] = Bf16(seed++, Hidden, 8);
        weights["model.diffusion_model.vae2llm.bias"] = Bf16(seed++, Hidden);
        weights["model.diffusion_model.llm2vae.bias"] = Bf16(seed++, 8);
        weights["model.diffusion_model.time_embedder.mlp.0.bias"] = Bf16(seed++, Hidden);
        weights["model.diffusion_model.time_embedder.mlp.2.bias"] = Bf16(seed++, Hidden);
        weights["model.diffusion_model.latent_pos_embed.pe"] = Bf16(seed, 4, Hidden);
        return weights;
    }

    private void AddInt8(Dictionary<string, Tensor> weights, string baseKey, int seed, long rows, long columns)
    {
        weights[$"{baseKey}.weight"] = Int8(seed, rows, columns);
        weights[$"{baseKey}.weight_scale"] = RowScale(seed, rows);
        weights[$"{baseKey}.comfy_quant"] = Descriptor();
    }

    private long QkvRows => (long)_geometry.NumAttentionHeads * _geometry.HeadDim
        + 2L * _geometry.NumKeyValueHeads * _geometry.HeadDim;

    private string Write(Dictionary<string, Tensor> weights)
    {
        string path = Path.Combine(_tempDir, "yue2_3b_int8_convrot.safetensors");
        SafeTensorsWriter.Save(path, weights);
        return path;
    }

    /// <summary>Decoded values of an int8_convrot weight, as F32.</summary>
    private static float[] Decode(Tensor weight, Tensor rowScale, int group)
    {
        using Tensor bf16 = Int8ConvRotCodec.DequantToBf16(weight, rowScale, group);
        using Tensor f32 = bf16.CastTo(DType.F32);
        return f32.AsSpan<float>().ToArray();
    }

    [Fact]
    public void Convert_RefusesADictionaryWhoseQuantCompanionsAreStillUnfolded()
    {
        Dictionary<string, Tensor> raw = BuildInt8Checkpoint();
        NotSupportedException error = Assert.Throws<NotSupportedException>(() => Yue2CheckpointConverter.Convert(raw, _geometry));
        Assert.Contains("CheckpointSource.Open", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Convert_NarrowsTheFusedQkvRowScaleToEachSplitsOwnRows()
    {
        Dictionary<string, Tensor> raw = BuildInt8Checkpoint();
        using CheckpointSource source = CheckpointSource.Open(Write(raw));
        using Yue2Weights converted = Yue2CheckpointConverter.Convert(source.Weights, _geometry);

        Tensor fused = source.Weights["text_encoders.model.layers.0.self_attn.qkv_proj.weight"];
        float[] fusedDecoded = Decode(fused, fused.QuantInfo!.RowScale!, Group);
        long q = (long)_geometry.NumAttentionHeads * _geometry.HeadDim;
        long kv = (long)_geometry.NumKeyValueHeads * _geometry.HeadDim;

        long offset = 0;
        foreach ((string name, long rows) in ((string, long)[])[("q_proj", q), ("k_proj", kv), ("v_proj", kv)])
        {
            Tensor split = converted.Ar[$"model.layers.0.self_attn.{name}.weight"];
            Assert.Equal(DType.I8, split.DType);
            Assert.Equal(rows, split.Shape[0]);
            Assert.NotNull(split.QuantInfo);
            Assert.Equal(Group, split.QuantInfo!.ConvRotGroupSize);
            Assert.Equal(rows, split.QuantInfo.RowScale!.ElementCount);

            ReadOnlySpan<float> sliceScale = split.QuantInfo.RowScale.AsSpan<float>();
            ReadOnlySpan<float> fusedScale = fused.QuantInfo.RowScale!.AsSpan<float>();
            for (int row = 0; row < rows; row++)
                Assert.Equal(fusedScale[(int)(offset + row)], sliceScale[row]);

            float[] decoded = Decode(split, split.QuantInfo.RowScale, Group);
            for (int i = 0; i < decoded.Length; i++)
                Assert.Equal(fusedDecoded[(int)(offset * Hidden) + i], decoded[i], 5);
            offset += rows;
        }
    }

    [Fact]
    public void Convert_NarrowsTheFusedGateUpRowScaleToEachHalf()
    {
        Dictionary<string, Tensor> raw = BuildInt8Checkpoint();
        using CheckpointSource source = CheckpointSource.Open(Write(raw));
        using Yue2Weights converted = Yue2CheckpointConverter.Convert(source.Weights, _geometry);

        Tensor fused = source.Weights["model.diffusion_model.model.layers.0.mlp.gate_up_proj.weight"];
        float[] fusedDecoded = Decode(fused, fused.QuantInfo!.RowScale!, Group);
        int half = _geometry.IntermediateSize;

        Tensor gate = converted.Nar["model.layers.0.mlp.gate_proj.weight"];
        Tensor up = converted.Nar["model.layers.0.mlp.up_proj.weight"];
        Assert.Equal(half, gate.QuantInfo!.RowScale!.ElementCount);
        Assert.Equal(half, up.QuantInfo!.RowScale!.ElementCount);

        float[] gateDecoded = Decode(gate, gate.QuantInfo.RowScale, Group);
        float[] upDecoded = Decode(up, up.QuantInfo.RowScale, Group);
        for (int i = 0; i < gateDecoded.Length; i++) Assert.Equal(fusedDecoded[i], gateDecoded[i], 5);
        for (int i = 0; i < upDecoded.Length; i++) Assert.Equal(fusedDecoded[half * Hidden + i], upDecoded[i], 5);
    }

    [Fact]
    public void Convert_DecodesTheEntriesYue2ReadsOnTheHostInsteadOfCastingTheirBytes()
    {
        Dictionary<string, Tensor> raw = BuildInt8Checkpoint();
        using CheckpointSource source = CheckpointSource.Open(Write(raw));
        using Yue2Weights converted = Yue2CheckpointConverter.Convert(source.Weights, _geometry);

        foreach ((string sourceKey, string convertedKey, bool acoustic) in
            (( string, string, bool)[])[
                ("model.diffusion_model.llm2vae.weight", "llm2vae.weight", true),
                ("model.diffusion_model.time_embedder.mlp.0.weight", "time_embedder.mlp.0.weight", true),
                ("text_encoders.model.embed_tokens.weight", "model.embed_tokens.weight", false)])
        {
            Tensor packed = source.Weights[sourceKey];
            Tensor decoded = acoustic ? converted.Nar[convertedKey] : converted.Ar[convertedKey];
            Assert.Equal(DType.F32, decoded.DType);
            Assert.Null(decoded.QuantInfo);
            float[] expected = Decode(packed, packed.QuantInfo!.RowScale!, Group);
            ReadOnlySpan<float> actual = decoded.AsSpan<float>();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
        }
    }

    /// <summary>The output head is a GEMM, so it stays packed — widening it would cost four bytes per parameter for the single largest weight in the model.</summary>
    [Fact]
    public void Convert_LeavesTheOutputHeadPacked()
    {
        Dictionary<string, Tensor> raw = BuildInt8Checkpoint();
        using CheckpointSource source = CheckpointSource.Open(Write(raw));
        using Yue2Weights converted = Yue2CheckpointConverter.Convert(source.Weights, _geometry);

        Tensor head = converted.Ar["lm_head.weight"];
        Assert.Equal(DType.I8, head.DType);
        Assert.Equal(Vocab, head.QuantInfo!.RowScale!.ElementCount);
    }

    /// <summary>A block quant's row length has to divide its block, and the copy has to be sized from the block layout — <c>DType.SizeInBytes</c> is 0 for every one of them, which copies nothing while every shape assertion still passes.</summary>
    [Fact]
    public void Convert_SplitsABlockQuantizedFusedProjectionByteExactly()
    {
        Dictionary<string, Tensor> raw = BuildInt8Checkpoint();
        foreach (string key in raw.Keys.ToList())
        {
            if (key.EndsWith(".weight_scale", StringComparison.Ordinal)
                || key.EndsWith(".comfy_quant", StringComparison.Ordinal))
            {
                raw.Remove(key);
                continue;
            }
            if (raw[key].DType != DType.I8) continue;
            using Tensor dense = Dense(raw[key].Shape[0], raw[key].Shape[1]);
            raw[key] = Track(GgufQuantizer.Quantize(dense, DType.Q8_0));
        }

        using Yue2Weights converted = Yue2CheckpointConverter.Convert(raw, _geometry);

        Tensor fused = raw["text_encoders.model.layers.0.self_attn.qkv_proj.weight"];
        using Tensor fusedDense = GgufDequantizer.Dequantize(fused, DType.F32);
        ReadOnlySpan<float> expected = fusedDense.AsSpan<float>();
        long q = (long)_geometry.NumAttentionHeads * _geometry.HeadDim;

        Tensor qSplit = converted.Ar["model.layers.0.self_attn.q_proj.weight"];
        Assert.Equal(DType.Q8_0, qSplit.DType);
        using Tensor qDense = GgufDequantizer.Dequantize(qSplit, DType.F32);
        ReadOnlySpan<float> actual = qDense.AsSpan<float>();
        Assert.Equal(q * Hidden, actual.Length);
        bool anyNonZero = false;
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 5);
            anyNonZero |= actual[i] != 0f;
        }
        Assert.True(anyNonZero, "a split of zeros would satisfy every other assertion here");
    }

    /// <summary>The tokenizer rides in the checkpoint as a rank-1 U8 tensor; the fold must leave it alone and its length must come from the element count, not a dtype whose size it shares with nothing.</summary>
    [Fact]
    public void Convert_ReadsTheEmbeddedTokenizerThroughTheContainerUntouched()
    {
        Dictionary<string, Tensor> raw = BuildInt8Checkpoint();
        byte[] original = raw["text_encoders.yue2_tokenizer_json"].AsSpan<byte>().ToArray();
        using CheckpointSource source = CheckpointSource.Open(Write(raw));
        using Yue2Weights converted = Yue2CheckpointConverter.Convert(source.Weights, _geometry);
        Assert.Equal(original, converted.TokenizerJson);
    }
}
