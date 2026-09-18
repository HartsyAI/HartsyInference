using System.Text;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins the runtime-adjunct path a LoRA takes into a block-quantized base. Q4_0/Q5_0/Q5_1 have no quantizer,
/// so a merged result cannot be written back to them; the delta rides on the weight instead and every GEMM adds it.
/// <para>The ownership assertion here is the one that matters most. The base <see cref="Tensor"/> is held by the
/// converted-weight cache, by resident models and by the identity-keyed device cache, so writing the adjunct onto it
/// would carry one request's LoRA into the next request that reuses the entry — with nothing in the recipe cache key
/// to tell them apart, and no error at any point.</para></summary>
public sealed unsafe class LoraQuantizedAdjunctTests : IDisposable
{
    private const int Rows = 8, Cols = 64, Rank = 2;
    private const float DownValue = 0.5f, UpValue = 0.125f;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lora-adjunct-" + Guid.NewGuid().ToString("N"));

    public LoraQuantizedAdjunctTests() => Directory.CreateDirectory(_dir);

    [Fact]
    public void QuantizedBase_GetsAnAdjunctOnACopy_NeverOnTheSharedTensor()
    {
        using Tensor quantized = QuantizedWeight(Rows, Cols);
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn.to_q.weight"] = quantized };

        LoraStack stack = new LoraStack();
        stack.AddFromPath(SplitLora("own", "transformer.blocks.0.attn.to_q", Rows), strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));

        Tensor patched = weights["blocks.0.attn.to_q.weight"];
        Assert.NotSame(quantized, patched);
        Assert.Null(quantized.LowRankAdjunct);
        Assert.NotNull(patched.LowRankAdjunct);
        // Aliased, not copied: a block-quantized weight staying packed is the whole reason this path exists.
        Assert.True(quantized.DataPointer == patched.DataPointer, "The adjunct view copied the packed bytes.");
        Assert.Equal(DType.Q8_0, patched.DType);

        // Disposing the stack frees the view and the factors; it must not reach the base tensor's bytes.
        using Tensor before = GgufDequantizer.Dequantize(quantized, DType.F32);
        float[] snapshot = [.. before.AsReadOnlySpan<float>()];
        stack.Dispose();
        using Tensor after = GgufDequantizer.Dequantize(quantized, DType.F32);
        Assert.Equal(snapshot, after.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void QuantizedBase_AdjunctReproducesTheDenseMerge()
    {
        // The reference is the merge this path cannot take: dequantize the SAME packed bytes, add the delta, and run
        // the projection against that. Anything the adjunct gets wrong — factor order, the alpha/rank scale, the
        // strength — shows up here and nowhere on the dense path.
        using Tensor quantized = QuantizedWeight(Rows, Cols);
        using Tensor dequantized = GgufDequantizer.Dequantize(quantized, DType.F32);
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn.to_q.weight"] = quantized };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(SplitLora("math", "transformer.blocks.0.attn.to_q", Rows), strength: 0.75f);
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);

        // The CPU backend holds no quantized GEMM (SupportsResidentQuant is false, so QuantizedWeightPolicy widens
        // before a merge ever runs), which is why the terms the QUANTIZED routing built are re-hung on the
        // dequantized bytes here — the arithmetic under test is the terms, not the codec.
        using Tensor runnable = dequantized.WithLowRankAdjunct(
            weights["blocks.0.attn.to_q.weight"].LowRankAdjunct!);
        using Tensor input = Ramp(3, Cols);
        using Tensor actual = new Tensor(new TensorShape(3, Rows), DType.F32);
        backend.Linear(actual, input, runnable, null);

        using Tensor mergedWeight = MergeReference(dequantized, 0.75f);
        using Tensor expected = new Tensor(new TensorShape(3, Rows), DType.F32);
        backend.Linear(expected, input, mergedWeight, null);

        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(e[i], a[i], 3);
        }
    }

    [Fact]
    public void QuantizedBase_WindowedGemmSeesOnlyItsOwnRows()
    {
        // LinearWeightRows is how MiniMax-H3 consumes a fused projection, so the adjunct has to narrow with the
        // window. A term that stayed whole would add every row's delta to a slice that holds a few of them.
        using Tensor quantized = QuantizedWeight(Rows, Cols);
        using Tensor dequantized = GgufDequantizer.Dequantize(quantized, DType.F32);
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn.to_q.weight"] = quantized };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(SplitLora("window", "transformer.blocks.0.attn.to_q", Rows), strength: 1.0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);

        using Tensor runnable = dequantized.WithLowRankAdjunct(
            weights["blocks.0.attn.to_q.weight"].LowRankAdjunct!);
        using Tensor input = Ramp(2, Cols);
        using Tensor windowed = new Tensor(new TensorShape(2, 3), DType.F32);
        backend.LinearWeightRows(windowed, input, runnable, null, 2, 3);

        using Tensor mergedWeight = MergeReference(dequantized, 1.0f);
        using Tensor full = new Tensor(new TensorShape(2, Rows), DType.F32);
        backend.Linear(full, input, mergedWeight, null);

        ReadOnlySpan<float> w = windowed.AsReadOnlySpan<float>();
        ReadOnlySpan<float> f = full.AsReadOnlySpan<float>();
        for (int row = 0; row < 2; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                Assert.Equal(f[(row * Rows) + 2 + col], w[(row * 3) + col], 3);
            }
        }
    }

    [Fact]
    public void QuantizedFusedProjection_PadsTheSliceToTheFullRowCount()
    {
        // A split-form LoRA landing on a fused qkv weight covers one third of its rows. The up matrix is zero-padded
        // to every row so LinearWeightRows can narrow it with the window; carrying a row offset instead would make
        // the term unusable by H3's chunked projections, which is the build this path exists for.
        using Tensor quantized = QuantizedWeight(Rows * 3, Cols);
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn.qkv_proj.weight"] = quantized };

        using LoraStack stack = new LoraStack();
        stack.AddFromPath(SplitLora("fused", "transformer.blocks.0.attn.to_k", Rows), strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));

        LowRankAdjunct adjunct = weights["blocks.0.attn.qkv_proj.weight"].LowRankAdjunct!;
        LowRankAdjunctTerm term = Assert.Single(adjunct.Terms);
        Assert.Equal(Rows * 3L, term.Up!.Shape[0]);
        ReadOnlySpan<float> up = term.Up.AsReadOnlySpan<float>();
        // k is slice 1, so rows [0, 8) and [16, 24) must stay zero and rows [8, 16) must carry the factor.
        for (int r = 0; r < Rows * 3; r++)
        {
            bool inSlice = r >= Rows && r < Rows * 2;
            for (int c = 0; c < Rank; c++)
            {
                if (inSlice) Assert.Equal(UpValue, up[(r * Rank) + c], 6);
                else Assert.Equal(0f, up[(r * Rank) + c], 6);
            }
        }
    }

    [Fact]
    public void StackedLorasOnOneQuantizedWeight_BecomeSeparateTerms()
    {
        using Tensor quantized = QuantizedWeight(Rows, Cols);
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn.to_q.weight"] = quantized };

        using LoraStack stack = new LoraStack();
        stack.AddFromPath(SplitLora("first", "transformer.blocks.0.attn.to_q", Rows), strength: 1.0f);
        stack.AddFromPath(SplitLora("second", "transformer.blocks.0.attn.to_q", Rows), strength: 0.5f);
        // One weight patched once, not twice: a second attach would drop the first LoRA's terms.
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));

        LowRankAdjunct adjunct = weights["blocks.0.attn.to_q.weight"].LowRankAdjunct!;
        Assert.Equal(2, adjunct.Terms.Count);
        Assert.Equal(2f * adjunct.Terms[1].Scale, adjunct.Terms[0].Scale, 5);
    }

    [Fact]
    public void LoHaOnAQuantizedBase_MaterializesOneRankFullTerm()
    {
        using Tensor quantized = QuantizedWeight(Rows, Cols);
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn.to_q.weight"] = quantized };

        using LoraStack stack = new LoraStack();
        stack.AddFromPath(LoHaLora("loha", "transformer.blocks.0.attn.to_q"), strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));

        LowRankAdjunctTerm term = Assert.Single(weights["blocks.0.attn.to_q.weight"].LowRankAdjunct!.Terms);
        Assert.Null(term.Up);
        Assert.Equal(Rows, term.Down.Shape[0]);
        Assert.Equal(Cols, term.Down.Shape[1]);
    }

    [Fact]
    public void DoraOnAQuantizedBase_RefusesByName()
    {
        using Tensor quantized = QuantizedWeight(Rows, Cols);
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn.to_q.weight"] = quantized };

        using LoraStack stack = new LoraStack();
        stack.AddFromPath(DoraLora("dora", "transformer.blocks.0.attn.to_q", Rows), strength: 1.0f);

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));
        Assert.Contains("blocks.0.attn.to_q.weight", error.Message);
        Assert.Contains("DoRA", error.Message);
    }

    [Fact]
    public void TextEncoderTargets_TakeTheTencStrength()
    {
        // SwarmUI's strength_clip is a separate number; the diffusion body keeps strength_model. One file names both
        // arms, so the two adjuncts differ only by which strength their target selected.
        string path = CreateSafeTensors("tenc", new()
        {
            ["transformer.blocks.0.attn.to_q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
            ["transformer.blocks.0.attn.to_q.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
            ["text_encoder.blocks.0.attn.to_q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
            ["text_encoder.blocks.0.attn.to_q.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
        });
        using Tensor ditWeight = QuantizedWeight(Rows, Cols);
        using Tensor clipWeight = QuantizedWeight(Rows, Cols);
        Dictionary<string, Tensor> dit = new() { ["blocks.0.attn.to_q.weight"] = ditWeight };
        Dictionary<string, Tensor> clip = new() { ["blocks.0.attn.to_q.weight"] = clipWeight };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(path, strength: 1.0f, tencStrength: 0.25f);
        stack.ApplyTo(dit, LoraTarget.Transformer, backend);
        stack.ApplyTo(clip, LoraTarget.ClipL, backend);

        float ditScale = dit["blocks.0.attn.to_q.weight"].LowRankAdjunct!.Terms[0].Scale;
        float clipScale = clip["blocks.0.attn.to_q.weight"].LowRankAdjunct!.Terms[0].Scale;
        Assert.Equal(ditScale * 0.25f, clipScale, 6);
    }

    private static Tensor QuantizedWeight(int rows, int cols)
    {
        using Tensor dense = Ramp(rows, cols);
        return GgufQuantizer.Quantize(dense, DType.Q8_0);
    }

    private static Tensor Ramp(int rows, int cols)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sin(i * 0.37f);
        }
        return t;
    }

    /// <summary>The dense weight the adjunct must reproduce: <c>W + strength · (alpha/rank) · B @ A</c>. The file
    /// carries no alpha, so the scale is 1.0 and every element of <c>B @ A</c> is <c>rank · up · down</c>.</summary>
    private static Tensor MergeReference(Tensor dequantized, float strength)
    {
        Tensor merged = dequantized.CastTo(DType.F32);
        Span<float> w = merged.AsSpan<float>();
        float delta = strength * Rank * DownValue * UpValue;
        for (int i = 0; i < w.Length; i++)
        {
            w[i] += delta;
        }
        return merged;
    }

    private string SplitLora(string name, string module, int outRows) => CreateSafeTensors(name, new()
    {
        [$"{module}.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
        [$"{module}.lora_B.weight"] = ([outRows, Rank], Filled(outRows * Rank, UpValue)),
    });

    private string DoraLora(string name, string module, int outRows) => CreateSafeTensors(name, new()
    {
        [$"{module}.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
        [$"{module}.lora_B.weight"] = ([outRows, Rank], Filled(outRows * Rank, UpValue)),
        [$"{module}.dora_scale"] = ([outRows, 1], Filled(outRows, 1.0f)),
    });

    private string LoHaLora(string name, string module) => CreateSafeTensors(name, new()
    {
        [$"{module}.hada_w1_a"] = ([Rows, Rank], Filled(Rows * Rank, 0.25f)),
        [$"{module}.hada_w1_b"] = ([Rank, Cols], Filled(Rank * Cols, 0.5f)),
        [$"{module}.hada_w2_a"] = ([Rows, Rank], Filled(Rows * Rank, 0.25f)),
        [$"{module}.hada_w2_b"] = ([Rank, Cols], Filled(Rank * Cols, 0.5f)),
    });

    private static float[] Filled(int count, float value)
    {
        float[] data = new float[count];
        Array.Fill(data, value);
        return data;
    }

    private string CreateSafeTensors(string name, Dictionary<string, (long[] Shape, float[] Data)> tensors)
    {
        using MemoryStream dataStream = new MemoryStream();
        Dictionary<string, (long Start, long End)> offsets = [];
        foreach (KeyValuePair<string, (long[] Shape, float[] Data)> kvp in tensors)
        {
            long start = dataStream.Position;
            foreach (float value in kvp.Value.Data)
            {
                dataStream.Write(BitConverter.GetBytes(value), 0, 4);
            }
            offsets[kvp.Key] = (start, dataStream.Position);
        }
        byte[] blob = dataStream.ToArray();

        Dictionary<string, object> header = [];
        foreach (KeyValuePair<string, (long[] Shape, float[] Data)> kvp in tensors)
        {
            (long start, long end) = offsets[kvp.Key];
            header[kvp.Key] = new Dictionary<string, object>
            {
                ["dtype"] = DType.F32.Name,
                ["shape"] = kvp.Value.Shape,
                ["data_offsets"] = new long[] { start, end },
            };
        }
        byte[] headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        string filePath = Path.Combine(_dir, $"{name}.safetensors");
        using FileStream fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new BinaryWriter(fs);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        writer.Write(blob);
        return filePath;
    }

    public void Dispose()
    {
        if (Directory.Exists(_dir))
        {
            Directory.Delete(_dir, recursive: true);
        }
    }
}
