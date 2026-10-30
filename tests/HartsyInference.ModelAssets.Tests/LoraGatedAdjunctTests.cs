using System.Text;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Gated LoRA adjuncts: a LoRA carried on a dense weight as a runtime term that a GEMM adds only while its
/// <see cref="AdjunctGate"/> is enabled. The gated-on projection must reproduce a dense merge of the same LoRA, the gated-off
/// projection must be the base weight's own, and the gate's scale must act exactly as a plain strength would.</summary>
public sealed class LoraGatedAdjunctTests : IDisposable
{
    private const string Key = "blocks.0.attn.to_q.weight";
    private const string Module = "transformer.blocks.0.attn.to_q";
    private const int Rows = 8, Cols = 64, Rank = 2;
    private const float DownValue = 0.5f, UpValue = 0.125f;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lora-gated-" + Guid.NewGuid().ToString("N"));

    public LoraGatedAdjunctTests() => Directory.CreateDirectory(_dir);

    [Fact]
    public void Gate_IsOffByDefault_AndAScopeRestoresItsPriorState()
    {
        AdjunctGate gate = new();
        Assert.False(gate.Enabled);

        using (gate.Enable(0.5f))
        {
            Assert.True(gate.Enabled);
            Assert.Equal(0.5f, gate.Scale);
        }

        Assert.False(gate.Enabled);
    }

    [Fact]
    public void Gate_Scopes_NestAndRestoreTheOuterState()
    {
        AdjunctGate gate = new();
        using IDisposable outer = gate.Enable(1.0f);
        using (gate.Enable(0.25f))
        {
            Assert.Equal(0.25f, gate.Scale);
        }
        Assert.True(gate.Enabled);
        Assert.Equal(1.0f, gate.Scale);
    }

    [Fact]
    public void Gate_RefusesANegativeOrNonFiniteScale()
    {
        AdjunctGate gate = new();
        Assert.Throws<ArgumentOutOfRangeException>(() => gate.Enable(-0.5f));
        Assert.Throws<ArgumentOutOfRangeException>(() => gate.Enable(float.NaN));
    }

    [Fact]
    public void GatedOff_TheProjectionIsTheBaseWeightsOwn()
    {
        using Tensor baseWeight = Weight();
        Dictionary<string, Tensor> weights = new() { [Key] = baseWeight };
        AdjunctGate gate = new();
        using LoraFile file = LoraFile.Load(SplitLora("off"));
        using LoraStack stack = new();
        stack.AddGated(file, gate, strength: 1.0f);
        IBackend backend = new CpuBackend();
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);

        Tensor patched = weights[Key];
        Assert.NotSame(baseWeight, patched);
        Assert.False(patched.LowRankAdjunct!.IsActive);

        using Tensor input = Ramp(3, Cols);
        using Tensor actual = Linear(backend, input, patched);
        using Tensor expected = Linear(backend, input, baseWeight);
        AssertEqualExact(expected, actual);
    }

    [Fact]
    public void GatedOn_ReproducesTheDenseMergeOfTheSameLora()
    {
        using Tensor baseWeight = Weight();
        AdjunctGate gate = new();
        using LoraFile file = LoraFile.Load(SplitLora("on"));
        IBackend backend = new CpuBackend();

        using Tensor merged = DenseMerge(backend, file, 0.75f, baseWeight);
        using Tensor input = Ramp(3, Cols);
        using Tensor expected = Linear(backend, input, merged);

        Dictionary<string, Tensor> weights = new() { [Key] = Copy(baseWeight) };
        using LoraStack stack = new();
        stack.AddGated(file, gate, strength: 0.75f);
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);
        using IDisposable scope = gate.Enable(1.0f);
        using Tensor actual = Linear(backend, input, weights[Key]);

        AssertClose(expected, actual);
    }

    [Fact]
    public void GateScale_ActsAsAPlainStrength()
    {
        // The gate's scale multiplies the term's strength: strength 0.75 under a gate at 0.5 must equal a plain merge at 0.375.
        using Tensor baseWeight = Weight();
        AdjunctGate gate = new();
        using LoraFile file = LoraFile.Load(SplitLora("scale"));
        IBackend backend = new CpuBackend();

        using Tensor merged = DenseMerge(backend, file, 0.375f, baseWeight);
        using Tensor input = Ramp(3, Cols);
        using Tensor expected = Linear(backend, input, merged);

        Dictionary<string, Tensor> weights = new() { [Key] = Copy(baseWeight) };
        using LoraStack stack = new();
        stack.AddGated(file, gate, strength: 0.75f);
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);
        using IDisposable scope = gate.Enable(0.5f);
        using Tensor actual = Linear(backend, input, weights[Key]);

        AssertClose(expected, actual);
    }

    [Fact]
    public void GatedWindow_KeepsTheGateThroughARowSlice()
    {
        // A fused projection consumed in parts slices its adjunct; the slice must carry the gate, or a windowed GEMM
        // would add the LoRA while the full GEMM skipped it.
        using Tensor baseWeight = Weight();
        AdjunctGate gate = new();
        using LoraFile file = LoraFile.Load(SplitLora("window"));
        IBackend backend = new CpuBackend();
        Dictionary<string, Tensor> weights = new() { [Key] = Copy(baseWeight) };
        using LoraStack stack = new();
        stack.AddGated(file, gate, strength: 1.0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);
        using Tensor input = Ramp(2, Cols);

        using Tensor off = new(new TensorShape(2, 3), DType.F32);
        backend.LinearWeightRows(off, input, weights[Key], null, 2, 3);
        using Tensor offReference = Linear(backend, input, baseWeight);
        AssertWindow(offReference, off, 2, 3);

        using IDisposable scope = gate.Enable(1.0f);
        using Tensor enabled = new(new TensorShape(2, 3), DType.F32);
        backend.LinearWeightRows(enabled, input, weights[Key], null, 2, 3);
        using Tensor mergedReference = DenseMerge(backend, file, 1.0f, baseWeight);
        using Tensor full = Linear(backend, input, mergedReference);
        AssertWindow(full, enabled, 2, 3);
    }

    [Fact]
    public void MixedGates_OnlyTheEnabledLoraIsAdded()
    {
        // Two gated LoRAs on one weight, one switched off: the projection is active, so the GEMM runs and must add only the
        // enabled term. Summing both would be the failure this pins.
        using Tensor baseWeight = Weight();
        AdjunctGate enabled = new(), disabled = new();
        using LoraFile first = LoraFile.Load(SplitLora("mix_first"));
        using LoraFile second = LoraFile.Load(SplitLora("mix_second"));
        IBackend backend = new CpuBackend();
        using Tensor merged = DenseMerge(backend, first, 1.0f, baseWeight);
        using Tensor input = Ramp(3, Cols);
        using Tensor expected = Linear(backend, input, merged);

        Dictionary<string, Tensor> weights = new() { [Key] = Copy(baseWeight) };
        using LoraStack stack = new();
        stack.AddGated(first, enabled, strength: 1.0f);
        stack.AddGated(second, disabled, strength: 1.0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);
        using IDisposable scope = enabled.Enable(1.0f);
        using Tensor actual = Linear(backend, input, weights[Key]);

        Assert.True(weights[Key].LowRankAdjunct!.IsActive);
        AssertClose(expected, actual);
    }

    [Fact]
    public void GatedAndUngatedOnTheSameKey_BothApply()
    {
        // The ungated LoRA merges into the dense weight while the gated one rides as an adjunct on that merged weight; with the gate on,
        // the projection must equal a dense merge of both.
        using Tensor baseWeight = Weight();
        AdjunctGate gate = new();
        using LoraFile ungated = LoraFile.Load(SplitLora("both_ungated"));
        using LoraFile gatedFile = LoraFile.Load(SplitLora("both_gated"));
        IBackend backend = new CpuBackend();
        Dictionary<string, Tensor> reference = new() { [Key] = Copy(baseWeight) };
        using LoraStack denseStack = new();
        denseStack.Add(ungated, 1.0f);
        denseStack.Add(gatedFile, 1.0f);
        denseStack.ApplyTo(reference, LoraTarget.Transformer, backend);
        using Tensor merged = Copy(reference[Key]);
        using Tensor input = Ramp(3, Cols);
        using Tensor expected = Linear(backend, input, merged);

        Dictionary<string, Tensor> weights = new() { [Key] = Copy(baseWeight) };
        using LoraStack stack = new();
        stack.Add(ungated, 1.0f);
        stack.AddGated(gatedFile, gate, strength: 1.0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);
        using IDisposable scope = gate.Enable(1.0f);
        using Tensor actual = Linear(backend, input, weights[Key]);

        AssertClose(expected, actual);
    }

    [Fact]
    public void RefusedGatedAdapter_LeavesTheWeightsUntouched()
    {
        // The gated target is missing, so the refusal must come before the ungated merge writes into the weights it was given.
        using Tensor baseWeight = Weight();
        Dictionary<string, Tensor> weights = new() { [Key] = baseWeight };
        using LoraFile ungated = LoraFile.Load(SplitLora("untouched_ungated"));
        using LoraFile gated = LoraFile.Load(CreateSafeTensors("untouched_gated", new()
        {
            ["transformer.blocks.9.attn.to_q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
            ["transformer.blocks.9.attn.to_q.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
        }));
        using LoraStack stack = new();
        stack.Add(ungated, 1.0f);
        stack.AddGated(gated, new AdjunctGate(), strength: 1.0f);

        Assert.Throws<HartsyInferenceException>(() => stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));
        Assert.Same(baseWeight, weights[Key]);
    }

    [Fact]
    public void GatedRank4Target_IsRefusedBeforeTheUngatedMergeWrites()
    {
        // A gated adapter needs a 2-D Linear to hang on. The refusal must come before the ungated LoRA merges into the dictionary, or a
        // caller that keeps the dictionary after the throw gets a partly merged checkpoint.
        using Tensor baseWeight = Weight();
        using Tensor conv = new(new TensorShape(Rows, Cols, 1, 1), DType.F32);
        Dictionary<string, Tensor> weights = new() { [Key] = baseWeight, ["blocks.2.attn.to_q.weight"] = conv };
        using LoraFile ungated = LoraFile.Load(SplitLora("rank4_ungated"));
        using LoraFile gated = LoraFile.Load(CreateSafeTensors("rank4_gated", new()
        {
            ["transformer.blocks.2.attn.to_q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
            ["transformer.blocks.2.attn.to_q.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
        }));
        using LoraStack stack = new();
        stack.Add(ungated, 1.0f);
        stack.AddGated(gated, new AdjunctGate(), strength: 1.0f);

        Assert.Throws<NotSupportedException>(() => stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));
        Assert.Same(baseWeight, weights[Key]);
    }

    [Fact]
    public void GatedDoraDelta_IsRefusedBeforeTheUngatedMergeWrites()
    {
        // A DoRA delta rescales the whole weight, so it cannot ride as an additive adjunct. The refusal must come before the ungated sibling merges.
        using Tensor baseWeight = Weight();
        using Tensor sibling = Weight();
        Dictionary<string, Tensor> weights = new() { [Key] = baseWeight, ["blocks.2.attn.to_q.weight"] = sibling };
        using LoraFile ungated = LoraFile.Load(SplitLora("dora_ungated"));
        using LoraFile gated = LoraFile.Load(CreateSafeTensors("dora_gated", new()
        {
            ["transformer.blocks.2.attn.to_q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
            ["transformer.blocks.2.attn.to_q.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
            ["transformer.blocks.2.attn.to_q.dora_scale"] = ([Rows], Filled(Rows, 1.0f)),
        }));
        using LoraStack stack = new();
        stack.Add(ungated, 1.0f);
        stack.AddGated(gated, new AdjunctGate(), strength: 1.0f);

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));
        Assert.Contains("DoRA", error.Message);
        Assert.Same(baseWeight, weights[Key]);
    }

    [Fact]
    public void GatedTarget_MissingFromTheWeights_IsRefusedNotSkipped()
    {
        // A gated adapter is applied whole. A key the checkpoint does not hold means the wrong checkpoint, and skipping
        // it would leave a partial adapter that generates without complaint.
        using LoraFile file = LoraFile.Load(SplitLora("missing"));
        AdjunctGate gate = new();
        using LoraStack stack = new();
        stack.AddGated(file, gate, strength: 1.0f);
        Dictionary<string, Tensor> weights = new();

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));
        Assert.Contains(Key, error.Message);
    }

    [Fact]
    public void GatedDelta_OfAnotherShape_IsRefusedNotSkipped()
    {
        // The same whole-or-nothing rule as a missing target: a delta trained for another build is refused, not skipped.
        string path = CreateSafeTensors("shape", new()
        {
            [$"{Module}.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
            [$"{Module}.lora_B.weight"] = ([Rows + 2, Rank], Filled((Rows + 2) * Rank, UpValue)),
        });
        using LoraFile file = LoraFile.Load(path);
        using Tensor baseWeight = Weight();
        Dictionary<string, Tensor> weights = new() { [Key] = baseWeight };
        using LoraStack stack = new();
        stack.AddGated(file, new AdjunctGate(), strength: 1.0f);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));
        Assert.Contains("different build", error.Message);
    }

    [Fact]
    public void GatedFile_WithAFullWeightDiff_IsRefusedByName()
    {
        // ComfyUI's Wan spelling is the one that carries a full-weight diff; the PEFT spelling skips it with a warning.
        string path = CreateSafeTensors("diff", new()
        {
            ["diffusion_model.blocks.0.self_attn.q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
            ["diffusion_model.blocks.0.self_attn.q.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
            ["diffusion_model.blocks.0.norm3.diff"] = ([Cols], Filled(Cols, 1.0f)),
        });
        using LoraFile file = LoraFile.Load(path);
        using LoraStack stack = new();

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => stack.AddGated(file, new AdjunctGate()));
        Assert.Contains("full-weight", error.Message);
    }

    [Fact]
    public void GatedAdjunct_DoesNotAttachToTheSharedBase()
    {
        using Tensor baseWeight = Weight();
        Dictionary<string, Tensor> weights = new() { [Key] = baseWeight };
        using LoraFile file = LoraFile.Load(SplitLora("shared"));
        using LoraStack stack = new();
        stack.AddGated(file, new AdjunctGate(), strength: 1.0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend());

        Assert.Null(baseWeight.LowRankAdjunct);
    }

    private static Tensor Linear(IBackend backend, Tensor input, Tensor weight)
    {
        Tensor output = new(new TensorShape(input.Shape[0], weight.Shape[0]), DType.F32);
        backend.Linear(output, input, weight, null);
        return output;
    }

    /// <summary>The reference: the same LoRA merged densely at <paramref name="strength"/>. Returns a copy the caller owns,
    /// since the stack that merged it owns the tensor it holds.</summary>
    private static Tensor DenseMerge(IBackend backend, LoraFile file, float strength, Tensor baseWeight)
    {
        Dictionary<string, Tensor> weights = new() { [Key] = Copy(baseWeight) };
        using LoraStack stack = new();
        stack.Add(file, strength);
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);
        return Copy(weights[Key]);
    }

    private static Tensor Weight()
    {
        Tensor w = new(new TensorShape(Rows, Cols), DType.F32);
        Ramp(w);
        return w;
    }

    private static Tensor Copy(Tensor source)
    {
        Tensor copy = new(source.Shape, DType.F32);
        source.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        return copy;
    }

    private static Tensor Ramp(int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        Ramp(t);
        return t;
    }

    private static void Ramp(Tensor t)
    {
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sin(i * 0.37f);
        }
    }

    private static void AssertEqualExact(Tensor expected, Tensor actual)
    {
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
        {
            Assert.Equal(e[i], a[i]);
        }
    }

    /// <summary>Compares two projections to a relative tolerance: the adjunct and the merge sum the same terms in a different
    /// order, so their float32 results differ by rounding, which grows with the magnitude of the value.</summary>
    private static void AssertClose(Tensor expected, Tensor actual)
    {
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
        {
            AssertNear(e[i], a[i]);
        }
    }

    private static void AssertNear(float expected, float actual)
    {
        float tolerance = 1e-5f * MathF.Max(1f, MathF.Abs(expected));
        Assert.True(MathF.Abs(expected - actual) <= tolerance, $"expected {expected}, actual {actual}");
    }

    private static void AssertWindow(Tensor fullOutput, Tensor window, int rows, int columns)
    {
        ReadOnlySpan<float> f = fullOutput.AsReadOnlySpan<float>();
        ReadOnlySpan<float> w = window.AsReadOnlySpan<float>();
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                AssertNear(f[(row * Rows) + 2 + col], w[(row * columns) + col]);
            }
        }
    }

    private string SplitLora(string name) => CreateSafeTensors(name, new()
    {
        [$"{Module}.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
        [$"{Module}.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
    });

    private static float[] Filled(int count, float value)
    {
        float[] data = new float[count];
        Array.Fill(data, value);
        return data;
    }

    private string CreateSafeTensors(string name, Dictionary<string, (long[] Shape, float[] Data)> tensors)
    {
        using MemoryStream dataStream = new();
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
        using FileStream fs = new(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new(fs);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        writer.Write(blob);
        return filePath;
    }

    public void Dispose()
    {
        if (Directory.Exists(_dir))
        {
            Directory.Delete(_dir, true);
        }
    }
}
