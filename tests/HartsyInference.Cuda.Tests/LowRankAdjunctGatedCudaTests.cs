using System.Text;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>A gated LoRA adjunct on the CUDA GEMM: the projection with the gate off is the base weight's own, with the gate on
/// it reproduces the dense merge of the same LoRA, and a windowed GEMM keeps the gate through its row slice.
/// <para>The CPU tests pin the arithmetic. These pin the CUDA dispatch: the gate must be read at the GEMM, not at the
/// load, or a gated-off projection would still add the LoRA on the device.</para></summary>
[Collection("CudaSerial")]
public sealed unsafe class LowRankAdjunctGatedCudaTests : IDisposable
{
    private const string Key = "blocks.0.attn.to_q.weight";
    private const string Module = "transformer.blocks.0.attn.to_q";
    private const int Rows = 256, Cols = 256, Rank = 8, Batch = 32;
    private const float DownValue = 0.02f, UpValue = 0.01f;

    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "cuda-gated-" + Guid.NewGuid().ToString("N"));

    public LowRankAdjunctGatedCudaTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void GatedOff_Linear_IsTheBaseProjection()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Tensor baseWeight = Weight();
        using Tensor input = Activation();
        using LoraFile file = LoraFile.Load(SplitLora("off"));
        using Tensor baseF16 = baseWeight.CastTo(DType.F16);
        Dictionary<string, Tensor> weights = new() { [Key] = baseF16 };
        AdjunctGate gate = new();
        using LoraStack stack = new();
        stack.AddGated(file, gate, strength: 1.0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend());

        // The stack replaces the dictionary entry, so baseF16 is still the unpatched base.
        using Tensor expected = Linear(backend, input, baseF16);
        using Tensor actual = Linear(backend, input, weights[Key]);

        AssertClose(backend, expected, actual, tolerance: 0.02f);
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void GatedOn_Linear_MatchesTheDenseMerge()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Tensor baseWeight = Weight();
        using Tensor input = Activation();
        using LoraFile file = LoraFile.Load(SplitLora("on"));
        using Tensor merged = DenseMerge(file, 1.0f, baseWeight);
        using Tensor mergedF16 = merged.CastTo(DType.F16);
        using Tensor expected = Linear(backend, input, mergedF16);

        using Tensor baseF16 = baseWeight.CastTo(DType.F16);
        Dictionary<string, Tensor> weights = new() { [Key] = baseF16 };
        AdjunctGate gate = new();
        using LoraStack stack = new();
        stack.AddGated(file, gate, strength: 1.0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend());
        using IDisposable scope = gate.Enable(1.0f);
        using Tensor actual = Linear(backend, input, weights[Key]);

        AssertClose(backend, expected, actual, tolerance: 0.02f);
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void GatedOn_WindowedLinear_KeepsTheGateThroughTheSlice()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Tensor baseWeight = Weight();
        using Tensor input = Activation();
        using LoraFile file = LoraFile.Load(SplitLora("window"));
        using Tensor merged = DenseMerge(file, 1.0f, baseWeight);
        using Tensor mergedF16 = merged.CastTo(DType.F16);
        using Tensor full = Linear(backend, input, mergedF16);

        using Tensor baseF16 = baseWeight.CastTo(DType.F16);
        Dictionary<string, Tensor> weights = new() { [Key] = baseF16 };
        AdjunctGate gate = new();
        using LoraStack stack = new();
        stack.AddGated(file, gate, strength: 1.0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend());
        using IDisposable scope = gate.Enable(1.0f);
        const int offset = 64, count = 96;
        using Tensor window = new(new TensorShape(Batch, count), DType.F16);
        backend.LinearWeightRows(window, input, weights[Key], null, offset, count);
        using Tensor expected = new(new TensorShape(Batch, count), DType.F16);
        CopyColumns(backend, full, expected, offset, count);

        AssertClose(backend, expected, window, tolerance: 0.02f);
    }

    private static Tensor Linear(CudaBackend backend, Tensor input, Tensor weight)
    {
        Tensor output = new(new TensorShape(Batch, weight.Shape[0]), DType.F16);
        backend.Linear(output, input, weight, null);
        return output;
    }

    /// <summary>The reference: the same LoRA merged densely on the CPU, which the device path must reproduce.</summary>
    private static Tensor DenseMerge(LoraFile file, float strength, Tensor baseWeight)
    {
        using CpuBackend cpu = new();
        Dictionary<string, Tensor> weights = new() { [Key] = Copy(baseWeight) };
        using LoraStack stack = new();
        stack.Add(file, strength);
        stack.ApplyTo(weights, LoraTarget.Transformer, cpu);
        return Copy(weights[Key]);
    }

    private static void CopyColumns(CudaBackend backend, Tensor full, Tensor window, int offset, int count)
    {
        backend.Sync();
        for (int row = 0; row < Batch; row++)
        {
            for (int col = 0; col < count; col++)
            {
                ((Half*)window.DataPointer)[(row * count) + col] = ((Half*)full.DataPointer)[(row * Rows) + offset + col];
            }
        }
    }

    private static Tensor Weight()
    {
        Tensor w = new(new TensorShape(Rows, Cols), DType.F32);
        Span<float> values = w.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sin(i * 0.37f) * 0.05f;
        }
        return w;
    }

    private static Tensor Activation()
    {
        Tensor x = new(new TensorShape(Batch, Cols), DType.F32);
        Span<float> values = x.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Cos(i * 0.21f);
        }
        return x.CastTo(DType.F16);
    }

    private static Tensor Copy(Tensor source)
    {
        Tensor copy = new(source.Shape, DType.F32);
        source.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        return copy;
    }

    private static void AssertClose(CudaBackend backend, Tensor expected, Tensor actual, float tolerance)
    {
        backend.Sync();
        int count = (int)expected.ElementCount;
        for (int i = 0; i < count; i++)
        {
            float e = Read(expected, i), a = Read(actual, i);
            Assert.True(MathF.Abs(e - a) <= tolerance + (tolerance * MathF.Abs(e)), $"element {i}: expected {e}, got {a}");
        }
    }

    private static float Read(Tensor t, int index) => t.DType == DType.F32
        ? ((float*)t.DataPointer)[index]
        : (float)((Half*)t.DataPointer)[index];

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
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
