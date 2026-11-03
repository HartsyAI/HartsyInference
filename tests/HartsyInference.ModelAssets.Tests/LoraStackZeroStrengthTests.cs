using System.Text;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins what <see cref="LoraStack.ApplyTo"/> does with zero-strength layers and how it reports its counts. A zero-strength
/// layer is bypassed, so the base tensor object is left as it was, and each count measures its own unit.</summary>
public sealed class LoraStackZeroStrengthTests : IDisposable
{
    private const int Rows = 8, Cols = 4, Rank = 2;
    private const string BaseKey = "blocks.0.to_q.weight";

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lora-zero-" + Guid.NewGuid().ToString("N"));

    public LoraStackZeroStrengthTests() => Directory.CreateDirectory(_dir);

    [Fact]
    public void ZeroStrength_LeavesTheBaseTensorObjectUntouched()
    {
        string loraPath = WriteLora("zero", "blocks.0");
        Tensor baseTensor = Zeros(Rows, Cols);
        Dictionary<string, Tensor> weights = new() { [BaseKey] = baseTensor };

        using LoraStack stack = new LoraStack();
        stack.AddFromPath(loraPath, strength: 0f);
        int merged = stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend());

        // Identity, not value equality: a bypassed layer never reaches the dequant-merge-requant round trip.
        Assert.Equal(0, merged);
        Assert.Same(baseTensor, weights[BaseKey]);
        Assert.Equal(new LoraStack.ApplyReport(0, 0, 0, 0, 0, 1), stack.Report);
    }

    [Fact]
    public void ZeroStrength_OnAnAbsentTarget_IsNeitherComputedNorCountedAsMatched()
    {
        string loraPath = WriteLora("absent", "blocks.0");
        Dictionary<string, Tensor> weights = new() { ["blocks.7.to_q.weight"] = Zeros(Rows, Cols) };

        using LoraStack stack = new LoraStack();
        stack.AddFromPath(loraPath, strength: 0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend());

        Assert.Equal(0, stack.Report.ZeroStrengthMatched);
        Assert.Equal(0, stack.Report.SkippedMissing);
    }

    [Fact]
    public void Report_CountsSourceUnitsDestinationsAndSkipsApart()
    {
        string loraPath = WriteLora("mixed", "blocks.0", "blocks.1");
        Dictionary<string, Tensor> weights = new() { [BaseKey] = Zeros(Rows, Cols) };

        using LoraStack stack = new LoraStack();
        stack.AddFromPath(loraPath, strength: 1f);
        int merged = stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend());

        // blocks.0 merges into one tensor; blocks.1 has no target and is counted as one skipped unit, not as a weight.
        Assert.Equal(1, merged);
        Assert.Equal(new LoraStack.ApplyReport(1, 1, 0, 1, 0, 0), stack.Report);
    }

    [Fact]
    public void Report_FusedSliceIsCountedApartFromItsDestination()
    {
        string loraPath = WriteLora("fused", "blocks.0");
        Dictionary<string, Tensor> weights = new() { ["blocks.0.qkv.weight"] = Zeros(Rows * 3, Cols) };

        using LoraStack stack = new LoraStack();
        stack.AddFromPath(loraPath, strength: 1f);
        int merged = stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend());

        // One split to_q unit lands as a row slice of the fused qkv weight: one unit, one fused slice, one destination.
        Assert.Equal(1, merged);
        Assert.Equal(new LoraStack.ApplyReport(1, 1, 1, 0, 0, 0), stack.Report);
    }

    [Fact]
    public void PartialApplication_Allow_AppliesWhatFitsAndCountsTheRest()
    {
        string loraPath = WriteLora("partial-allow", "blocks.0", "blocks.9");
        Tensor baseTensor = Zeros(Rows, Cols);
        Dictionary<string, Tensor> weights = new() { [BaseKey] = baseTensor };

        using LoraStack stack = new LoraStack();
        stack.AddFromPath(loraPath, strength: 1f);
        int merged = stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend());

        Assert.Equal(1, merged);
        Assert.NotSame(baseTensor, weights[BaseKey]);
        Assert.Equal(1, stack.Report.SkippedMissing);
    }

    [Fact]
    public void PartialApplication_Refuse_RefusesBeforeAnyWriteAndNamesTheOmittedUnit()
    {
        string loraPath = WriteLora("partial-refuse", "blocks.0", "blocks.9");
        Tensor baseTensor = Zeros(Rows, Cols);
        Dictionary<string, Tensor> weights = new() { [BaseKey] = baseTensor };

        using LoraStack stack = new LoraStack { PartialPolicy = LoraPartialPolicy.Refuse };
        stack.AddFromPath(loraPath, strength: 1f);

        NotSupportedException refused = Assert.Throws<NotSupportedException>(
            () => stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));
        Assert.Contains("blocks.9", refused.Message, StringComparison.Ordinal);
        Assert.Contains("lora.partialApplication", refused.Message, StringComparison.Ordinal);
        Assert.Same(baseTensor, weights[BaseKey]);
    }

    /// <summary>A PEFT LoRA with a rank-2 pair for each named block's <c>to_q</c>, so every module carries a real delta.</summary>
    private string WriteLora(string name, params string[] blocks)
    {
        Dictionary<string, (long[] Shape, float[] Data)> tensors = new(StringComparer.Ordinal);
        foreach (string block in blocks)
        {
            tensors[$"transformer.{block}.to_q.lora_A.weight"] = (new long[] { Rank, Cols }, Filled(Rank * Cols, 0.5f));
            tensors[$"transformer.{block}.to_q.lora_B.weight"] = (new long[] { Rows, Rank }, Filled(Rows * Rank, 0.25f));
        }
        return CreateSafeTensors(name, tensors);
    }

    private static float[] Filled(int count, float value)
    {
        float[] data = new float[count];
        Array.Fill(data, value);
        return data;
    }

    private static unsafe Tensor Zeros(int rows, int cols)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            p[i] = 0f;
        }
        return t;
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
