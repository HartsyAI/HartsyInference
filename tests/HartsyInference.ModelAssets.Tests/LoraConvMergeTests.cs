using System.Text;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Merging a LoRA into a CONVOLUTION weight, which SD1.5 and SDXL UNets are mostly made of.
/// <para>The delta arrives flattened to <c>[out, in·kh·kw]</c> while the weight is <c>[out, in, kh, kw]</c>. Those
/// are the same bytes in the same order row-major, so the add lands correctly once the shapes agree — the previous
/// rank-2-only gate was conservative rather than necessary. What makes that worth a test is that being wrong here
/// is silent: a misaligned add still produces a weight of the right size and a picture of the right subject.</para>
/// </summary>
public sealed class LoraConvMergeTests : IDisposable
{
    private const int OutCh = 4, InCh = 3, K = 3, Rank = 2;
    private const int FlatIn = InCh * K * K;
    private const float DownValue = 0.5f, UpValue = 0.25f;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lora-conv-" + Guid.NewGuid().ToString("N"));

    public LoraConvMergeTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>Every element of a rank-4 weight gets its own delta element, at the offset the flatten implies.
    /// Asserted elementwise against a hand-computed reference rather than by norm, because a delta added in the
    /// wrong order has the same norm as one added in the right order.</summary>
    [Theory]
    [InlineData(1.0f)]
    [InlineData(0.5f)]
    public void AConvolutionWeightTakesItsFlattenedDelta(float strength)
    {
        string path = ConvLora("conv");
        using Tensor merged = MergeThroughStack(path, strength, out int mergedCount);
        Assert.Equal(1, mergedCount);

        // B @ A with both filled: every element of ΔW is Rank · DownValue · UpValue.
        float deltaValue = Rank * DownValue * UpValue * strength;
        float[] actual = Read(merged);
        Assert.Equal(OutCh * FlatIn, actual.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.Equal(BaseValue(i) + deltaValue, actual[i], 4);
        }
    }

    /// <summary>The weight keeps its rank-4 shape through the merge — a convolution whose weight came back as
    /// <c>[out, in·kh·kw]</c> would not be usable by the conv kernels that read its spatial axes.</summary>
    [Fact]
    public void TheMergedWeightIsStillRankFour()
    {
        string path = ConvLora("conv-shape");
        using Tensor merged = MergeThroughStack(path, 1.0f, out _);
        Assert.Equal(4, merged.Shape.Rank);
        Assert.Equal(OutCh, (int)merged.Shape[0]);
        Assert.Equal(InCh, (int)merged.Shape[1]);
        Assert.Equal(K, (int)merged.Shape[2]);
        Assert.Equal(K, (int)merged.Shape[3]);
    }

    /// <summary>A DoRA adapter on a convolution is refused by name. Its magnitude vector normalizes by a row norm
    /// of the weight as a MATRIX, and a convolution has no such matrix until it is flattened — which axis the
    /// vector describes is then the file's choice, not ours.</summary>
    [Fact]
    public void ADoraAdapterOnAConvolutionIsRefused()
    {
        string path = CreateSafeTensors("conv-dora", new Dictionary<string, (long[] Shape, float[] Data)>
        {
            ["transformer.blocks.0.conv.lora_A.weight"] = ([Rank, FlatIn], Filled(Rank * FlatIn, DownValue)),
            ["transformer.blocks.0.conv.lora_B.weight"] = ([OutCh, Rank], Filled(OutCh * Rank, UpValue)),
            ["transformer.blocks.0.conv.dora_scale"] = ([OutCh], Filled(OutCh, 1.2f)),
        });
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => MergeThroughStack(path, 1.0f, out _));
        Assert.Contains("convolution", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    private string ConvLora(string name) =>
        CreateSafeTensors(name, new Dictionary<string, (long[] Shape, float[] Data)>
        {
            // The file stores the down projection already flattened, which is what a LoCon/LyCORIS conv adapter
            // does — ComfyUI's calculate_weight flattens from dim 1 before reshaping onto the weight.
            ["transformer.blocks.0.conv.lora_A.weight"] = ([Rank, FlatIn], Filled(Rank * FlatIn, DownValue)),
            ["transformer.blocks.0.conv.lora_B.weight"] = ([OutCh, Rank], Filled(OutCh * Rank, UpValue)),
        });

    /// <summary>Deliberately non-constant so a delta landing on the wrong element changes the result.</summary>
    private static float BaseValue(int index) => 0.1f + (0.013f * index);

    private Tensor MergeThroughStack(string path, float strength, out int merged)
    {
        Tensor baseW = new Tensor(new TensorShape(OutCh, InCh, K, K), DType.F32);
        Span<float> span = baseW.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = BaseValue(i);

        Dictionary<string, Tensor> weights = new() { ["blocks.0.conv.weight"] = baseW };
        using LoraStack stack = new LoraStack();
        using CpuBackend backend = new CpuBackend();
        stack.AddFromPath(path, strength: strength);
        merged = stack.ApplyTo(weights, LoraTarget.Transformer, backend);
        Tensor result = weights["blocks.0.conv.weight"];
        Tensor copy = new Tensor(result.Shape, DType.F32);
        Read(result).AsSpan().CopyTo(copy.AsSpan<float>());
        baseW.Dispose();
        return copy;
    }

    private static unsafe float[] Read(Tensor t)
    {
        float[] data = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) data[i] = p[i];
        return data;
    }

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
            foreach (float value in kvp.Value.Data) dataStream.Write(BitConverter.GetBytes(value), 0, 4);
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

    /// <summary>A half-precision convolution base, which is what an SD1.5 fp16 checkpoint actually has. The merge
    /// runs in F32 and the result has to come back at the base's own dtype AND its own rank — a conv weight
    /// returned as F32, or flattened, is not something the conv kernels can read.</summary>
    [Fact]
    public void AHalfPrecisionConvolutionBaseComesBackAtItsOwnDtypeAndRank()
    {
        string path = ConvLora("conv-bf16");
        using Tensor baseW = new Tensor(new TensorShape(OutCh, InCh, K, K), DType.F32);
        Span<float> span = baseW.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = BaseValue(i);
        using Tensor bf16Base = baseW.CastTo(DType.BF16);

        Dictionary<string, Tensor> weights = new() { ["blocks.0.conv.weight"] = bf16Base };
        using LoraStack stack = new LoraStack();
        using CpuBackend backend = new CpuBackend();
        stack.AddFromPath(path, strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));

        Tensor merged = weights["blocks.0.conv.weight"];
        Assert.Equal(DType.BF16, merged.DType);
        Assert.Equal(4, merged.Shape.Rank);
        Assert.Equal(InCh, (int)merged.Shape[1]);
        // BF16 carries about three decimal digits, so the delta has to be visible well above that.
        using Tensor asF32 = merged.CastTo(DType.F32);
        float expected = BaseValue(0) + (Rank * DownValue * UpValue);
        Assert.Equal(expected, Read(asF32)[0], 2);
    }
}
