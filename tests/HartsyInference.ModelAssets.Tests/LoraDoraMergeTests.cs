using System.Text;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>That a DoRA adapter reaching a dense weight is actually decomposed, rather than added as if it were a
/// plain LoRA.
/// <para><see cref="LoraDoraDecompose"/> was built and pinned against ComfyUI's reference before anything called it,
/// so the merge path dropped the magnitude vector on the floor: correct-looking output, no log line, and a result
/// the user reads as "the LoRA is weak". That is the one failure this phase exists to eliminate, and the quantized
/// path already refused by name — only the dense path was silent.</para></summary>
public sealed class LoraDoraMergeTests : IDisposable
{
    private const int Rows = 4, Cols = 6, Rank = 2;
    private const float DownValue = 0.5f, UpValue = 0.25f;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lora-dora-" + Guid.NewGuid().ToString("N"));

    public LoraDoraMergeTests() => Directory.CreateDirectory(_dir);

    /// <summary>The output-axis branch (magnitude length == row count), which is the common LyCORIS case. Asserted
    /// against the decomposition AND against the additive merge, because only the second assertion fails when the
    /// magnitude vector is ignored — the first would pass on any implementation that happens to touch the weight.</summary>
    [Theory]
    [InlineData(1.0f)]
    [InlineData(0.5f)]
    public void DenseMerge_DecomposesOnTheOutputAxis(float strength)
    {
        float[] magnitude = [1.3f, 0.7f, 2.1f, 0.9f];
        string path = DoraLora("out-axis", magnitude, [Rows]);

        using Tensor merged = MergeThroughStack(path, strength, out int mergedCount);
        Assert.Equal(1, mergedCount);

        float[] expected = DecomposeReference(magnitude, strength);
        float[] additive = AdditiveReference(strength);
        float[] actual = Read(merged);

        for (int i = 0; i < actual.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 4);
        }
        // The negative control: without the magnitude vector the merge lands on `additive` instead. If the two
        // references ever coincide the first assertion proves nothing, so the fixture has to keep them apart.
        int differing = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            if (Math.Abs(expected[i] - additive[i]) > 1e-3f) { differing++; }
        }
        Assert.True(differing > actual.Length / 2,
            $"fixture is vacuous: only {differing}/{actual.Length} elements distinguish the decomposition from a "
            + "plain additive merge.");
    }

    /// <summary>The other branch, selected when the magnitude vector's length matches the COLUMN count: the
    /// normalizer becomes a column norm of the LoRA'd weight rather than a row norm of the original.</summary>
    [Fact]
    public void DenseMerge_DecomposesOnTheInputAxis()
    {
        float[] magnitude = [1.1f, 0.8f, 1.6f, 0.6f, 1.4f, 0.95f];
        string path = DoraLora("in-axis", magnitude, [Cols]);

        using Tensor merged = MergeThroughStack(path, strength: 1.0f, out int mergedCount);
        Assert.Equal(1, mergedCount);

        float[] expected = DecomposeReference(magnitude, strength: 1.0f);
        float[] actual = Read(merged);
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 4);
        }
    }

    /// <summary>The file has to parse as DoRA in the first place — if the <c>.dora_scale</c> suffix stopped routing to
    /// <see cref="LoraDelta.DoraScale"/> the merge above would silently go back to being a plain LoRA and still pass
    /// its own reference.</summary>
    [Fact]
    public void ADoraFileParsesAsDoRA()
    {
        using LoraFile file = LoraFile.Load(DoraLora("variant", [1.3f, 0.7f, 2.1f, 0.9f], [Rows]));
        LoraLayer layer = Assert.Single(file.Layers);
        Assert.Equal(LoraVariant.DoRA, layer.Delta.Variant);
        Assert.NotNull(layer.Delta.DoraScale);
    }

    /// <summary>A DoRA adapter aimed at one slice of a fused projection is refused by name. The magnitude vector is
    /// defined over a whole weight; on the input axis its normalizer is a column norm across every row, which a third
    /// of the rows cannot supply, and which branch applies is the file's choice rather than ours.</summary>
    [Fact]
    public void FusedSliceMerge_RefusesADoraAdapterByName()
    {
        float[] magnitude = [1.3f, 0.7f, 2.1f, 0.9f];
        string path = CreateSafeTensors("fused-dora", new Dictionary<string, (long[] Shape, float[] Data)>
        {
            ["transformer.blocks.0.to_q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
            ["transformer.blocks.0.to_q.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
            ["transformer.blocks.0.to_q.dora_scale"] = ([Rows], magnitude),
        });

        using Tensor fused = Weight(Rows * 3, Cols);
        Dictionary<string, Tensor> weights = new() { ["blocks.0.qkv.weight"] = fused };
        using LoraStack stack = new LoraStack();
        using CpuBackend backend = new CpuBackend();
        stack.AddFromPath(path, strength: 1.0f);

        NotSupportedException ex = Assert.Throws<NotSupportedException>(
            () => stack.ApplyTo(weights, LoraTarget.Transformer, backend));
        Assert.Contains("blocks.0.qkv.weight", ex.Message);
        Assert.Contains("DoRA", ex.Message);
    }

    private Tensor MergeThroughStack(string path, float strength, out int merged)
    {
        Tensor baseW = Weight(Rows, Cols);
        Dictionary<string, Tensor> weights = new() { ["blocks.0.to_q.weight"] = baseW };
        using LoraStack stack = new LoraStack();
        using CpuBackend backend = new CpuBackend();
        stack.AddFromPath(path, strength: strength);
        merged = stack.ApplyTo(weights, LoraTarget.Transformer, backend);
        Tensor result = weights["blocks.0.to_q.weight"];
        // ApplyTo replaced the entry with a tensor the stack owns; copy it out before the stack disposes.
        Tensor copy = new Tensor(result.Shape, DType.F32);
        Read(result).AsSpan().CopyTo(copy.AsSpan<float>());
        baseW.Dispose();
        return copy;
    }

    /// <summary>W and ΔW run through the decomposition directly. Its own arithmetic is pinned against ComfyUI in
    /// <c>LoraDeltaMathTests</c>; what this file tests is that the merge path reaches it at all.</summary>
    private static float[] DecomposeReference(float[] magnitude, float strength)
    {
        using Tensor weight = Weight(Rows, Cols);
        using Tensor delta = new Tensor(new TensorShape(Rows, Cols), DType.F32);
        delta.AsSpan<float>().Fill(Rank * DownValue * UpValue);
        using Tensor scale = new Tensor(new TensorShape(magnitude.Length), DType.F32);
        magnitude.AsSpan().CopyTo(scale.AsSpan<float>());
        LoraDoraDecompose.Apply(weight, delta, scale, scale: 1.0f, strength);
        return Read(weight);
    }

    /// <summary>What the merge produced before the decomposition was wired in: W + strength·scale·ΔW.</summary>
    private static float[] AdditiveReference(float strength)
    {
        using Tensor weight = Weight(Rows, Cols);
        float[] values = Read(weight);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] += strength * Rank * DownValue * UpValue;
        }
        return values;
    }

    /// <summary>A base with distinct per-row and per-column norms, so neither branch's normalizer collapses to a
    /// constant and a swapped axis is visible in the result.</summary>
    private static Tensor Weight(int rows, int cols)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        Span<float> span = t.AsSpan<float>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                span[(r * cols) + c] = 0.1f + (0.37f * r) - (0.11f * c);
            }
        }
        return t;
    }

    private string DoraLora(string name, float[] magnitude, long[] magnitudeShape) =>
        CreateSafeTensors(name, new Dictionary<string, (long[] Shape, float[] Data)>
        {
            ["transformer.blocks.0.to_q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
            ["transformer.blocks.0.to_q.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
            ["transformer.blocks.0.to_q.dora_scale"] = (magnitudeShape, magnitude),
        });

    private static unsafe float[] Read(Tensor t)
    {
        float[] data = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            data[i] = p[i];
        }
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
