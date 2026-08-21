using System.Text;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins the split-LoRA-onto-fused-weight fallback in <see cref="LoraStack"/> for the attention spellings that
/// are NOT Flux-lineage <c>attn.qkv</c>: Ideogram 4's <c>layers.{i}.attention.qkv.weight</c> and F-Lite's bare
/// <c>blocks.{i}.qkv.weight</c>.
/// <para>This is the one LoRA failure mode with no runtime signal. A split-form LoRA whose attention keys find no home
/// still merges its non-attention weights, so the merge count is non-zero, <c>LoraApplier</c>'s zero-match refusal never
/// fires, and the generation succeeds — producing an image that is subtly under-LoRA'd and reads to the user as a weak
/// LoRA rather than a bug. That is exactly what commit <c>fc975b71</c> found on Chroma (418 weights merged, visibly
/// muted output) and fixed for <c>attn.qkv</c> only; these two spellings were still open when their families were
/// wired for LoRA on 2026-08-20.</para></summary>
public sealed class LoraFusedAttentionSliceTests : IDisposable
{
    private const int HeadRows = 8, Cols = 4, Rank = 2;

    /// <summary>Per-element delta the merge must produce. Comfortably above F32 noise so a zero merge is unmistakable.</summary>
    private const float Delta = 0.25f;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lora-fused-" + Guid.NewGuid().ToString("N"));

    public LoraFusedAttentionSliceTests() => Directory.CreateDirectory(_dir);

    /// <summary>Ideogram 4 (<c>attention.qkv</c>) and F-Lite (bare <c>qkv</c>) both resolve, and the delta lands on the
    /// correct third of the fused rows — q on [0,8), k on [8,16), v on [16,24). A fallback that resolved the key but
    /// mis-sliced would corrupt attention rather than weaken it, so the row window is asserted, not just the count.</summary>
    [Theory]
    [InlineData("layers.0.attention", "attention.qkv")]
    [InlineData("blocks.0", "qkv")]
    public void SplitLoraMergesIntoFusedQkv_AtTheRightRows(string prefix, string fusedSuffix)
    {
        string loraPath = SplitQkvLora("split", prefix);
        string fusedKey = $"{FusedRoot(prefix, fusedSuffix)}.weight";
        Tensor fused = Zeros(HeadRows * 3, Cols);
        Dictionary<string, Tensor> weights = new() { [fusedKey] = fused };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(loraPath, strength: 1.0f);
        int merged = stack.ApplyTo(weights, LoraTarget.Transformer, backend);

        Assert.Equal(1, merged);
        float[] result = Read(weights[fusedKey]);
        // q rows carry the delta (only lora_A/B for to_q were written); k and v rows stay zero.
        for (int r = 0; r < HeadRows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                Assert.Equal(Delta, result[(r * Cols) + c], 3);
            }
        }
        for (int r = HeadRows; r < HeadRows * 3; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                Assert.Equal(0f, result[(r * Cols) + c], 5);
            }
        }
    }

    /// <summary>The fallback must never shadow a direct hit: when the dict carries the SPLIT weight, the ordinary
    /// lookup serves it and the fused path is not consulted. Guards the broad bare-<c>qkv</c> table entry, whose suffix
    /// (<c>.to_q.weight</c>) also matches every Flux-lineage key.</summary>
    [Fact]
    public void SplitWeightPresent_TakesTheDirectPath_NotTheFallback()
    {
        string loraPath = SplitQkvLora("direct", "blocks.0");
        Tensor split = Zeros(HeadRows, Cols);
        Tensor fused = Zeros(HeadRows * 3, Cols);
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.to_q.weight"] = split,
            ["blocks.0.qkv.weight"] = fused,
        };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(loraPath, strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));

        Assert.All(Read(weights["blocks.0.to_q.weight"]), v => Assert.Equal(Delta, v, 3));
        Assert.All(Read(weights["blocks.0.qkv.weight"]), v => Assert.Equal(0f, v, 5));
    }

    private static string FusedRoot(string prefix, string fusedSuffix) =>
        prefix.EndsWith(".attention", StringComparison.Ordinal)
            ? prefix[..^".attention".Length] + "." + fusedSuffix
            : prefix + "." + fusedSuffix;

    private static Tensor Zeros(int rows, int cols)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        Read(t).AsSpan().Clear();
        unsafe
        {
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < t.ElementCount; i++)
            {
                p[i] = 0f;
            }
        }
        return t;
    }

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

    /// <summary>A PEFT LoRA naming only the SPLIT <c>to_q</c> module under <paramref name="prefix"/>.</summary>
    private string SplitQkvLora(string name, string prefix)
    {
        float a = 0.5f;
        float b = Delta / (Rank * a);
        return CreateSafeTensors(name, new Dictionary<string, (long[] Shape, float[] Data)>
        {
            [$"transformer.{prefix}.to_q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, a)),
            [$"transformer.{prefix}.to_q.lora_B.weight"] = ([HeadRows, Rank], Filled(HeadRows * Rank, b)),
        });
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
