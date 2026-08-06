using System.Text;
using System.Text.Json;
using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>MiniMax-H3 LoRA support. H3's converted DiT keys are verbatim upstream names, so a diffusers-PEFT export
/// lands on the existing architecture-agnostic passthrough with no H3-specific mapper — these tests pin that, and pin
/// the merge arithmetic against H3's real tensor shapes, including its <b>fused</b> qkv projection.</summary>
public sealed class LoraMiniMaxH3Tests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "h3-lora-" + Guid.NewGuid().ToString("N"));

    public LoraMiniMaxH3Tests() => Directory.CreateDirectory(_dir);

    /// <summary>H3's attention is one fused <c>qkv_proj</c> of 3·heads·headDim rows, not three matrices — a LoRA
    /// trained against split q/k/v could not merge into it, so the fused shape is the one that must round-trip.</summary>
    [Fact]
    public void DiffusersPeftKeysPassThroughToCanonicalH3Names()
    {
        const int hidden = 32, inner = 3 * 16, rank = 4;
        string path = CreateSafeTensors("h3lora", new Dictionary<string, (long[] shape, float[] data)>
        {
            ["transformer.blocks.0.attn.qkv_proj.lora_A.weight"] = ([rank, hidden], Filled(rank * hidden, 0.1f)),
            ["transformer.blocks.0.attn.qkv_proj.lora_B.weight"] = ([inner, rank], Filled(inner * rank, 0.2f)),
            ["transformer.blocks.0.mlp.fc2.lora_A.weight"] = ([rank, hidden], Filled(rank * hidden, 0.1f)),
            ["transformer.blocks.0.mlp.fc2.lora_B.weight"] = ([hidden, rank], Filled(hidden * rank, 0.3f)),
        });

        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.DiffusersFlux, file.Format);
        List<string> targets = [.. file.Layers.Select(l => l.TargetKey).OrderBy(k => k, StringComparer.Ordinal)];
        Assert.Equal(["blocks.0.attn.qkv_proj.weight", "blocks.0.mlp.fc2.weight"], targets);
        Assert.All(file.Layers, l => Assert.Equal(LoraTarget.Transformer, l.Target));
    }

    /// <summary>The merged delta must land on the fused qkv rows with the right magnitude — a silently no-op merge and
    /// a correct one are indistinguishable without checking the arithmetic.</summary>
    [Fact]
    public void MergeAppliesTheDeltaToFusedQkvWeights()
    {
        const int hidden = 32, inner = 3 * 16, rank = 4;
        string path = CreateSafeTensors("h3merge", new Dictionary<string, (long[] shape, float[] data)>
        {
            ["transformer.blocks.0.attn.qkv_proj.lora_A.weight"] = ([rank, hidden], Filled(rank * hidden, 0.1f)),
            ["transformer.blocks.0.attn.qkv_proj.lora_B.weight"] = ([inner, rank], Filled(inner * rank, 0.2f)),
        });

        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>
        {
            ["blocks.0.attn.qkv_proj.weight"] = Zeros(inner, hidden),
        };
        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(path, strength: 1.0f);
        int merged = stack.ApplyTo(weights, LoraTarget.Transformer, backend);

        Assert.Equal(1, merged);
        // alpha defaults to rank, so the scale is 1: every output element is rank · 0.1 · 0.2.
        float expected = rank * 0.1f * 0.2f;
        Tensor fused = weights["blocks.0.attn.qkv_proj.weight"];
        Assert.Equal(inner, (int)fused.Shape[0]);
        unsafe
        {
            float* p = (float*)fused.DataPointer;
            for (long i = 0; i < fused.ElementCount; i++)
            {
                Assert.Equal(expected, p[i], 4);
            }
        }
    }

    /// <summary>A LoRA whose keys name no H3 weight must be reported, not silently ignored — the engine logs a
    /// zero-match warning, and this is the unit-level counterpart of that signal.</summary>
    [Fact]
    public void KeysThatNameNoH3WeightMergeNothing()
    {
        const int hidden = 32, rank = 4;
        string path = CreateSafeTensors("h3miss", new Dictionary<string, (long[] shape, float[] data)>
        {
            ["transformer.blocks.0.attn.to_q.lora_A.weight"] = ([rank, hidden], Filled(rank * hidden, 0.1f)),
            ["transformer.blocks.0.attn.to_q.lora_B.weight"] = ([hidden, rank], Filled(hidden * rank, 0.2f)),
        });
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>
        {
            ["blocks.0.attn.qkv_proj.weight"] = Zeros(3 * 16, hidden),
        };
        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(path, strength: 1.0f);
        Assert.Equal(0, stack.ApplyTo(weights, LoraTarget.Transformer, backend));
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
        new Span<float>((void*)t.DataPointer, rows * cols).Clear();
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
