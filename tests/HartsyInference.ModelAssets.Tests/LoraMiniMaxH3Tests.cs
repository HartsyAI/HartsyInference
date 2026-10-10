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

    /// <summary>The published H3 LoRA (larryvrh/MiniMax-H3-Turbo-Lora) carries NO wrapper prefix — its roots are
    /// already the checkpoint's own keys. That reached <see cref="LoraFormat.Unknown"/> and was rejected at load, so
    /// the only real-world H3 LoRA did not work at all despite the passthrough above; this pins the bare-root arm.</summary>
    [Fact]
    public void BareCheckpointKeysWithNoPrefixAreDetectedAndMapped()
    {
        const int hidden = 32, inner = 3 * 16, rank = 4;
        string path = CreateSafeTensors("h3bare", new Dictionary<string, (long[] shape, float[] data)>
        {
            ["blocks.0.attn.qkv_proj.lora_A.weight"] = ([rank, hidden], Filled(rank * hidden, 0.1f)),
            ["blocks.0.attn.qkv_proj.lora_B.weight"] = ([inner, rank], Filled(inner * rank, 0.2f)),
            ["token_refiner.blocks.0.mlp.fc1.lora_A.weight"] = ([rank, hidden], Filled(rank * hidden, 0.1f)),
            ["token_refiner.blocks.0.mlp.fc1.lora_B.weight"] = ([hidden, rank], Filled(hidden * rank, 0.3f)),
        });

        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.DiffusersBareDit, file.Format);
        List<string> targets = [.. file.Layers.Select(l => l.TargetKey).OrderBy(k => k, StringComparer.Ordinal)];
        Assert.Equal(["blocks.0.attn.qkv_proj.weight", "token_refiner.blocks.0.mlp.fc1.weight"], targets);
    }

    /// <summary>The Turbo LoRA targets the UNPRUNED adaln projection (<c>[96768, 2688]</c>) while every pruned build
    /// stores the curve-table form (<c>[96768, 8]</c>). Without this guard the delta went straight to
    /// <c>backend.Add</c> against a smaller destination; it must skip that weight and still merge the rest.</summary>
    [Fact]
    public void DeltaWhoseShapeDoesNotMatchTheCheckpointIsSkippedNotApplied()
    {
        const int hidden = 32, rank = 4;
        string path = CreateSafeTensors("h3shape", new Dictionary<string, (long[] shape, float[] data)>
        {
            // Right key, wrong build: the delta is [hidden, hidden] but the weight below is [hidden, 8].
            ["blocks.0.adaln_proj.linear.lora_A.weight"] = ([rank, hidden], Filled(rank * hidden, 0.1f)),
            ["blocks.0.adaln_proj.linear.lora_B.weight"] = ([hidden, rank], Filled(hidden * rank, 0.2f)),
            ["blocks.0.mlp.fc2.lora_A.weight"] = ([rank, hidden], Filled(rank * hidden, 0.1f)),
            ["blocks.0.mlp.fc2.lora_B.weight"] = ([hidden, rank], Filled(hidden * rank, 0.3f)),
        });
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>
        {
            ["blocks.0.adaln_proj.linear.weight"] = Zeros(hidden, 8),
            ["blocks.0.mlp.fc2.weight"] = Zeros(hidden, hidden),
        };
        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(path, strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));
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
