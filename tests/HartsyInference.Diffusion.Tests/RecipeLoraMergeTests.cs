using System.Text;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Engine.Features;
using Xunit;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The one LoRA call site every recipe now uses. Pins the two things it owns that a recipe used to restate
/// for itself: the split between model and text-encoder strength, and the refusal when a file matches nothing.</summary>
public sealed class RecipeLoraMergeTests : IDisposable
{
    private const int Rows = 4, Cols = 8, Rank = 2;
    private const float DownValue = 0.5f, UpValue = 0.25f;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "recipe-lora-" + Guid.NewGuid().ToString("N"));

    public RecipeLoraMergeTests() => Directory.CreateDirectory(_dir);

    [Fact]
    public void EncoderTargetsTakeTencStrength_WhileTheBodyKeepsModelStrength()
    {
        string path = TwoArmLora("split");
        Dictionary<string, Tensor> transformer = new() { ["blocks.0.attn.to_q.weight"] = Zeros() };
        Dictionary<string, Tensor> clipL = new() { ["blocks.0.attn.to_q.weight"] = Zeros() };

        using MergedLoraStack? stack = RecipeLoraMerge.Apply(
            [new LoraResolver.LoraSpec { FilePath = path, ModelStrength = 1.0f, TencStrength = 0.25f }],
            new CpuBackend(),
            new LoraMergeTargets { Transformer = transformer, ClipL = clipL },
            "RecipeLoraMergeTests");

        Assert.NotNull(stack);
        float bodyDelta = transformer["blocks.0.attn.to_q.weight"].AsReadOnlySpan<float>()[0];
        float encoderDelta = clipL["blocks.0.attn.to_q.weight"].AsReadOnlySpan<float>()[0];
        Assert.Equal(Rank * DownValue * UpValue, bodyDelta, 5);
        Assert.Equal(bodyDelta * 0.25f, encoderDelta, 5);
    }

    [Fact]
    public void NoLoras_ReturnsNullWithoutTouchingTheWeights()
    {
        Dictionary<string, Tensor> transformer = new() { ["blocks.0.attn.to_q.weight"] = Zeros() };
        Tensor original = transformer["blocks.0.attn.to_q.weight"];

        MergedLoraStack? stack = RecipeLoraMerge.Apply(
            [], new CpuBackend(), new LoraMergeTargets { Transformer = transformer }, "RecipeLoraMergeTests");

        Assert.Null(stack);
        Assert.Same(original, transformer["blocks.0.attn.to_q.weight"]);
        original.Dispose();
    }

    [Fact]
    public void ZeroMatches_RefusesRatherThanGeneratingWithoutTheLora()
    {
        // The failure this refusal exists for: a wrong-family LoRA that merges nothing still generates, and the
        // user reads the unchanged image as a weak LoRA rather than a broken one.
        string path = TwoArmLora("nomatch");
        Dictionary<string, Tensor> transformer = new() { ["some.other.module.weight"] = Zeros() };

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => RecipeLoraMerge.Apply(
            [new LoraResolver.LoraSpec { ModelId = "wrong-family", FilePath = path, ModelStrength = 1.0f, TencStrength = 1.0f }],
            new CpuBackend(),
            new LoraMergeTargets { Transformer = transformer },
            "RecipeLoraMergeTests"));
        Assert.Contains("wrong-family", error.Message);
        transformer["some.other.module.weight"].Dispose();
    }

    private static Tensor Zeros()
    {
        Tensor t = new Tensor(new TensorShape(Rows, Cols), DType.F32);
        t.AsSpan<float>().Clear();
        return t;
    }

    /// <summary>A PEFT file naming the SAME module on both the transformer and the CLIP-L arm.</summary>
    private string TwoArmLora(string name) => CreateSafeTensors(name, new()
    {
        ["transformer.blocks.0.attn.to_q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
        ["transformer.blocks.0.attn.to_q.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
        ["text_encoder.blocks.0.attn.to_q.lora_A.weight"] = ([Rank, Cols], Filled(Rank * Cols, DownValue)),
        ["text_encoder.blocks.0.attn.to_q.lora_B.weight"] = ([Rows, Rank], Filled(Rows * Rank, UpValue)),
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
