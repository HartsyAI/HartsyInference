using System.Text;
using System.Text.Json;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>LoRA against the real MiniMax-H3 DiT. The synthetic unit tests pin the merge arithmetic; these pin the
/// parts only a real checkpoint can: that the converter's canonical key names are the ones a diffusers-PEFT LoRA
/// targets, that the real fused-qkv shape merges, and that the quantized build is refused with a message naming the
/// build to use rather than an opaque failure from inside the merge.</summary>
[Trait("Category", "Integration")]
public sealed class MiniMaxH3LoraRealWeightTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "h3-lora-real-" + Guid.NewGuid().ToString("N"));

    public MiniMaxH3LoraRealWeightTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    /// <summary>The bf16 build, which is the one a LoRA can merge into. Env override MINIMAX_H3_DIT_BF16.</summary>
    private static string Bf16Dit => Environment.GetEnvironmentVariable("MINIMAX_H3_DIT_BF16")
        ?? Path.Combine(RepoRoot.Path, "Models", "Stable-Diffusion", "MiniMaxH3", "FL2VA", "transformer",
            "minimax_h3_fl2va_bf16.safetensors");

    [Fact]
    public void DiffusersPeftLoraMergesIntoTheRealFusedQkv()
    {
        if (!File.Exists(Bf16Dit))
        {
            _output.WriteLine($"skipped: {Bf16Dit} not present");
            return;
        }
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(Bf16Dit);
        Dictionary<string, Tensor> raw = new Dictionary<string, Tensor>(loader.GetAllTensors());
        MiniMaxH3CheckpointConverter.ConvertedWeights converted =
            MiniMaxH3CheckpointConverter.Convert(raw, castToF32: false);

        const string target = "blocks.0.attn.qkv_proj.weight";
        Assert.True(converted.Transformer.ContainsKey(target),
            $"the converted checkpoint has no '{target}' — the canonical naming a PEFT LoRA targets has changed");
        Tensor baseWeight = converted.Transformer[target];
        int outFeatures = (int)baseWeight.Shape[0], inFeatures = (int)baseWeight.Shape[1];
        _output.WriteLine($"{target}: [{outFeatures}, {inFeatures}] {baseWeight.DType.Name}");

        // Only this one tensor is merged, so the test costs one F32 copy rather than the whole 66 GB DiT.
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor> { [target] = baseWeight };
        float[] before = Sample(baseWeight);

        const int rank = 4;
        // The merged weight is stored back at the checkpoint's BF16, whose quantum near these magnitudes is ~1e-3.
        // A delta below that would round away — so the probe delta is deliberately well clear of it.
        const float aValue = 0.1f, bValue = 0.2f;
        string lora = CreateSafeTensors("h3real", new Dictionary<string, (long[] Shape, float[] Data)>
        {
            [$"transformer.{target[..^".weight".Length]}.lora_A.weight"] =
                ([rank, inFeatures], Filled(rank * inFeatures, aValue)),
            [$"transformer.{target[..^".weight".Length]}.lora_B.weight"] =
                ([outFeatures, rank], Filled(outFeatures * rank, bValue)),
        });

        using LoraFile file = LoraFile.Load(lora);
        Assert.Equal(target, Assert.Single(file.Layers).TargetKey);

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(lora, strength: 1.0f);
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));

        float[] after = Sample(weights[target]);
        float expectedDelta = rank * aValue * bValue;
        Assert.Equal(baseWeight.DType, weights[target].DType);
        double worst = 0;
        for (int i = 0; i < before.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs((after[i] - before[i]) - expectedDelta));
        }
        // BF16 carries 8 mantissa bits, so a round-tripped sum is only good to ~0.4% of its own magnitude.
        double tolerance = Math.Abs(expectedDelta) * 0.02 + 0.004;
        _output.WriteLine($"expected delta {expectedDelta:F5}, worst deviation {worst:E3} (tolerance {tolerance:E3})");
        Assert.True(worst < tolerance, $"merged delta is off by {worst:E3}, beyond BF16 round-trip precision");
    }

    /// <summary>The production build is fp8, and a merge cannot rewrite a quantized weight. The failure has to name
    /// the bf16 build, because "scale factor is not 1" tells a user nothing about what to do.</summary>
    [Fact]
    public void TheQuantizedBuildIsRefusedByTheMerge()
    {
        if (!File.Exists(TestPaths.MiniMaxH3.DitFp8))
        {
            _output.WriteLine($"skipped: {TestPaths.MiniMaxH3.DitFp8} not present");
            return;
        }
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(TestPaths.MiniMaxH3.DitFp8);
        Dictionary<string, Tensor> raw = new Dictionary<string, Tensor>(loader.GetAllTensors());
        MiniMaxH3CheckpointConverter.ConvertedWeights converted =
            MiniMaxH3CheckpointConverter.Convert(raw, castToF32: false);

        int quantized = 0;
        foreach (KeyValuePair<string, Tensor> kv in converted.Transformer)
        {
            if (kv.Value.DType.IsFp8 || Math.Abs(kv.Value.Fp8ScaleFactor - 1.0f) > 1e-6f)
            {
                quantized++;
            }
        }
        _output.WriteLine($"{quantized} of {converted.Transformer.Count} tensors are quantized");
        Assert.True(quantized > 0,
            "the fp8 build reported no quantized tensors — the recipe's LoRA guard would then let a bad merge through");
    }

    private static unsafe float[] Sample(Tensor t)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        try
        {
            float[] samples = new float[16];
            float* p = (float*)f32.DataPointer;
            long stride = Math.Max(1, f32.ElementCount / samples.Length);
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] = p[i * stride];
            }
            return samples;
        }
        finally
        {
            if (!ReferenceEquals(f32, t))
            {
                f32.Dispose();
            }
        }
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
