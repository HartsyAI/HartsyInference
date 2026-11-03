using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A Qwen3.5 encoder saved as ComfyUI writes its int8 builds (<c>qwen3.5_9b_int8_convrot</c>, and what the w4a8
/// pass turns <c>qwen3.5_9b_w4a8_gptq</c> into): the token embedding and every trunk linear <c>int8_tensorwise</c> with
/// ConvRot, each with its <c>.weight_scale</c> and <c>.comfy_quant</c>. On a backend with no int8 path the encoder gathers
/// embedding rows from the packed table and widens the linears, which must reproduce the same int8 weights widened to BF16
/// before saving, bit for bit.</summary>
public sealed class Qwen35QuantizedEncoderTests : IDisposable
{
    private const int ConvRotGroup = 16;
    private static readonly string Fixture = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Qwen35Vl", "encoder");
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-qwen35-int8-" + Guid.NewGuid().ToString("N"));

    [Fact]
    public void Int8ConvRotEncoder_OnCpu_MatchesItsWeightsWidenedToBf16()
    {
        string int8Dir = Path.Combine(_root, "int8"), wideDir = Path.Combine(_root, "bf16");
        WriteBuilds(int8Dir, wideDir);
        int[] ids = [3, 17, 42, 5, 79, 0, 11, 64, 23];

        using CpuBackend backend = new();
        using Qwen35TextEncoder quantized = Qwen35TextEncoder.Load(int8Dir, 64, "test", backend);
        using Qwen35TextEncoder widened = Qwen35TextEncoder.Load(wideDir, 64, "test", backend);
        using Tensor got = quantized.Encode(backend, ids);
        using Tensor want = widened.Encode(backend, ids);

        Assert.Equal(want.Shape, got.Shape);
        Assert.Equal(want.AsReadOnlySpan<float>().ToArray(), got.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Int8ConvRotEncoder_RefusesWithoutABackendThatCanRunIt()
    {
        string int8Dir = Path.Combine(_root, "int8"), wideDir = Path.Combine(_root, "bf16");
        WriteBuilds(int8Dir, wideDir);

        // Without a backend nothing is widened, and the CPU Linear has no int8_tensorwise path.
        using CpuBackend backend = new();
        using Qwen35TextEncoder unprepared = Qwen35TextEncoder.Load(int8Dir, 64, "test");
        Assert.ThrowsAny<Exception>(() => unprepared.Encode(backend, [1, 2, 3]).Dispose());
    }

    /// <summary>Writes the fixture twice: once as ComfyUI's int8 build, once with those same int8 weights widened to BF16.</summary>
    private static void WriteBuilds(string int8Dir, string wideDir)
    {
        Directory.CreateDirectory(int8Dir);
        Directory.CreateDirectory(wideDir);
        File.Copy(Path.Combine(Fixture, "config.json"), Path.Combine(int8Dir, "config.json"));
        File.Copy(Path.Combine(Fixture, "config.json"), Path.Combine(wideDir, "config.json"));

        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Fixture, "model.safetensors"));
        Dictionary<string, Tensor> int8 = new(StringComparer.Ordinal), wide = new(StringComparer.Ordinal);
        List<Tensor> created = [];
        try
        {
            foreach (string key in loader.Descriptors.Keys)
            {
                Tensor source = loader.GetTensor(key);
                if (!IsQuantizedInComfyBuilds(key, source))
                {
                    int8[key] = source;
                    wide[key] = source;
                    continue;
                }
                // QuantizeFromF32 rotates its input in place, so it gets a copy rather than the loader's tensor.
                using Tensor values = new Tensor(source.Shape, DType.F32);
                source.AsReadOnlySpan<float>().CopyTo(values.AsSpan<float>());
                (Tensor weight, Tensor rowScale) = Int8ConvRotCodec.QuantizeFromF32(values, ConvRotGroup);
                created.Add(weight);
                created.Add(rowScale);
                string baseKey = key[..^".weight".Length];
                Tensor shapedScale = rowScale.Reshape(new TensorShape(rowScale.ElementCount, 1));
                Tensor descriptor = Descriptor();
                Tensor bf16 = Int8ConvRotCodec.DequantToBf16(weight, rowScale, ConvRotGroup);
                created.AddRange([shapedScale, descriptor, bf16]);
                int8[key] = weight;
                int8[baseKey + ".weight_scale"] = shapedScale;
                int8[baseKey + ComfyQuantDescriptor.Suffix] = descriptor;
                wide[key] = bf16;
            }
            SafeTensorsWriter.Save(Path.Combine(int8Dir, "model.safetensors"), int8);
            SafeTensorsWriter.Save(Path.Combine(wideDir, "model.safetensors"), wide);
        }
        finally
        {
            foreach (Tensor tensor in created)
                tensor.Dispose();
        }
    }

    /// <summary>The token embedding and the trunk's matrices, as Comfy-Org's Qwen3.5 int8 build quantizes them.</summary>
    private static bool IsQuantizedInComfyBuilds(string key, Tensor tensor) =>
        tensor.Shape.Rank == 2 && key.EndsWith(".weight", StringComparison.Ordinal)
        && (key == "model.language_model.embed_tokens.weight" || key.StartsWith("model.language_model.layers.", StringComparison.Ordinal));

    private static Tensor Descriptor()
    {
        byte[] json = Encoding.UTF8.GetBytes($"{{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": {ConvRotGroup}}}");
        Tensor blob = new Tensor(new TensorShape(json.Length), DType.U8);
        json.CopyTo(blob.AsSpan<byte>());
        return blob;
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (IOException)
        {
            // A memory map the runtime has not released yet; the temp folder is reclaimed later.
        }
    }
}
