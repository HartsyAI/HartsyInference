using System.Text;
using System.Linq;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>What the FastFrames recipe refuses, tested through its real load path: a safetensors generator on disk is read,
/// converted and checked by <see cref="FastFramesRecipe.LoadTransformer"/>, the same call a generation makes.</summary>
public sealed class FastFramesQuantRefusalTests : IDisposable
{
    private const string Key = "model._fsdp_wrapped_module.blocks.0.cross_attn.k.weight";
    private readonly string _dir = Directory.CreateTempSubdirectory("ff-refuse-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    /// <summary>Writes a one-tensor safetensors file with the given dtype name and raw bytes.</summary>
    private string WriteSafetensors(string dtypeName, byte[] payload)
    {
        string header = "{\"" + Key + "\":{\"dtype\":\"" + dtypeName + "\",\"shape\":[2,3],\"data_offsets\":[0," + payload.Length + "]}}";
        byte[] headerBytes = Encoding.UTF8.GetBytes(header);
        string path = Path.Combine(_dir, dtypeName + ".safetensors");
        using FileStream file = File.Create(path);
        file.Write(BitConverter.GetBytes((ulong)headerBytes.Length));
        file.Write(headerBytes);
        file.Write(payload);
        return path;
    }

    /// <summary>Writes a safetensors file from a header JSON and its raw payload.</summary>
    private string WriteRaw(string header, byte[] payload)
    {
        byte[] headerBytes = Encoding.UTF8.GetBytes(header);
        string path = Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".safetensors");
        using FileStream file = File.Create(path);
        file.Write(BitConverter.GetBytes((ulong)headerBytes.Length));
        file.Write(headerBytes);
        file.Write(payload);
        return path;
    }

    [Fact]
    public void AnFp8GeneratorIsRefusedByNameThroughTheLoadPath()
    {
        string path = WriteSafetensors("F8_E4M3", new byte[6]);
        List<IDisposable> loaders = [];
        try
        {
            NotSupportedException ex = Assert.Throws<NotSupportedException>(
                () => FastFramesRecipe.LoadTransformer(path, DType.F16, castForCpu: false, loaders));
            Assert.Contains("loads float weights only", ex.Message);
            Assert.Contains("F8_E4M3", ex.Message);
        }
        finally
        {
            foreach (IDisposable loader in loaders) loader.Dispose();
        }
    }

    [Fact]
    public void AnF16GeneratorLoadsThroughTheSamePath()
    {
        byte[] payload = new byte[6 * 2];
        string path = WriteSafetensors("F16", payload);
        List<IDisposable> loaders = [];
        try
        {
            Dictionary<string, Tensor> weights = FastFramesRecipe.LoadTransformer(path, DType.F16, castForCpu: false, loaders);
            Assert.Single(weights);
            Assert.Contains(weights.Keys, k => k.EndsWith("blocks.0.attn2.to_k.weight", StringComparison.Ordinal));
        }
        finally
        {
            foreach (IDisposable loader in loaders) loader.Dispose();
        }
    }

    [Fact]
    public void AQuantizedWeightIsRefusedByNameAndFormat()
    {
        using Tensor quant = new Tensor(new TensorShape(2, 3), DType.F32) { QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise" } };
        Dictionary<string, Tensor> weights = new() { ["blocks.1.ffn.w"] = quant };

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => FastFramesRecipe.RefuseUnsupported(weights));
        Assert.Contains("blocks.1.ffn.w", ex.Message);
        Assert.Contains("int8_tensorwise", ex.Message);
    }

    [Fact]
    public void AnInt8ComfyQuantGeneratorIsRefusedThroughTheLoadPath()
    {
        // ComfyUI's int8_tensorwise layout: the weight is I8, its per-row scale is a sibling F32 .weight_scale, and the
        // .comfy_quant companion is a U8 JSON blob that names the format. The load path must refuse it, not read it as plain floats.
        string key = "model._fsdp_wrapped_module.blocks.0.cross_attn.k";
        byte[] weight = new byte[6];
        byte[] scale = BitConverter.GetBytes(1f).Concat(BitConverter.GetBytes(1f)).ToArray();
        byte[] quant = Encoding.UTF8.GetBytes("{\"format\":\"int8_tensorwise\"}");
        string path = WriteRaw(
            "{\"" + key + ".weight\":{\"dtype\":\"I8\",\"shape\":[2,3],\"data_offsets\":[0,6]},"
            + "\"" + key + ".weight_scale\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[6,14]},"
            + "\"" + key + ".comfy_quant\":{\"dtype\":\"U8\",\"shape\":[" + quant.Length + "],\"data_offsets\":[14," + (14 + quant.Length) + "]}}",
            weight.Concat(scale).Concat(quant).ToArray());
        List<IDisposable> loaders = [];
        try
        {
            Exception ex = Record.Exception(() => FastFramesRecipe.LoadTransformer(path, DType.F16, castForCpu: false, loaders));
            Assert.NotNull(ex);
            Assert.IsType<NotSupportedException>(ex);
            // The quantized branch, not the dtype branch: the companion binds QuantInfo to the weight, so the I8 weight is named as quantized.
            Assert.Contains("does not load quantized weights", ex.Message);
            Assert.Contains("int8_tensorwise", ex.Message);
        }
        finally
        {
            foreach (IDisposable loader in loaders) loader.Dispose();
        }
    }
}
