using Xunit;
using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Comfy tags every quantized MiniMax-H3 build with the same companion suffixes
/// (<c>.weight_scale</c>/<c>.input_scale</c>/<c>.comfy_quant</c>) — only the <c>.comfy_quant</c> descriptor says which
/// format it actually is. Rejecting on the companions alone locked out <c>pruned_fp8_scaled</c>, the one variant that
/// fits a 24 GB card, with an int8-convrot error it isn't.</summary>
public unsafe class MiniMaxH3QuantGuardTests
{
    private static Tensor Descriptor(string json)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        Tensor t = new Tensor(new TensorShape(bytes.Length), DType.U8);
        bytes.CopyTo(new Span<byte>((void*)t.DataPointer, bytes.Length));
        return t;
    }

    private static Dictionary<string, Tensor> Checkpoint(string descriptorJson) => new()
    {
        ["blocks.0.attn.out_proj.weight"] = new Tensor(new TensorShape(8, 8), DType.F32),
        ["blocks.0.attn.out_proj.weight_scale"] = new Tensor(new TensorShape(1), DType.F32),
        ["blocks.0.attn.out_proj.input_scale"] = new Tensor(new TensorShape(1), DType.F32),
        ["blocks.0.attn.out_proj.comfy_quant"] = Descriptor(descriptorJson),
    };

    /// <summary>The shipped <c>minimax_h3_fl2va_pruned_fp8_scaled</c> descriptor — must pass through to the shared
    /// scale fold.</summary>
    [Fact]
    public void Fp8ScaledDescriptor_IsAccepted() =>
        MiniMaxH3CheckpointConverter.ThrowIfInt8Convrot(Checkpoint("{\"format\": \"float8_e4m3fn\"}"));

    /// <summary>The shipped <c>int8_convrot</c> descriptor — still unimplemented, still rejected.</summary>
    [Fact]
    public void Int8ConvrotDescriptor_IsRejected()
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() =>
            MiniMaxH3CheckpointConverter.ThrowIfInt8Convrot(
                Checkpoint("{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": 256}")));
        Assert.Contains("convrot", ex.Message);
    }

    /// <summary>An unreadable or absent descriptor keeps the conservative reject — better a clear error than
    /// silently feeding a rotated weight to the GEMM.</summary>
    [Fact]
    public void UnknownDescriptor_IsRejected()
    {
        Dictionary<string, Tensor> weights = Checkpoint("{\"format\": \"float8_e4m3fn\"}");
        weights.Remove("blocks.0.attn.out_proj.comfy_quant");
        Assert.Throws<NotSupportedException>(() => MiniMaxH3CheckpointConverter.ThrowIfInt8Convrot(weights));
    }

    [Fact]
    public void PlainCheckpoint_IsAccepted() =>
        MiniMaxH3CheckpointConverter.ThrowIfInt8Convrot(new Dictionary<string, Tensor>
        {
            ["blocks.0.attn.out_proj.weight"] = new Tensor(new TensorShape(8, 8), DType.F32),
        });
}
