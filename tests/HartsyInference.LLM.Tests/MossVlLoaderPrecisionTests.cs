using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.LLM.Models.MossVl;

namespace HartsyInference.LLM.Tests;

/// <summary>Which storage each MOSS-VL weight keeps on each kind of backend, and how a GGUF's padded vision qkv is un-padded.
/// No weights or GPU: a proxy backend answers only <see cref="IBackend.Capabilities"/>.</summary>
public sealed class MossVlLoaderPrecisionTests
{
    /// <summary>An <see cref="IBackend"/> that reports the given capabilities and refuses every other call.</summary>
    public class CapabilityBackend : DispatchProxy
    {
        public BackendCapabilities Capabilities { get; set; } = new() { Name = "stub" };

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) =>
            targetMethod?.Name == "get_Capabilities" ? Capabilities
                : throw new NotSupportedException($"The capability stub has no {targetMethod?.Name}.");

        public static IBackend Create(bool bf16)
        {
            IBackend backend = Create<IBackend, CapabilityBackend>();
            ((CapabilityBackend)(object)backend).Capabilities = new BackendCapabilities { Name = "stub", SupportsBF16 = bf16, SupportsF16 = true };
            return backend;
        }
    }

    private static Tensor F32(float[] values, TensorShape shape)
    {
        Tensor t = new(shape, DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor Bf16(params float[] values)
    {
        using Tensor f32 = F32(values, new TensorShape(1, values.Length));
        return f32.CastTo(DType.BF16);
    }

    [Fact]
    public void DenseCast_CpuOrNoBackendWidensToF32()
    {
        using Tensor w = Bf16(1f, -2f);
        using Tensor host = DenseWeightCasts.ForBackend(w, null, "w");
        using CpuBackend cpu = new();
        using Tensor onCpu = DenseWeightCasts.ForBackend(w, cpu, "w");
        Assert.Equal(DType.F32, host.DType);
        Assert.Equal(DType.F32, onCpu.DType);
    }

    [Fact]
    public void DenseCast_Bf16BackendKeepsTheWeight()
    {
        using Tensor w = Bf16(1f, -2f);
        Assert.Same(w, DenseWeightCasts.ForBackend(w, CapabilityBackend.Create(bf16: true), "w"));
    }

    [Fact]
    public void DenseCast_NoBf16BackendCastsToF16Exactly()
    {
        using Tensor w = Bf16(1.5f, -3f, 1024f);
        using Tensor f16 = DenseWeightCasts.ForBackend(w, CapabilityBackend.Create(bf16: false), "w");
        Assert.Equal(DType.F16, f16.DType);
        using Tensor back = f16.CastTo(DType.F32);
        Assert.Equal([1.5f, -3f, 1024f], back.AsSpan<float>().ToArray());
    }

    [Fact]
    public void DenseCast_RefusesBf16ThatOverflowsF16ByName()
    {
        using Tensor w = Bf16(1f, 70000f);
        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => DenseWeightCasts.ForBackend(w, CapabilityBackend.Create(bf16: false), "layers.3.mlp.down_proj.weight"));
        Assert.Contains("layers.3.mlp.down_proj.weight", error.Message);
    }

    [Fact]
    public void DecoderStore_ProjectionsFollowTheBackendAndTheRestIsF32()
    {
        IBackend vulkanLike = CapabilityBackend.Create(bf16: false);
        const string L = "model.language_model.layers.2.";
        using Tensor proj = Bf16(1f, 2f);
        using Tensor normF32 = F32([1f, 2f], new TensorShape(2));
        using Tensor norm = normF32.CastTo(DType.BF16);

        using Tensor storedProj = MossVlTextDecoder.Store(L + "cross_attn.q_proj.weight", proj, vulkanLike, device: true, keepQuantized: false);
        using Tensor storedHead = MossVlTextDecoder.Store("lm_head.weight", proj, vulkanLike, device: true, keepQuantized: false);
        using Tensor storedEmbed = MossVlTextDecoder.Store("model.language_model.embed_tokens.weight", proj, vulkanLike, device: true, keepQuantized: false);
        using Tensor storedNorm = MossVlTextDecoder.Store(L + "cross_attn.q_norm.weight", norm, vulkanLike, device: true, keepQuantized: false);
        using Tensor storedGate = MossVlTextDecoder.Store(L + "cross_attn_attn_gate", norm, vulkanLike, device: true, keepQuantized: false);

        Assert.Equal(DType.F16, storedProj.DType);
        Assert.Equal(DType.F16, storedHead.DType);
        Assert.Equal(DType.F32, storedEmbed.DType);   // host gather
        Assert.Equal(DType.F32, storedNorm.DType);
        Assert.Equal(DType.F32, storedGate.DType);
        Assert.Same(proj, MossVlTextDecoder.Store(L + "mlp.up_proj.weight", proj, CapabilityBackend.Create(bf16: true), device: true, keepQuantized: false));
    }

    [Fact]
    public void DecoderStore_PackedProjectionsStayPackedOnADeviceAndWidenOnCpu()
    {
        using Tensor f32 = F32(Enumerable.Range(0, 64).Select(i => i * 0.01f).ToArray(), new TensorShape(2, 32));
        using Tensor q8 = HartsyInference.ModelAssets.Gguf.GgufQuantizer.Quantize(f32, DType.Q8_0);
        const string Key = "model.language_model.layers.0.self_attn.q_proj.weight";

        Assert.Same(q8, MossVlTextDecoder.Store(Key, q8, CapabilityBackend.Create(bf16: false), device: true, keepQuantized: false));
        using CpuBackend cpu = new();
        using Tensor onCpu = MossVlTextDecoder.Store(Key, q8, cpu, device: false, keepQuantized: false);
        Assert.Equal(DType.F32, onCpu.DType);
        Assert.Same(q8, MossVlTextDecoder.Store(Key, q8, null, device: false, keepQuantized: true));
        using Tensor legacy = MossVlTextDecoder.Store(Key, q8, null, device: false, keepQuantized: false);
        Assert.Equal(DType.F32, legacy.DType);
    }

    [Fact]
    public void UnpadQkv_DropsTheRotaryHalfPadOfQkAndTheHeadPadOfV()
    {
        // 2 heads of width 4 padded to 6: q and k pad each half (2 -> 3), v pads each head (4 -> 6). Row r holds the value r.
        const int Heads = 2, HeadDim = 4, Padded = 6, Cols = 3;
        int rows = 3 * Heads * Padded;
        float[] data = new float[rows * Cols];
        for (int r = 0; r < rows; r++) for (int c = 0; c < Cols; c++) data[r * Cols + c] = r;
        using Tensor padded = F32(data, new TensorShape(rows, Cols));

        using Tensor trimmed = MossVlVisionTower.UnpadQkv(padded, Heads, HeadDim);

        Assert.Equal(new TensorShape(3 * Heads * HeadDim, Cols), trimmed.Shape);
        float[] kept = trimmed.AsSpan<float>().ToArray().Where((_, i) => i % Cols == 0).ToArray();
        Assert.Equal([0f, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 26, 27, 30, 31, 32, 33], kept);
    }

    [Fact]
    public void UnpadQkv_HandlesTheBiasAndLeavesAnUnpaddedTensorAlone()
    {
        using Tensor bias = F32(Enumerable.Range(0, 36).Select(i => (float)i).ToArray(), new TensorShape(36));
        using Tensor trimmed = MossVlVisionTower.UnpadQkv(bias, heads: 2, headDim: 4);
        Assert.Equal(new TensorShape(24), trimmed.Shape);
        Assert.Equal(27f, trimmed.AsSpan<float>().ToArray()[19]);

        using Tensor plain = F32(new float[24], new TensorShape(24));
        Assert.Same(plain, MossVlVisionTower.UnpadQkv(plain, heads: 2, headDim: 4));
        Assert.Throws<InvalidDataException>(() => MossVlVisionTower.UnpadQkv(F32(new float[25], new TensorShape(25)), 2, 4));
    }
}
