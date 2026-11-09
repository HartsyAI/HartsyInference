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
        // 2 heads of width 4 padded to 6: q and k pad each half (2 -> 3), v pads each head (4 -> 6). A kept row r holds r + 1; padding is 0.
        const int Heads = 2, HeadDim = 4, Cols = 3;
        using Tensor padded = F32(PaddedRows(Cols), new TensorShape(36, Cols));

        using Tensor trimmed = MossVlVisionTower.UnpadQkv(padded, Heads, HeadDim);

        Assert.Equal(new TensorShape(3 * Heads * HeadDim, Cols), trimmed.Shape);
        float[] kept = trimmed.AsSpan<float>().ToArray().Where((_, i) => i % Cols == 0).Select(v => v - 1).ToArray();
        Assert.Equal([0f, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 26, 27, 30, 31, 32, 33], kept);
    }

    [Fact]
    public void UnpadQkv_RefusesPaddingThatIsNotZero()
    {
        // A converter that padded some other way leaves data where this layout expects zeros: refuse rather than load wrong features.
        float[] rows = PaddedRows(1);
        rows[2] = 7f;   // row 2 is padding of q's first rotary half
        using Tensor odd = F32(rows, new TensorShape(36));
        Assert.Throws<InvalidDataException>(() => MossVlVisionTower.UnpadQkv(odd, heads: 2, headDim: 4));
    }

    /// <summary>36 padded qkv rows (2 heads, width 4 padded to 6) of <paramref name="cols"/> columns: kept row r holds r + 1, padding 0.</summary>
    private static float[] PaddedRows(int cols)
    {
        float[] data = new float[36 * cols];
        for (int r = 0; r < 36; r++)
        {
            int i = r % 12;
            bool kept = r < 24 ? i % 3 < 2 : i % 6 < 4;   // q, k: 2 of every 3 rows; v: 4 of every 6
            for (int c = 0; c < cols; c++) data[r * cols + c] = kept ? r + 1 : 0f;
        }
        return data;
    }

    [Fact]
    public void UnpadQkv_HandlesTheBiasAndLeavesAnUnpaddedTensorAlone()
    {
        using Tensor bias = F32(PaddedRows(1).Select(v => v == 0 ? 0f : v - 1).ToArray(), new TensorShape(36));
        using Tensor trimmed = MossVlVisionTower.UnpadQkv(bias, heads: 2, headDim: 4);
        Assert.Equal(new TensorShape(24), trimmed.Shape);
        Assert.Equal(27f, trimmed.AsSpan<float>().ToArray()[19]);

        using Tensor plain = F32(new float[24], new TensorShape(24));
        Assert.Same(plain, MossVlVisionTower.UnpadQkv(plain, heads: 2, headDim: 4));
        using Tensor bad = F32(new float[25], new TensorShape(25));
        Assert.Throws<InvalidDataException>(() => MossVlVisionTower.UnpadQkv(bad, 2, 4));
    }

    /// <summary>The released MOSS-VL GGUF embeds no tokenizer, so a folder holding only a GGUF is refused with the file to add and the
    /// repository it comes from, before any weight is read.</summary>
    [Fact]
    public void LoneGgufWithoutTokenizerNamesTheMissingFile()
    {
        string folder = Path.Combine(Path.GetTempPath(), "hartsy-moss-lone-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        string gguf = Path.Combine(folder, "moss.gguf");
        try
        {
            using (HartsyInference.ModelAssets.Gguf.GgufWriter writer = new(gguf))
            {
                writer.SetMetadata("general.architecture", "moss_vl");
                using Tensor marker = F32([0f, 0f], new TensorShape(2));
                writer.AddTensor("blk.0.cross_gate_attn.weight", marker);
                writer.Flush();
            }
            Assert.True(MossVlModel.IsMossVl(gguf));
            HartsyInference.Core.Exceptions.HartsyInferenceException error = Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(
                () => MossVlModel.Open(gguf, new MossVlLoadOptions()));
            Assert.Contains("tokenizer.json", error.Message);
            Assert.Contains(MossVlModel.TokenizerRepo, error.Message);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
