using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Contract-negative controls prove score ordering and zero-output exclusions on executed CUDA rows.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class AttentionSemanticParityTests
{
    [CudaTheory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void FlashSoftcapPrecedesAlibi(bool halfKv, bool causal)
    {
        using IDisposable policy = Policy(false).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, 1, 1, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 2, 64), halfKv ? DType.F16 : DType.F32);
        using Tensor v = new(k.Shape, k.DType);
        using Tensor output = new(q.Shape, DType.F32);
        using Tensor slopes = new(new TensorShape(1), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        for (int key = 0; key < 2; key++)
            for (int c = 0; c < 64; c++) Write(v, key * 64 + c, key == 0 ? 2 : 10);
        slopes.AsSpan<float>()[0] = 4;
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        backend.FlashAttention(output, q, k, v, 2, 1, causal, 1, 1, softcap: 1, alibiSlopes: slopes);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Implementation == "flash.decode");
        double expected = (2 * Math.Exp(-4) + 10) / (Math.Exp(-4) + 1);
        foreach (float actual in output.AsReadOnlySpan<float>())
            Assert.InRange(Math.Abs(actual - expected), 0, 2e-5 * expected);
    }

    [CudaTheory]
    [InlineData("f32", false)]
    [InlineData("f16", false)]
    [InlineData("bf16", false)]
    [InlineData("f16", true)]
    [InlineData("bf16", true)]
    public void FullyExcludedRowsAreZero(string dtype, bool cudnn)
    {
        using IDisposable policy = Policy(cudnn).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        DType storage = dtype == "f16" ? DType.F16 : dtype == "bf16" ? DType.BF16 : DType.F32;
        using Tensor q = new(new TensorShape(1, 1, 2, 64), storage);
        using Tensor k = new(new TensorShape(1, 1, 3, 64), storage);
        using Tensor v = new(k.Shape, storage);
        using Tensor output = new(q.Shape, storage);
        using Tensor bias = new(new TensorShape(1, 1, 2, 3), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        for (int key = 0; key < 3; key++)
            for (int c = 0; c < 64; c++) Write(v, key * 64 + c, 7 + key * 6);
        bias.AsSpan<float>().Fill(float.NegativeInfinity);
        bias.AsSpan<float>()[4] = 0;
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        RunDense(backend, output, q, k, v, bias, cudnn);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Kind == AttentionEvidenceKind.Execution
            && e.Implementation == (cudnn ? "cudnn.f16" : "materialized"));
        for (int c = 0; c < 64; c++)
        {
            Assert.Equal(0, Read(output, c));
            Assert.Equal(13, Read(output, 64 + c));
        }
    }

    [CudaTheory]
    [InlineData(false, float.NegativeInfinity, false)]
    [InlineData(true, float.NegativeInfinity, false)]
    [InlineData(false, float.PositiveInfinity, true)]
    [InlineData(true, float.PositiveInfinity, true)]
    [InlineData(false, float.NaN, true)]
    [InlineData(true, float.NaN, true)]
    public void EmptyFlashRowsPreserveSinkSemantics(bool halfKv, float sinkValue, bool expectNaN)
    {
        using IDisposable policy = Policy(false).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, 1, 1, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 2, 64), halfKv ? DType.F16 : DType.F32);
        using Tensor v = new(k.Shape, k.DType);
        using Tensor output = new(q.Shape, DType.F32);
        using Tensor sink = new(new TensorShape(1), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        Fill(v, 7);
        sink.AsSpan<float>()[0] = sinkValue;
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        // A noncausal local window beyond the valid prefix is empty, while KV storage remains valid.
        backend.FlashAttention(output, q, k, v, 2, 1, false, 3, 1, sink: sink, slidingWindow: 1);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Implementation == "flash.decode");
        foreach (float actual in output.AsReadOnlySpan<float>())
            if (expectNaN) Assert.True(float.IsNaN(actual)); else Assert.Equal(0, actual);
    }

    [CudaTheory]
    [InlineData("f32", false)]
    [InlineData("f16", false)]
    [InlineData("bf16", false)]
    [InlineData("f16", true)]
    [InlineData("bf16", true)]
    public void BiasRetainsF32RangeAndPropagatesNonFiniteScores(string dtype, bool cudnn)
    {
        using IDisposable policy = Policy(cudnn).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        DType storage = dtype == "f16" ? DType.F16 : dtype == "bf16" ? DType.BF16 : DType.F32;
        CheckBias(backend, storage, cudnn, "cuda", cudnn ? "cudnn.f16" : "materialized");
    }

    [VulkanFact]
    public void VulkanBiasRetainsF32RangeAndPropagatesNonFiniteScores()
    {
        using IBackend backend = new VulkanBackend();
        CheckBias(backend, DType.F32, false, "vulkan", "flash.scalar");
    }

    private static void CheckBias(IBackend backend, DType storage, bool cudnn, string site, string implementation)
    {
        using Tensor q = new(new TensorShape(1, 1, 3, 64), storage);
        using Tensor k = new(new TensorShape(1, 1, 3, 64), storage);
        using Tensor v = new(k.Shape, storage);
        using Tensor output = new(q.Shape, storage);
        using Tensor bias = new(new TensorShape(3, 3), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        for (int key = 0; key < 3; key++)
            for (int c = 0; c < 64; c++) Write(v, key * 64 + c, 7 + key * 6);
        float[] values = [100_000, 100_001, float.NegativeInfinity, 0, float.NaN, float.NegativeInfinity,
            0, float.PositiveInfinity, float.NegativeInfinity];
        values.CopyTo(bias.AsSpan<float>());
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        RunDense(backend, output, q, k, v, bias, cudnn);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == site && e.Kind == AttentionEvidenceKind.Execution
            && e.Implementation == implementation);
        double expected = (7 + 13 * Math.E) / (1 + Math.E);
        double tolerance = storage == DType.F32 ? 2e-5 * expected : storage == DType.F16 ? 0.02 : 0.1;
        for (int c = 0; c < 64; c++)
        {
            Assert.InRange(Math.Abs(Read(output, c) - expected), 0, tolerance);
            Assert.True(float.IsNaN(Read(output, 64 + c)));
            Assert.True(float.IsNaN(Read(output, 128 + c)));
        }
    }

    [VulkanFact]
    public void VulkanFullyExcludedRowsAreZero()
    {
        using IBackend backend = new VulkanBackend();
        using Tensor q = new(new TensorShape(1, 1, 2, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 3, 64), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);
        using Tensor bias = new(new TensorShape(2, 3), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        for (int key = 0; key < 3; key++)
            for (int c = 0; c < 64; c++) Write(v, key * 64 + c, 7 + key * 6);
        bias.AsSpan<float>().Fill(float.NegativeInfinity);
        bias.AsSpan<float>()[4] = 0;
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        backend.ScaledDotProductAttention(output, q, k, v, bias, 1);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "vulkan" && e.Kind == AttentionEvidenceKind.Execution
            && e.Implementation == "flash.scalar");
        for (int c = 0; c < 64; c++)
        {
            Assert.Equal(0, Read(output, c));
            Assert.Equal(13, Read(output, 64 + c));
        }
    }

    [CudaTheory]
    [InlineData("f32", false)]
    [InlineData("f16", false)]
    [InlineData("bf16", false)]
    [InlineData("f16", true)]
    public void ExclusionsRespectBatchHeadAndReductionTails(string dtype, bool cudnn)
    {
        using IDisposable policy = Policy(cudnn).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        DType storage = dtype == "f16" ? DType.F16 : dtype == "bf16" ? DType.BF16 : DType.F32;
        const int heads = 2, keys = 516, rows = 3, dim = 64;
        using Tensor q = new(new TensorShape(2, heads, rows, dim), storage);
        using Tensor k = new(new TensorShape(2, heads, keys, dim), storage);
        using Tensor v = new(k.Shape, storage);
        using Tensor output = new(q.Shape, storage);
        using Tensor bias = new(new TensorShape(2, heads, rows, keys), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        bias.AsSpan<float>().Fill(float.NegativeInfinity);
        for (int bh = 0; bh < 4; bh++)
        {
            for (int key = 0; key < keys; key++)
                for (int c = 0; c < dim; c++) Write(v, (bh * keys + key) * dim + c, key % 7 + bh * 2);
            bias.AsSpan<float>()[(bh * rows + 1) * keys + bh * 17] = 0;
            bias.AsSpan<float>()[(bh * rows + 2) * keys + keys - 1] = 0;
        }
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        RunDense(backend, output, q, k, v, bias, cudnn);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Kind == AttentionEvidenceKind.Execution
            && e.Implementation == (cudnn ? "cudnn.f16" : "materialized"));
        for (int bh = 0; bh < 4; bh++)
            for (int c = 0; c < dim; c++)
            {
                Assert.Equal(0, Read(output, (bh * rows) * dim + c));
                Assert.Equal(bh * 17 % 7 + bh * 2, Read(output, (bh * rows + 1) * dim + c));
                Assert.Equal((keys - 1) % 7 + bh * 2, Read(output, (bh * rows + 2) * dim + c));
            }
    }

    private static KnobProfile Policy(bool cudnn) => KnobProfile.Create("semantic-tests")
        .With(EngineKnobs.SdpaCudnn, cudnn).With(EngineKnobs.SageAttn, false)
        .With(EngineKnobs.FlashF16, false).With(EngineKnobs.SdpaForceTiled, false)
        .With(EngineKnobs.SdpaNoTf32, true);

    private static void RunDense(IBackend backend, Tensor output, Tensor q, Tensor k, Tensor v, Tensor bias, bool cudnn)
    {
        if (cudnn && q.DType == DType.BF16)
            backend.Attention(AttentionCall.FromScaledDotProductAttention(output, q, k, v, bias, 1, false)
                with { Numerics = AttentionNumerics.Bf16Storage });
        else
            backend.ScaledDotProductAttention(output, q, k, v, bias, 1);
    }

    private static void Fill(Tensor tensor, float value)
    {
        for (int i = 0; i < tensor.ElementCount; i++) Write(tensor, i, value);
    }

    private static void Write(Tensor tensor, int index, float value)
    {
        if (tensor.DType == DType.F32) tensor.AsSpan<float>()[index] = value;
        else if (tensor.DType == DType.F16) tensor.AsSpan<Half>()[index] = (Half)value;
        else tensor.AsSpan<ushort>()[index] = (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);
    }

    private static float Read(Tensor tensor, int index) => tensor.DType == DType.F32 ? tensor.AsReadOnlySpan<float>()[index]
        : tensor.DType == DType.F16 ? (float)tensor.AsReadOnlySpan<Half>()[index]
        : BitConverter.UInt32BitsToSingle((uint)tensor.AsReadOnlySpan<ushort>()[index] << 16);
}
