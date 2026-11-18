using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Checks that public legacy entries and direct calls select the same device row and produce identical output.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class AttentionExecutionParityTests
{
    [CudaFact]
    public void CudaDenseLegacyAndDirectEntriesAgree() => Dense(new CudaBackend(0, CudaPtxDirectory.Resolve()), "cuda");

    [VulkanFact]
    public void VulkanDenseLegacyAndDirectEntriesAgree() => Dense(new VulkanBackend(), "vulkan");

    [CudaFact]
    public void CudaFlashLegacyAndDirectEntriesAgree() => Flash(new CudaBackend(0, CudaPtxDirectory.Resolve()), "cuda");

    [VulkanFact]
    public void VulkanFlashLegacyAndDirectEntriesAgree() => Flash(new VulkanBackend(), "vulkan");

    [CudaFact]
    public void CudaUnboundedDirectFlashUsesFullPrefix() => Flash(new CudaBackend(0, CudaPtxDirectory.Resolve()), "cuda", unbounded: true);

    [VulkanFact]
    public void VulkanUnboundedDirectFlashUsesFullPrefix() => Flash(new VulkanBackend(), "vulkan", unbounded: true);

    [CudaFact]
    public void CudaTokenMajorLegacyAndDirectEntriesAgree() => TokenMajor(new CudaBackend(0, CudaPtxDirectory.Resolve()), "cuda", grouped: false);

    [VulkanFact]
    public void VulkanTokenMajorLegacyAndDirectEntriesAgree() => TokenMajor(new VulkanBackend(), "vulkan", grouped: false);

    [CudaFact]
    public void CudaTokenMajorGqaLegacyAndDirectEntriesAgree() => TokenMajor(new CudaBackend(0, CudaPtxDirectory.Resolve()), "cuda", grouped: true);

    [VulkanFact]
    public void VulkanTokenMajorGqaLegacyAndDirectEntriesAgree() => TokenMajor(new VulkanBackend(), "vulkan", grouped: true);

    [VulkanFact]
    public void VulkanRank4TokenMajorLegacyKeepsSingleSequenceView()
    {
        using VulkanBackend backend = new();
        using Tensor q = new(new TensorShape(2, 1, 2, 64), DType.F32);
        using Tensor k = new(q.Shape, DType.F32);
        using Tensor v = new(q.Shape, DType.F32);
        using Tensor legacy = new(q.Shape, DType.F32);
        using Tensor direct = new(q.Shape, DType.F32);
        Fill(q, 0.01f);
        Fill(k, -0.02f);
        Fill(v, 0.1f);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(16);
        backend.ScaledDotProductAttentionTokenMajor(legacy, q, k, v, null, 2, 64, 0.5f);
        AttentionCall directCall = AttentionCall.FromTokenMajor(direct, q, k, v, null, 2, 2, 64, 0.5f, false, batch: 1);
        backend.Attention(in directCall);
        backend.Sync();

        AttentionEvidenceEvent[] executions = AssertSameExecution(evidence.Snapshot(), "vulkan");
        Assert.All(executions, e =>
        {
            Assert.Equal(1, e.Batch);
            Assert.Equal(2, e.QueryTokens);
        });
        Assert.True(legacy.AsReadOnlySpan<float>().SequenceEqual(direct.AsReadOnlySpan<float>()));
    }

    [CudaFact]
    public void CudaFlashDeviceLegacyAndDirectEntriesAgree() => FlashDevice(new CudaBackend(0, CudaPtxDirectory.Resolve()), "cuda");

    [VulkanFact]
    public void VulkanFlashDeviceLegacyAndDirectEntriesAgree() => FlashDevice(new VulkanBackend(), "vulkan");

    [CudaFact]
    public void CudaFlashDeviceFallbackPreservesDirectNumerics()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, 2, 1, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 4, 64), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);
        Fill(q, 0.01f);
        Fill(k, -0.02f);
        Fill(v, 0.1f);
        AttentionCall call = AttentionCall.FromFlashAttentionDev(output, q, k, v, 4, true, 3, 0.5f, 0, 0f, 0)
            with { Numerics = AttentionNumerics.Exact with { Refused = AttentionFormats.Tf32 } };

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(8);
        backend.Attention(in call);
        backend.Sync();

        AttentionEvidenceEvent[] executions = Array.FindAll(evidence.Snapshot(),
            e => e.Site == "cuda" && e.Kind == AttentionEvidenceKind.Execution);
        Assert.NotEmpty(executions);
        Assert.All(executions, e =>
        {
            Assert.Equal(AttentionRoute.Flash, e.Route);
            Assert.Equal(AttentionFormats.Tf32, e.Numerics.Refused);
        });
    }

    [CudaFact]
    public void CudaMalformedLegacyAndDirectCallsLeaveOutputUntouched() =>
        MalformedCallsLeaveOutputUntouched(new CudaBackend(0, CudaPtxDirectory.Resolve()));

    [VulkanFact]
    public void VulkanMalformedLegacyAndDirectCallsLeaveOutputUntouched() =>
        MalformedCallsLeaveOutputUntouched(new VulkanBackend());

    [CudaFact]
    public void CudaUnsupportedFlashRefusesHostFallback() =>
        UnsupportedFlashRefusesHostFallback(new CudaBackend(0, CudaPtxDirectory.Resolve()), "cuda", headDim: 1025, softcap: 0f);

    [VulkanFact]
    public void VulkanUnsupportedFlashRefusesHostFallback() =>
        UnsupportedFlashRefusesHostFallback(new VulkanBackend(), "vulkan", headDim: 64, softcap: 1f);

    private static void Dense(IBackend backend, string site)
    {
        using (backend)
        {
            using Tensor q = new(new TensorShape(1, 2, 2, 64), DType.F32);
            using Tensor k = new(new TensorShape(1, 2, 3, 64), DType.F32);
            using Tensor v = new(k.Shape, DType.F32);
            using Tensor legacy = new(q.Shape, DType.F32);
            using Tensor direct = new(q.Shape, DType.F32);
            Fill(q, 0.01f);
            Fill(k, -0.02f);
            Fill(v, 0.1f);

            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(8);
            backend.ScaledDotProductAttention(legacy, q, k, v, null, 0.5f);
            AttentionCall directCall = AttentionCall.FromScaledDotProductAttention(direct, q, k, v, null, 0.5f, false);
            backend.Attention(in directCall);
            backend.Sync();

            AssertSameExecution(evidence.Snapshot(), site);
            Assert.True(legacy.AsReadOnlySpan<float>().SequenceEqual(direct.AsReadOnlySpan<float>()));
        }
    }

    private static void Flash(IBackend backend, string site, bool unbounded = false)
    {
        using (backend)
        {
            using Tensor q = new(new TensorShape(1, 2, 1, 64), DType.F32);
            using Tensor k = new(new TensorShape(1, 1, 4, 64), DType.F32);
            using Tensor v = new(k.Shape, DType.F32);
            using Tensor legacy = new(q.Shape, DType.F32);
            using Tensor direct = new(q.Shape, DType.F32);
            Fill(q, 0.01f);
            Fill(k, -0.02f);
            Fill(v, 0.1f);

            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(8);
            backend.FlashAttention(legacy, q, k, v, 4, 2, false, 2, 0.5f);
            AttentionCall directCall = AttentionCall.FromFlashAttention(direct, q, k, v, 4, false, 2, 0.5f, 0f, null, 0, null);
            if (unbounded) directCall = directCall with { Positions = new AttentionPositions(0, 2, 0) };
            backend.Attention(in directCall);
            backend.Sync();

            AttentionEvidenceEvent[] executions = AssertSameExecution(evidence.Snapshot(), site);
            Assert.All(executions, e => Assert.Equal(4, e.ValidKeys));
            Assert.True(legacy.AsReadOnlySpan<float>().SequenceEqual(direct.AsReadOnlySpan<float>()));
        }
    }

    private static void TokenMajor(IBackend backend, string site, bool grouped)
    {
        using (backend)
        {
            int kvHeads = grouped ? 1 : 2;
            using Tensor q = new(new TensorShape(1, 2, 2, 64), DType.F32);
            using Tensor k = new(new TensorShape(1, 2, kvHeads, 64), DType.F32);
            using Tensor v = new(k.Shape, DType.F32);
            using Tensor legacy = new(q.Shape, DType.F32);
            using Tensor direct = new(q.Shape, DType.F32);
            Fill(q, 0.01f);
            Fill(k, -0.02f);
            Fill(v, 0.1f);

            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(128);
            if (grouped) backend.ScaledDotProductAttentionTokenMajor(legacy, q, k, v, null, 2, kvHeads, 64, 0.5f);
            else backend.ScaledDotProductAttentionTokenMajor(legacy, q, k, v, null, 2, 64, 0.5f);
            AttentionCall directCall = AttentionCall.FromTokenMajor(direct, q, k, v, null, 2, kvHeads, 64, 0.5f, false);
            if (grouped) directCall = directCall with { Route = AttentionRoute.TokenMajorGqa };
            backend.Attention(in directCall);
            backend.Sync();

            AssertSameExecution(evidence.Snapshot(), site);
            Assert.True(legacy.AsReadOnlySpan<float>().SequenceEqual(direct.AsReadOnlySpan<float>()));
        }
    }

    private static void FlashDevice(IBackend backend, string site)
    {
        using (backend)
        {
            using Tensor q = new(new TensorShape(1, 2, 1, 64), DType.F32);
            using Tensor k = new(new TensorShape(1, 1, 4, 64), DType.F32);
            using Tensor v = new(k.Shape, DType.F32);
            using Tensor legacy = new(q.Shape, DType.F32);
            using Tensor direct = new(q.Shape, DType.F32);
            Fill(q, 0.01f);
            Fill(k, -0.02f);
            Fill(v, 0.1f);
            ulong devicePos = backend.AllocDevicePos();
            try
            {
                backend.WriteDevicePos(devicePos, 4, 3);
                using AttentionEvidenceSession evidence = AttentionEvidence.Begin(8);
                backend.FlashAttentionDev(legacy, q, k, v, 0, 2, true, 0, 0.5f, devicePos);
                AttentionCall directCall = AttentionCall.FromFlashAttentionDev(direct, q, k, v, 0, true, 0, 0.5f, devicePos, 0f, 0);
                backend.Attention(in directCall);
                backend.Sync();

                AssertSameExecution(evidence.Snapshot(), site);
                Assert.True(legacy.AsReadOnlySpan<float>().SequenceEqual(direct.AsReadOnlySpan<float>()));
            }
            finally
            {
                backend.FreeDevicePos(devicePos);
            }
        }
    }

    private static void UnsupportedFlashRefusesHostFallback(IBackend backend, string site, int headDim, float softcap)
    {
        using (backend)
        {
            using Tensor q = new(new TensorShape(1, 1, 1, headDim), DType.F32);
            using Tensor k = new(q.Shape, DType.F32);
            using Tensor v = new(q.Shape, DType.F32);
            using Tensor output = new(q.Shape, DType.F32);
            Fill(q, 0.01f);
            Fill(k, -0.02f);
            Fill(v, 0.1f);
            output.AsSpan<float>().Fill(7f);
            AttentionCall call = AttentionCall.FromFlashAttention(output, q, k, v, 1, false, 0, 0.5f, softcap, null, 0, null);

            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(8);
            Assert.Null(backend.PlanAttention(in call).Row);
            Assert.Throws<AttentionUnsupportedException>(() => backend.FlashAttention(output, q, k, v, 1, 1, false, 0, 0.5f, softcap));
            Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in call));
            Assert.DoesNotContain(evidence.Snapshot(), e => e.Site == site && e.Kind == AttentionEvidenceKind.Execution);
            Assert.All(output.AsReadOnlySpan<float>().ToArray(), value => Assert.Equal(7f, value));

            AttentionHostReference.Run(in call);
            Assert.All(output.AsReadOnlySpan<float>().ToArray(), value => Assert.NotEqual(7f, value));
        }
    }

    private static void MalformedCallsLeaveOutputUntouched(IBackend backend)
    {
        using (backend)
        {
            using Tensor q = new(new TensorShape(1, 1, 1, 64), DType.F32);
            using Tensor k = new(q.Shape, DType.F32);
            using Tensor v = new(q.Shape, DType.F32);
            using Tensor legacy = new(q.Shape, DType.F32);
            using Tensor direct = new(q.Shape, DType.F32);
            legacy.AsSpan<float>().Fill(7f);
            direct.AsSpan<float>().Fill(7f);

            Assert.Throws<ArgumentOutOfRangeException>(() => backend.ScaledDotProductAttention(legacy, q, k, v, null, float.NaN));
            AttentionCall call = AttentionCall.FromScaledDotProductAttention(direct, q, k, v, null, float.NaN, false);
            Assert.Throws<ArgumentOutOfRangeException>(() => backend.Attention(in call));

            Assert.All(legacy.AsReadOnlySpan<float>().ToArray(), value => Assert.Equal(7f, value));
            Assert.All(direct.AsReadOnlySpan<float>().ToArray(), value => Assert.Equal(7f, value));
        }
    }

    private static AttentionEvidenceEvent[] AssertSameExecution(AttentionEvidenceEvent[] events, string site)
    {
        AttentionEvidenceEvent[] executions = Array.FindAll(events,
            e => e.Site == site && e.Kind == AttentionEvidenceKind.Execution);
        Assert.NotEmpty(executions);
        Assert.Equal(0, executions.Length % 2);
        int perCall = executions.Length / 2;
        for (int i = 0; i < perCall; i++)
        {
            Assert.Equal(executions[i].Implementation, executions[i + perCall].Implementation);
            Assert.Equal(executions[i].Branch, executions[i + perCall].Branch);
        }
        Assert.All(executions, e => Assert.NotEqual("host.reference", e.Implementation));
        return executions;
    }

    private static void Fill(Tensor tensor, float step)
    {
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (i - values.Length / 2f) * step;
    }
}
