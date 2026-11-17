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
            AttentionCall legacyCall = AttentionCall.FromScaledDotProductAttention(legacy, q, k, v, null, 0.5f, false);
            AttentionCall directCall = AttentionCall.FromScaledDotProductAttention(direct, q, k, v, null, 0.5f, false);

            Assert.Equal(backend.PlanAttention(in legacyCall).RowId, backend.PlanAttention(in directCall).RowId);
            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(8);
            backend.ScaledDotProductAttention(legacy, q, k, v, null, 0.5f);
            backend.Attention(in directCall);
            backend.Sync();

            AssertSameExecution(evidence.Snapshot(), site);
            Assert.True(legacy.AsReadOnlySpan<float>().SequenceEqual(direct.AsReadOnlySpan<float>()));
        }
    }

    private static void Flash(IBackend backend, string site)
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
            AttentionCall legacyCall = AttentionCall.FromFlashAttention(legacy, q, k, v, 3, false, 2, 0.5f, 0f, null, 0, null);
            AttentionCall directCall = AttentionCall.FromFlashAttention(direct, q, k, v, 3, false, 2, 0.5f, 0f, null, 0, null);

            Assert.Equal(backend.PlanAttention(in legacyCall).RowId, backend.PlanAttention(in directCall).RowId);
            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(8);
            backend.FlashAttention(legacy, q, k, v, 3, 2, false, 2, 0.5f);
            backend.Attention(in directCall);
            backend.Sync();

            AssertSameExecution(evidence.Snapshot(), site);
            Assert.True(legacy.AsReadOnlySpan<float>().SequenceEqual(direct.AsReadOnlySpan<float>()));
        }
    }

    private static void AssertSameExecution(AttentionEvidenceEvent[] events, string site)
    {
        AttentionEvidenceEvent[] executions = Array.FindAll(events,
            e => e.Site == site && e.Kind == AttentionEvidenceKind.Execution);
        Assert.Equal(2, executions.Length);
        Assert.Equal(executions[0].Implementation, executions[1].Implementation);
        Assert.Equal(executions[0].Branch, executions[1].Branch);
        Assert.NotEqual("host.reference", executions[0].Implementation);
    }

    private static void Fill(Tensor tensor, float step)
    {
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (i - values.Length / 2f) * step;
    }
}
