using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Preparation a row runs before or after its kernel (casts, mask expansion, a materialized score matrix, staging
/// copies, the host reference) is recorded where it runs, with its bytes. Each case forces one route, checks which row ran,
/// and checks the records that row's preparation leaves. Where the row states its preparation for planning, the statement
/// is compared with those records step by step.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class AttentionPreparationEvidenceTests
{
    private static KnobProfile Policy(bool cudnn) => KnobProfile.Create("preparation-evidence-tests")
        .With(EngineKnobs.SdpaCudnn, cudnn).With(EngineKnobs.SageAttn, false)
        .With(EngineKnobs.FlashF16, false).With(EngineKnobs.SdpaForceTiled, false)
        .With(EngineKnobs.SdpaNoTf32, true);

    private static AttentionEvidenceEvent[] Of(AttentionEvidenceEvent[] events, AttentionEvidenceKind kind, string site) =>
        [.. events.Where(e => e.Kind == kind && e.Site == site)];

    private static string Ran(AttentionEvidenceEvent[] events) =>
        string.Join(", ", events.Where(e => e.Kind == AttentionEvidenceKind.Execution).Select(e => e.Implementation));

    private static long Bytes(Tensor tensor) => tensor.DType.ComputeByteCount(tensor.ElementCount);

    private static readonly AttentionEvidenceKind[] StatedKinds =
    [
        AttentionEvidenceKind.Cast, AttentionEvidenceKind.Repeat, AttentionEvidenceKind.Permute, AttentionEvidenceKind.MaterializeMask,
        AttentionEvidenceKind.ExpandMask, AttentionEvidenceKind.MaterializeScores, AttentionEvidenceKind.HostReference,
    ];

    /// <summary>The plan names the row that ran, the row states its preparation, and for every step the bytes it stated
    /// are the bytes recorded. The plan is taken after the run, once the backend's kernels are loaded.</summary>
    private static void AssertStatedPreparationRan(IBackend backend, in AttentionCall call, AttentionEvidenceEvent[] events)
    {
        AttentionPlan plan = backend.PlanAttention(in call);
        Assert.Equal(Ran(events), plan.RowId);
        Assert.True(plan.Preparation.HasValue, $"{plan.RowId} states no preparation.");
        foreach (AttentionEvidenceKind kind in StatedKinds)
        {
            long recorded = events.Where(e => e.Kind == kind).Sum(e => e.Bytes);
            Assert.True(plan.Preparation.Value.Of(kind) == recorded,
                $"{plan.RowId} stated {plan.Preparation.Value.Of(kind)} bytes of {kind}; {recorded} were recorded.");
        }
    }

    [CudaFact]
    public void CudaCausalPrefillOverAnF32CacheRecordsTheF16CopiesItMakes()
    {
        const int heads = 4, kvHeads = 2, rows = 24, capacity = 40, dim = 64;
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, heads, rows, dim), DType.F32);
        using Tensor k = new(new TensorShape(1, kvHeads, capacity, dim), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        AttentionCall call = AttentionCall.FromFlashAttention(output, q, k, v, rows, causal: true, qOffset: 0, scale: 0.125f, softcap: 0f,
            sink: null, slidingWindow: 0, alibiSlopes: null);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        Assert.Equal("fa2.causal", Ran(events));
        AssertStatedPreparationRan(backend, in call, events);
        // Keys and values each become a tight F16 copy of the valid rows.
        long copies = 2L * kvHeads * rows * dim * sizeof(ushort);
        Assert.Equal(copies, Assert.Single(Of(events, AttentionEvidenceKind.Cast, "cuda.fa2.kv.f16")).Bytes);
        Assert.Equal(copies, Assert.Single(Of(events, AttentionEvidenceKind.Workspace, "cuda.fa2.kv.f16")).Bytes);
    }

    [CudaFact]
    public void CudaMaterializedBf16AttentionRecordsItsCastsItsScoresAndItsOutputCast()
    {
        using IDisposable policy = Policy(cudnn: false).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, 2, 6, 64), DType.BF16);
        using Tensor k = new(new TensorShape(1, 2, 9, 64), DType.BF16);
        using Tensor v = new(k.Shape, DType.BF16);
        using Tensor output = new(q.Shape, DType.BF16);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, q, k, v, null, 0.125f, allowF16: false);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        Assert.Equal("materialized", Ran(events));
        AssertStatedPreparationRan(backend, in call, events);
        long operands = q.ElementCount + k.ElementCount + v.ElementCount;
        Assert.Equal(operands * sizeof(float), Assert.Single(Of(events, AttentionEvidenceKind.Cast, "cuda.materialized.inputs")).Bytes);
        Assert.Equal((operands + output.ElementCount) * sizeof(float), Assert.Single(Of(events, AttentionEvidenceKind.Workspace, "cuda.materialized.casts")).Bytes);
        Assert.Equal(2L * 6 * 9 * sizeof(float), Assert.Single(Of(events, AttentionEvidenceKind.MaterializeScores, "cuda.materialized.scores")).Bytes);
        Assert.Equal(Bytes(output), Assert.Single(Of(events, AttentionEvidenceKind.Cast, "cuda.materialized.output")).Bytes);
    }

    [CudaFact]
    public void CudaMaterializedAttentionRecordsTheKeyOnlyMaskItExpands()
    {
        const int heads = 2, sq = 6, skv = 9;
        using IDisposable policy = Policy(cudnn: false).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, heads, sq, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, heads, skv, 64), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);
        using Tensor mask = new(new TensorShape(1, 1, 1, skv), DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, q, k, v, mask, 0.125f, allowF16: false);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        Assert.Equal("materialized", Ran(events));
        AssertStatedPreparationRan(backend, in call, events);
        Assert.Equal((long)sq * skv * sizeof(float), Assert.Single(Of(events, AttentionEvidenceKind.ExpandMask, "cuda.materialized.mask")).Bytes);
    }

    [CudaFact]
    public void CudaCudnnCausalPrefillRecordsItsOutputCastBesideItsInputCasts()
    {
        const int heads = 2, rows = 12, dim = 64;
        using IDisposable policy = Policy(cudnn: true).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, heads, rows, dim), DType.F32);
        // Exactly the valid keys, which is what admits the masked cuDNN row.
        using Tensor k = new(new TensorShape(1, heads, rows, dim), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        AttentionCall call = AttentionCall.FromFlashAttention(output, q, k, v, rows, causal: true, qOffset: 0, scale: 0.125f, softcap: 0f,
            sink: null, slidingWindow: 0, alibiSlopes: null);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        Assert.Equal("cudnn.causal", Ran(events));
        AssertStatedPreparationRan(backend, in call, events);
        Assert.Equal((q.ElementCount + k.ElementCount + v.ElementCount) * sizeof(ushort),
            Assert.Single(Of(events, AttentionEvidenceKind.Cast, "cudnn.inputs")).Bytes);
        Assert.Equal(Bytes(output), Assert.Single(Of(events, AttentionEvidenceKind.Cast, "cudnn.output")).Bytes);
    }

    [CudaFact]
    public void CudaTokenMajorCudnnOverF32StatesAndRecordsItsCasts()
    {
        const int heads = 2, sq = 6, skv = 9, dim = 64;
        using IDisposable policy = Policy(cudnn: true).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, sq, heads, dim), DType.F32);
        using Tensor k = new(new TensorShape(1, skv, heads, dim), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        AttentionCall call = AttentionCall.FromTokenMajor(output, q, k, v, null, heads, heads, dim, 0.125f, allowF16: true);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        // The row declares no lowering, and casts its F32 operands all the same.
        Assert.Equal("cudnn.tokenmajor", Ran(events));
        AssertStatedPreparationRan(backend, in call, events);
        Assert.Equal((q.ElementCount + k.ElementCount + v.ElementCount) * sizeof(ushort) + Bytes(output),
            events.Where(e => e.Kind == AttentionEvidenceKind.Cast).Sum(e => e.Bytes));
    }

    [CudaFact]
    public void AnUnpinnedCallStagedIntoTheFlashEntryRecordsItsCopies()
    {
        const int heads = 4, sq = 1, skv = 10, dim = 32;
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, heads, sq, dim), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, skv, dim), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromFlashAttention(output, q, k, v, skv, causal: false, qOffset: 0, scale: 0.2f, softcap: 0f,
            sink: null, slidingWindow: 0, alibiSlopes: null) with { Route = AttentionRoute.Auto };

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        // Query, key and value are gathered into contiguous tensors, and the output is copied back out.
        Assert.Equal(Bytes(q) + Bytes(k) + Bytes(v), Of(events, AttentionEvidenceKind.StagingCopy, "attention.flash.lowering.gather").Sum(e => e.Bytes));
        Assert.Equal(3, Of(events, AttentionEvidenceKind.StagingCopy, "attention.flash.lowering.gather").Length);
        Assert.Equal(Bytes(output), Assert.Single(Of(events, AttentionEvidenceKind.StagingCopy, "attention.flash.lowering.output")).Bytes);
    }

    [CudaFact]
    public void AnUnpinnedCallWithANarrowerValueRecordsThePadItStages()
    {
        const int heads = 4, skv = 10, dim = 32, valueDim = 16;
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, heads, 1, dim), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, skv, dim), DType.F32);
        using Tensor v = new(new TensorShape(1, 1, skv, valueDim), DType.F32);
        using Tensor output = new(new TensorShape(1, heads, 1, valueDim), DType.F32);
        AttentionCall call = AttentionCall.FromFlashAttention(output, q, k, v, skv, causal: false, qOffset: 0, scale: 0.2f, softcap: 0f,
            sink: null, slidingWindow: 0, alibiSlopes: null) with { Route = AttentionRoute.Auto };

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        // The value is padded with zero columns to the key's width before the flash entry reads it.
        Assert.Equal((long)skv * dim * sizeof(float), Assert.Single(Of(events, AttentionEvidenceKind.StagingCopy, "attention.flash.lowering.pad")).Bytes);
        Assert.Equal(Bytes(q) + Bytes(k) + Bytes(v), Of(events, AttentionEvidenceKind.StagingCopy, "attention.flash.lowering.gather").Sum(e => e.Bytes));
        Assert.Equal(Bytes(output), Assert.Single(Of(events, AttentionEvidenceKind.StagingCopy, "attention.flash.lowering.output")).Bytes);
    }

    [VulkanFact]
    public void VulkanHostReferenceRowCountsTheSinksItReadsOnTheHost()
    {
        const int heads = 2;
        using VulkanBackend backend = new();
        using Tensor q = new(new TensorShape(1, heads, 1, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 3, 64), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);
        using Tensor sinks = new(new TensorShape(heads), DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        AttentionCall call = AttentionCall.FromFlashAttention(output, q, k, v, 3, causal: true, qOffset: 2, scale: 0.5f, softcap: 50f,
            sink: sinks, slidingWindow: 0, alibiSlopes: null);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        Assert.Equal("host.reference", Ran(events));
        AssertStatedPreparationRan(backend, in call, events);
        AttentionEvidenceEvent host = Assert.Single(Of(events, AttentionEvidenceKind.HostReference, "vulkan.host.reference"));
        Assert.Equal(Bytes(q) + Bytes(k) + Bytes(v) + Bytes(output) + Bytes(sinks), host.Bytes);
        Assert.Equal(5, host.Count);
    }

    [VulkanFact]
    public void VulkanScalarFlashOverBf16RecordsItsCastsAndItsOutputCast()
    {
        using VulkanBackend backend = new();
        // Head dim 32 keeps the call off the cooperative-matrix kernel; BF16 attention computes in F32.
        using Tensor q = new(new TensorShape(1, 2, 6, 32), DType.BF16);
        using Tensor k = new(new TensorShape(1, 2, 9, 32), DType.BF16);
        using Tensor v = new(k.Shape, DType.BF16);
        using Tensor output = new(q.Shape, DType.BF16);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, q, k, v, null, 0.2f, allowF16: false);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        Assert.Equal("flash.scalar", Ran(events));
        AssertStatedPreparationRan(backend, in call, events);
        AttentionEvidenceEvent inputs = Assert.Single(Of(events, AttentionEvidenceKind.Cast, "vulkan.flash.inputs"));
        Assert.Equal((q.ElementCount + k.ElementCount + v.ElementCount) * sizeof(float), inputs.Bytes);
        Assert.Equal(3, inputs.Count);
        Assert.Equal(Bytes(output), Assert.Single(Of(events, AttentionEvidenceKind.Cast, "vulkan.flash.output")).Bytes);
    }

    [VulkanFact]
    public void VulkanScalarFlashOverF32CastsNothingAndRecordsNoCast()
    {
        using VulkanBackend backend = new();
        using Tensor q = new(new TensorShape(1, 2, 6, 32), DType.F32);
        using Tensor k = new(new TensorShape(1, 2, 9, 32), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, q, k, v, null, 0.2f, allowF16: false);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        Assert.Equal("flash.scalar", Ran(events));
        AssertStatedPreparationRan(backend, in call, events);
        Assert.DoesNotContain(events, e => e.Kind == AttentionEvidenceKind.Cast);
        Assert.Equal(0, backend.PlanAttention(in call).Preparation!.Value.Total);
    }

    [VulkanFact]
    public void VulkanHostReferenceRowRecordsTheTensorsThatCrossToTheHost()
    {
        using VulkanBackend backend = new();
        using Tensor q = new(new TensorShape(1, 2, 3, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 3, 64), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        // Soft-cap has no Vulkan kernel: the declared host row serves it.
        AttentionCall call = AttentionCall.FromFlashAttention(output, q, k, v, 3, causal: true, qOffset: 0, scale: 0.5f, softcap: 50f,
            sink: null, slidingWindow: 0, alibiSlopes: null);
        backend.Attention(in call);
        backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();

        Assert.Equal("host.reference", Ran(events));
        AssertStatedPreparationRan(backend, in call, events);
        AttentionEvidenceEvent host = Assert.Single(Of(events, AttentionEvidenceKind.HostReference, "vulkan.host.reference"));
        Assert.Equal(Bytes(q) + Bytes(k) + Bytes(v) + Bytes(output), host.Bytes);
    }
}
