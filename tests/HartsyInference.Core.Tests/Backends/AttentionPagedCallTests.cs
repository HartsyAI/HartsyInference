using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>A paged call's key operand is a page buffer. Only a row that declares paged keys and values may read it, a
/// backend without such a row refuses the call, and a malformed page table is rejected before anything is written.</summary>
public sealed class AttentionPagedCallTests
{
    private const int Heads = 4, KvHeads = 2, HeadDim = 8, PageSize = 4, PoolPages = 6, ValidKeys = 10;

    /// <summary>One decode row over three pages of a six-page pool.</summary>
    private sealed class Operands : IDisposable
    {
        public readonly Tensor Query = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
        public readonly Tensor Output = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
        public readonly Tensor KeyPages = new(new TensorShape(1, KvHeads, PoolPages * PageSize, HeadDim), DType.F32);
        public readonly Tensor ValuePages = new(new TensorShape(1, KvHeads, PoolPages * PageSize, HeadDim), DType.F32);
        public readonly Tensor Table = new(new TensorShape(3), DType.I32);

        public Operands()
        {
            Output.AsSpan<float>().Fill(-123f);
            ReadOnlySpan<int> pages = [4, 1, 5];
            pages.CopyTo(Table.AsSpan<int>());
        }

        public AttentionCall Call(AttentionKvPages? pages = null, int validKeys = ValidKeys) =>
            AttentionCall.FromPagedFlashAttention(Output, Query, KeyPages, ValuePages, pages ?? new AttentionKvPages(Table, PageSize, 5),
                validKeys, causal: true, qOffset: validKeys - 1, scale: 0.35f, softcap: 0f, sink: null, slidingWindow: 0, alibiSlopes: null);

        public AttentionCall Contiguous() =>
            AttentionCall.FromFlashAttention(Output, Query, KeyPages, ValuePages, ValidKeys, causal: true, qOffset: ValidKeys - 1, scale: 0.35f,
                softcap: 0f, sink: null, slidingWindow: 0, alibiSlopes: null);

        public void AssertNothingWritten() => Assert.All(Output.AsReadOnlySpan<float>().ToArray(), value => Assert.Equal(-123f, value));

        public void Dispose()
        {
            Query.Dispose();
            Output.Dispose();
            KeyPages.Dispose();
            ValuePages.Dispose();
            Table.Dispose();
        }
    }

    private static AttentionKernelCapability FlashRow(bool paged) => new()
    {
        Id = paged ? "test.paged" : "test.contiguous", Entry = AttentionRoute.Flash, QueryTypes = AttentionDTypes.F32, KvTypes = AttentionDTypes.F32,
        Layouts = AttentionLayouts.HeadMajor, Gqa = true, Masks = AttentionMaskSupport.Causal | AttentionMaskSupport.Window, LogitTerms = true,
        PagedKv = paged,
    };

    [Fact]
    public void ARowThatReadsContiguousKeysExcludesAPagedCall()
    {
        using Operands operands = new();
        AttentionCall paged = operands.Call();
        Assert.Equal(AttentionExclusion.PagedKv, FlashRow(paged: false).Excludes(in paged));
        Assert.Equal(AttentionExclusion.None, FlashRow(paged: true).Excludes(in paged));
    }

    [Fact]
    public void ARowThatFollowsAPageTableExcludesAContiguousCall()
    {
        using Operands operands = new();
        AttentionCall contiguous = operands.Contiguous();
        Assert.Equal(AttentionExclusion.PagedKv, FlashRow(paged: true).Excludes(in contiguous));
        Assert.Equal(AttentionExclusion.None, FlashRow(paged: false).Excludes(in contiguous));
    }

    [Fact]
    public void ThePlannerNeverGivesAPagedCallToAContiguousRow()
    {
        using Operands operands = new();
        AttentionCall paged = operands.Call();
        AttentionKernelRow contiguous = new DelegateAttentionRow(FlashRow(paged: false), null, static (in AttentionCall _) => true,
            static (in AttentionCall _) => true);
        AttentionPlan plan = AttentionPlanner.Plan([contiguous], in paged);
        Assert.Null(plan.Row);
        Assert.Throws<AttentionUnsupportedException>(() => AttentionPlanner.Run([contiguous], in paged));
    }

    [Fact]
    public void TheCpuBackendRefusesAPagedCallAndDoesNotAdmitIt()
    {
        using Operands operands = new();
        AttentionCall paged = operands.Call();
        using IBackend backend = new CpuBackend();
        Assert.False(backend.AdmitsPagedAttention(in paged));
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in paged));
        operands.AssertNothingWritten();
    }

    [Fact]
    public void TheHostReferenceRefusesAPagedCall()
    {
        using Operands operands = new();
        AttentionCall paged = operands.Call();
        Assert.Throws<AttentionUnsupportedException>(() => AttentionHostReference.Run(in paged));
        operands.AssertNothingWritten();
    }

    [Fact]
    public void APagedCallIsNotCarriedByALegacyEntryOrTheFlashLowering()
    {
        using Operands operands = new();
        AttentionCall paged = operands.Call();
        Assert.NotNull(paged.LegacyGap());
        Assert.NotNull(paged.TermsGap());
        Assert.NotNull(AttentionFlashLowering.Gap(paged with { Route = AttentionRoute.Auto }));
    }

    [Fact]
    public void APageTableShorterThanTheValidKeysIsMalformed()
    {
        using Operands operands = new();
        // Three pages of four rows reach 12 keys; 13 need a fourth.
        AttentionCall call = operands.Call(validKeys: 13);
        using IBackend backend = new CpuBackend();
        Assert.Throws<ArgumentException>(() => backend.Attention(in call));
        operands.AssertNothingWritten();
    }

    [Fact]
    public void APageBeyondThePageBuffersIsMalformed()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call(new AttentionKvPages(operands.Table, PageSize, PoolPages));
        using IBackend backend = new CpuBackend();
        Assert.Throws<ArgumentException>(() => backend.Attention(in call));
        operands.AssertNothingWritten();
    }

    [Fact]
    public void APageTableThatIsNotI32OrHasNoPageSizeIsMalformed()
    {
        using Operands operands = new();
        using Tensor floats = new(new TensorShape(3), DType.F32);
        using IBackend backend = new CpuBackend();
        AttentionCall wrongType = operands.Call(new AttentionKvPages(floats, PageSize, 5));
        AttentionCall noSize = operands.Call(new AttentionKvPages(operands.Table, 0, 5));
        Assert.Throws<ArgumentException>(() => backend.Attention(in wrongType));
        Assert.Throws<ArgumentException>(() => backend.Attention(in noSize));
        operands.AssertNothingWritten();
    }

    [Fact]
    public void APagedCallStatesItsKeyLengthOnTheHost()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call() with { Positions = AttentionPositions.Device(0x1000, ValidKeys, ValidKeys - 1) };
        Assert.Throws<ArgumentException>(() => AttentionExecution.Prepare(in call));
    }
}
