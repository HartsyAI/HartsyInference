using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>What makes an operand view plain, what it covers, and how a row that reads whole tensors only is kept off a strided call.</summary>
public sealed class AttentionOperandViewTests
{
    private const int B = 2, H = 4, S = 8, D = 16;

    private static Tensor Of(DType dt, params long[] dims) => new(new TensorShape(dims), dt);

    [Fact]
    public void APlainView_IsItsWholeTensorInTheLayoutsNaturalStrides()
    {
        Assert.True(AttentionOperand.HeadMajor(Of(DType.F16, B, H, S, D)).IsPlain);
        Assert.True(AttentionOperand.TokenMajor(Of(DType.F16, B, S, H, D), H, D).IsPlain);
        Assert.True(AttentionOperand.TokenMajor(Of(DType.F16, S, H * D), H, D).IsPlain);
        Assert.False(AttentionOperand.Packed(Of(DType.F16, S, 3 * H * D), S, 3 * H * D, 0, H, D).IsPlain);

        AttentionOperand hm = AttentionOperand.HeadMajor(Of(DType.F16, B, H, S, D));
        Assert.False((hm with { Offset = 8 }).IsPlain);
        // A head-major label on token-major strides is not plain, whatever its element count.
        Assert.False((hm with { HeadStride = D, TokenStride = H * D }).IsPlain);
        AttentionOperand tm = AttentionOperand.TokenMajor(Of(DType.F16, B, S, H, D), H, D);
        Assert.False((tm with { TokenStride = 2 * H * D }).IsPlain);
    }

    [Fact]
    public void AViewCoversItsStorage_InAnyAxisOrder_WithNoGapOrOverlap()
    {
        AttentionOperand hm = AttentionOperand.HeadMajor(Of(DType.F16, B, H, S, D));
        AttentionOperand tm = AttentionOperand.TokenMajor(Of(DType.F16, B, S, H, D), H, D);
        Assert.True(hm.CoversStorage);
        Assert.True(tm.CoversStorage);
        Assert.True((hm with { HeadStride = D, TokenStride = H * D, BatchStride = S * H * D }).CoversStorage);   // head-major extents, token-major order
        Assert.True(AttentionOperand.TokenMajor(Of(DType.F16, S, 1L * D), 1, D).CoversStorage);                 // single head: its stride is unused

        Assert.False(AttentionOperand.Packed(Of(DType.F16, B * S, 3 * H * D), S, 3 * H * D, 0, H, D, B).CoversStorage);
        Assert.False((tm with { Offset = 8 }).CoversStorage);
        Assert.False((tm with { TokenStride = 2 * H * D }).CoversStorage);       // a gap between rows
        Assert.False((tm with { HeadStride = 0 }).CoversStorage);                // heads overlap
        Assert.False((tm with { Batch = 1 }).CoversStorage);                     // half the tensor
    }

    [Fact]
    public void APackedView_CarriesABatch_AndReadsTheSameValuesAsPlainCopies()
    {
        const int tokens = 6, heads = 2, d = 8, width = 3 * heads * d;
        Tensor qkv = Of(DType.F32, B * tokens, width);
        Random rng = new(5);
        foreach (ref float x in qkv.AsSpan<float>()) x = (float)(rng.NextDouble() * 2 - 1);

        AttentionOperand q = AttentionOperand.Packed(qkv, tokens, width, 0, heads, d, B);
        AttentionOperand k = AttentionOperand.Packed(qkv, tokens, width, heads * d, heads, d, B);
        AttentionOperand v = AttentionOperand.Packed(qkv, tokens, width, 2 * heads * d, heads, d, B);
        Assert.Equal((long)tokens * width, q.BatchStride);
        Assert.True(q.Extent <= qkv.ElementCount);

        Tensor Copy(int column)
        {
            Tensor t = Of(DType.F32, B, tokens, heads, d);
            Span<float> dst = t.AsSpan<float>();
            ReadOnlySpan<float> src = qkv.AsReadOnlySpan<float>();
            for (int row = 0; row < B * tokens; row++)
                src.Slice(row * width + column, heads * d).CopyTo(dst.Slice(row * heads * d, heads * d));
            return t;
        }

        using Tensor outPacked = Of(DType.F32, B, tokens, heads, d), outPlain = Of(DType.F32, B, tokens, heads, d);
        using Tensor qc = Copy(0), kc = Copy(heads * d), vc = Copy(2 * heads * d);
        AttentionCall packed = new()
        {
            Query = q, Key = k, Value = v, Output = AttentionOperand.TokenMajor(outPacked, heads, d), Scale = 0.35f, Route = AttentionRoute.TokenMajor,
        };
        AttentionCall plain = AttentionCall.FromTokenMajor(outPlain, qc, kc, vc, null, heads, heads, d, 0.35f, false);
        Assert.NotNull(packed.LegacyGap());
        Assert.Null(packed.TermsGap());
        AttentionHostReference.Run(in packed);
        AttentionHostReference.Run(in plain);
        Assert.True(outPacked.AsReadOnlySpan<float>().SequenceEqual(outPlain.AsReadOnlySpan<float>()));
        Assert.Contains(outPlain.AsReadOnlySpan<float>().ToArray(), x => x != 0f);
    }

    // ── planner: strides and batch are declared limits ───────────────────────────────────────────────────────

    private static AttentionKernelCapability Cap(bool strided = false, int maxBatch = int.MaxValue,
        AttentionLayouts layouts = AttentionLayouts.TokenMajor | AttentionLayouts.PackedRows | AttentionLayouts.HeadMajor) => new()
    {
        Id = "row", Entry = AttentionRoute.TokenMajor, QueryTypes = AttentionDTypes.F16, KvTypes = AttentionDTypes.F16, Layouts = layouts,
        StridedViews = strided, MaxBatch = maxBatch,
    };

    private static AttentionCall Call(int batch = 1, bool packed = false)
    {
        Tensor o = Of(DType.F16, batch, S, H, D);
        Tensor qkv = Of(DType.F16, (long)batch * S, 3 * H * D);
        Tensor Whole() => Of(DType.F16, batch, S, H, D);
        return new()
        {
            Query = packed ? AttentionOperand.Packed(qkv, S, 3 * H * D, 0, H, D, batch) : AttentionOperand.TokenMajor(Whole(), H, D),
            Key = packed ? AttentionOperand.Packed(qkv, S, 3 * H * D, H * D, H, D, batch) : AttentionOperand.TokenMajor(Whole(), H, D),
            Value = packed ? AttentionOperand.Packed(qkv, S, 3 * H * D, 2 * H * D, H, D, batch) : AttentionOperand.TokenMajor(Whole(), H, D),
            Output = AttentionOperand.TokenMajor(o, H, D), Scale = 1f, Route = AttentionRoute.TokenMajor,
        };
    }

    [Fact]
    public void ARowThatReadsWholeTensors_ExcludesAStridedCall_AndAStridedRowDoesNot()
    {
        AttentionCall packed = Call(packed: true);
        Assert.Equal(AttentionExclusion.Strides, Cap().Excludes(in packed));
        Assert.Equal(AttentionExclusion.None, Cap(strided: true).Excludes(in packed));
        AttentionCall plain = Call();
        Assert.Equal(AttentionExclusion.None, Cap().Excludes(in plain));
        AttentionCall offset = plain with { Key = plain.Key with { Offset = 8 } };
        Assert.Equal(AttentionExclusion.Strides, Cap().Excludes(in offset));
    }

    [Fact]
    public void ALayoutIsDeclaredForEveryOperand_NotJustQueryAndKey()
    {
        AttentionCall plain = Call();
        AttentionKernelCapability tokenMajorOnly = Cap(layouts: AttentionLayouts.TokenMajor);
        Assert.Equal(AttentionExclusion.None, tokenMajorOnly.Excludes(in plain));
        AttentionCall headMajorValue = plain with { Value = AttentionOperand.HeadMajor(Of(DType.F16, 1, H, S, D)) };
        Assert.Equal(AttentionExclusion.Layout, tokenMajorOnly.Excludes(in headMajorValue));
        AttentionCall headMajorOutput = plain with { Output = AttentionOperand.HeadMajor(Of(DType.F16, 1, H, S, D)) };
        Assert.Equal(AttentionExclusion.Layout, tokenMajorOnly.Excludes(in headMajorOutput));
    }

    [Fact]
    public void TheBatchIsADeclaredLimit()
    {
        AttentionCall batched = Call(batch: 2);
        Assert.Equal(AttentionExclusion.Batch, Cap(maxBatch: 1).Excludes(in batched));
        Assert.Equal(AttentionExclusion.None, Cap(maxBatch: 2).Excludes(in batched));
        AttentionCall single = Call();
        Assert.Equal(AttentionExclusion.None, Cap(maxBatch: 1).Excludes(in single));
    }

    // ── descriptor: what a legacy entry cannot carry ─────────────────────────────────────────────────────────

    [Fact]
    public void TheTermsGap_IgnoresTheOperandViews_ButNotTheMaskOrLogitTerms()
    {
        AttentionCall packed = Call(batch: 2, packed: true);
        Assert.Contains("operands", packed.LegacyGap());
        Assert.Null(packed.TermsGap());
        AttentionCall causal = packed with { Mask = new() { Causal = true } };
        Assert.NotNull(causal.TermsGap());
        AttentionCall softcap = packed with { Logits = new(30f, null, null) };
        Assert.NotNull(softcap.TermsGap());

        // A plain call is unchanged: no gap with an operand view, none without.
        AttentionCall plain = Call();
        Assert.Null(plain.LegacyGap());
        Assert.Null(plain.TermsGap());
        AttentionCall batched = Call(batch: 2);
        Assert.NotNull(batched.LegacyGap());
        Assert.Null(batched.TermsGap());
    }

    [Theory]
    [InlineData(new long[] { S, S })]
    [InlineData(new long[] { 1, S })]
    [InlineData(new long[] { H, 1, S })]
    [InlineData(new long[] { 1, H, S, S })]
    [InlineData(new long[] { B, 1, 1, S })]
    [InlineData(new long[] { 3, S, S })]
    [InlineData(new long[] { 1, 3, S, S })]
    [InlineData(new long[] { 1, 1, S, S + 1 })]
    [InlineData(new long[] { S })]
    public void Broadcasts_IsWhatOfAccepts(long[] shape)
    {
        using Tensor mask = Of(DType.F32, shape);
        bool fits = AttentionMaskLayout.Broadcasts(mask.Shape, B, H, S, S);
        try
        {
            AttentionMaskLayout.Of(mask, B, H, S, S);
            Assert.True(fits);
        }
        catch (ArgumentException)
        {
            Assert.False(fits);
        }
    }
}
