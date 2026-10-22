using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>What keys a strided cuDNN plan, which operand views the cuDNN rows take, and how the planner routes a call over them; no device.</summary>
public sealed class CudnnSdpaStridedViewTests
{
    private const int S = 16, D = 64;

    private static Tensor Of(DType dt, params long[] dims) => new(new TensorShape(dims), dt);

    private static CudaBackend Backend(bool cudnn = true) => CudnnSdpaBackendRowsTests.Backend(cudnn);

    // ── plan key ──────────────────────────────────────────────────────────────────────────────────────────────

    private static AttentionOperand Col(Tensor qkv, int column, int heads, int batch = 1, int tokens = S) =>
        AttentionOperand.Packed(qkv, tokens, (int)qkv.Shape[1], column, heads, D, batch);

    private static CudnnSdpa.PlanKey KeyOf(in AttentionCall c) => CudnnSdpa.PlanKey.Of(CudnnSdpa.SdpaDtype.F16, c.QkHeadDim,
        CudnnSdpa.View.Of(c.Query), CudnnSdpa.View.Of(c.Key), CudnnSdpa.View.Of(c.Value), CudnnSdpa.View.Of(c.Output), CudnnBias.None);

    /// <summary>Fused QKV rows of 3·H·D, the three parts read from the given column order.</summary>
    private static AttentionCall Fused(int batch, int heads, int qCol, int kCol, int vCol, DType? type = null)
    {
        DType dt = type ?? DType.F16;
        Tensor qkv = Of(dt, (long)batch * S, 3L * heads * D), o = Of(dt, batch, S, heads, D);
        return new()
        {
            Query = Col(qkv, qCol, heads, batch), Key = Col(qkv, kCol, heads, batch), Value = Col(qkv, vCol, heads, batch),
            Output = AttentionOperand.TokenMajor(o, heads, D), Scale = 0.125f, Numerics = AttentionNumerics.FromLegacy(true),
            Route = AttentionRoute.TokenMajor,
        };
    }

    [Fact]
    public void PlanKey_HoldsBatchAndStrides_ButNotOffsets()
    {
        const int hd = 4 * D;
        AttentionCall a = Fused(2, 4, 0, hd, 2 * hd), b = Fused(2, 4, hd, 2 * hd, 0);
        Assert.Equal(KeyOf(a), KeyOf(b));
        Assert.Equal(KeyOf(a).GetHashCode(), KeyOf(b).GetHashCode());
        Assert.NotEqual(CudnnSdpa.Offsets.Of(a), CudnnSdpa.Offsets.Of(b));
        Assert.Equal(new CudnnSdpa.Offsets(0, hd, 2 * hd, 0), CudnnSdpa.Offsets.Of(a));
        Assert.DoesNotContain(typeof(CudnnSdpa.PlanKey).GetProperties(), p => p.Name.Contains("Offset", StringComparison.OrdinalIgnoreCase));

        Assert.NotEqual(KeyOf(a), KeyOf(Fused(1, 4, 0, hd, 2 * hd)));   // batch
        AttentionCall plain = a with { Query = AttentionOperand.TokenMajor(Of(DType.F16, 2, S, 4, D), 4, D) };
        Assert.NotEqual(KeyOf(a), KeyOf(plain));                           // the query's token stride
    }

    [Fact]
    public void PlanKey_EachTensorsEachStrideIsKeyed()
    {
        CudnnSdpa.View v = CudnnSdpa.View.TokenMajor(2, 4, S, D);
        CudnnSdpa.PlanKey key = new(CudnnSdpa.SdpaDtype.F16, D, v, v, v, v);
        CudnnSdpa.View[] variants = [v with { BatchStride = v.BatchStride + 8 }, v with { HeadStride = v.HeadStride + 8 }, v with { TokenStride = v.TokenStride + 8 }];
        foreach (CudnnSdpa.View w in variants)
        {
            Assert.NotEqual(key, key with { Q = w });
            Assert.NotEqual(key, key with { K = w });
            Assert.NotEqual(key, key with { V = w });
            Assert.NotEqual(key, key with { O = w });
        }
    }

    [Fact]
    public void PlanKey_BiasDimsAndStridesAreKeyed_AndTheMaskKindFollowsThem()
    {
        CudnnSdpa.View v = CudnnSdpa.View.HeadMajor(2, 4, S, D);
        CudnnSdpa.PlanKey none = new(CudnnSdpa.SdpaDtype.F16, D, v, v, v, v);
        CudnnSdpa.PlanKey keyOnly = none with { Bias = CudnnSdpa.View.Bias(new CudnnBias(1, 1, 1), S) };
        CudnnSdpa.PlanKey perBatch = none with { Bias = CudnnSdpa.View.Bias(new CudnnBias(2, 1, S), S) };
        Assert.Equal(CudnnSdpaMask.None, none.Mask);
        Assert.Equal(CudnnSdpaMask.Bias, keyOnly.Mask);
        Assert.Equal(3, new[] { none, keyOnly, perBatch }.Distinct().Count());
        Assert.NotEqual(perBatch, perBatch with { Bias = perBatch.Bias with { BatchStride = perBatch.Bias.BatchStride + 4 } });
    }

    [Fact]
    public void PlanKey_Validate_RefusesViewsCudnnCannotTake()
    {
        CudnnSdpa.View v = CudnnSdpa.View.TokenMajor(2, 4, S, D);
        CudnnSdpa.PlanKey ok = new(CudnnSdpa.SdpaDtype.F16, D, v, v, v, v);
        ok.Validate();
        Assert.Throws<ArgumentException>(() => (ok with { K = v with { DimStride = 2 } }).Validate());
        Assert.Throws<ArgumentException>(() => (ok with { Q = v with { TokenStride = v.TokenStride + 4 } }).Validate());
        Assert.Throws<ArgumentException>(() => (ok with { V = v with { B = 3 } }).Validate());                  // batch disagrees
        Assert.Throws<ArgumentException>(() => (ok with { O = CudnnSdpa.View.TokenMajor(2, 3, S, D) }).Validate());
    }

    // ── failure state: a refused view stays off the whole-tensor plans ───────────────────────────────────────

    [Fact]
    public void TheSignature_SeparatesViewsFromWholeTensors()
    {
        CudnnSdpaSignature viewSignature = CudnnSdpaSignature.For(DType.F16, 4, 4, D, CudnnBias.None, strided: true);
        AttentionCall packed = Fused(1, 4, 0, 256, 512);
        Assert.Equal(viewSignature, CudnnSdpaSignature.Of(in packed));
        Assert.NotEqual(CudnnSdpaSignature.For(DType.F16, 4, 4, D, CudnnBias.None), viewSignature);
        Assert.Contains("strided", viewSignature.ToString());

        AttentionCall tokenMajor = TokenMajor(DType.F16, 1, 4, 4), batched = TokenMajor(DType.F16, 2, 4, 4);
        Assert.False(CudnnSdpaSignature.Of(in tokenMajor).Strided);
        Assert.False(CudnnSdpaSignature.Of(in batched).Strided);                  // whole tensors, natural strides
        AttentionCall offset = tokenMajor with { Key = tokenMajor.Key with { Offset = 8 } };
        Assert.True(CudnnSdpaSignature.Of(in offset).Strided);
        Tensor k = Of(DType.F16, 1, 4, S, D);
        AttentionCall mixed = tokenMajor with { Key = AttentionOperand.HeadMajor(k), Value = AttentionOperand.HeadMajor(k) };
        Assert.True(CudnnSdpaSignature.Of(in mixed).Strided);
        AttentionCall headMajor = new()
        {
            Query = AttentionOperand.HeadMajor(k), Key = AttentionOperand.HeadMajor(k), Value = AttentionOperand.HeadMajor(k),
            Output = AttentionOperand.HeadMajor(k), Scale = 1f, Route = AttentionRoute.Dense,
        };
        Assert.False(CudnnSdpaSignature.Of(in headMajor).Strided);
    }

    [Fact]
    public void RefusedViews_NeverDisableTheWholeTensorPlansOfTheSameDtypeAndHeadDim()
    {
        CudnnSdpaHealth health = new();
        CudnnSdpaSignature views = CudnnSdpaSignature.For(DType.F16, 4, 4, D, CudnnBias.None, strided: true);
        CudnnSdpaSignature whole = CudnnSdpaSignature.For(DType.F16, 4, 4, D, CudnnBias.None);
        for (int i = 0; i < CudnnSdpaHealth.RefusalsToDisable + 2; i++)
        {
            CudnnSdpa.View v = CudnnSdpa.View.HeadMajor(1, 4, S + 8 * i, D);
            health.RecordFailure(new CudnnSdpaShape(views, DType.F16, new(CudnnSdpa.SdpaDtype.F16, D, v, v, v, v)), permanent: true, nowTicks: 0);
        }
        Assert.False(health.Eligible(views, 0));
        Assert.True(health.Eligible(whole, 0));
    }

    // ── graph identity for the plain calls ────────────────────────────────────────────────────────────────────

    /// <summary>The five-to-ten tensors the pre-stride BuildPlan declared, transcribed with its uids and creation order.</summary>
    private static CudnnSdpa.GraphTensor[] MainGraph(long b, long h, long hKv, long sq, long sk, long d, bool tokenMajor, int io,
        long biasB, long biasH, long biasSq)
    {
        long qHeadStr = tokenMajor ? d : sq * d, qSeqStr = tokenMajor ? h * d : d;
        long kHeadStr = tokenMajor ? d : sk * d, kSeqStr = tokenMajor ? hKv * d : d;
        long[] sDim = [b, h, sq, sk], sStr = [h * sq * sk, sq * sk, sk, 1], one = [1, 1, 1, 1];
        List<CudnnSdpa.GraphTensor> t =
        [
            new(1, [b, h, sq, d], [h * sq * d, qHeadStr, qSeqStr, 1], io, false),
            new(2, [b, hKv, d, sk], [hKv * sk * d, kHeadStr, 1, kSeqStr], io, false),
            new(3, [b, hKv, sk, d], [hKv * sk * d, kHeadStr, kSeqStr, 1], io, false),
            new(4, [b, h, sq, d], [h * sq * d, qHeadStr, qSeqStr, 1], io, false),
            new(5, one, one, CudnnApi.CUDNN_DATA_FLOAT, false, true),
            new(100, sDim, sStr, CudnnApi.CUDNN_DATA_FLOAT, true),
            new(101, sDim, sStr, CudnnApi.CUDNN_DATA_FLOAT, true),
            new(102, sDim, sStr, io, true),
        ];
        if (biasSq > 0)
        {
            t.Add(new(6, [biasB, biasH, biasSq, sk], [biasH * biasSq * sk, biasSq * sk, sk, 1], CudnnApi.CUDNN_DATA_FLOAT, false));
            t.Add(new(103, sDim, sStr, CudnnApi.CUDNN_DATA_FLOAT, true));
        }
        return [.. t];
    }

    private static void AssertSameGraph(CudnnSdpa.GraphTensor[] want, CudnnSdpa.GraphTensor[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(want[i].Uid == got[i].Uid && want[i].Dtype == got[i].Dtype && want[i].Virtual == got[i].Virtual
                    && want[i].ByValue == got[i].ByValue && want[i].Dims.SequenceEqual(got[i].Dims) && want[i].Strides.SequenceEqual(got[i].Strides),
                $"{what}: tensor {want[i].Uid} differs: want [{string.Join(",", want[i].Dims)}]/[{string.Join(",", want[i].Strides)}], got [{string.Join(",", got[i].Dims)}]/[{string.Join(",", got[i].Strides)}]");
        }
    }

    public static IEnumerable<object[]> PlainShapes() =>
        from shape in new (int H, int Hkv, int Sq, int Sk, int D)[] { (8, 8, 64, 64, 128), (24, 24, 48, 96, 64), (48, 12, 32, 32, 128), (1, 1, 8, 8, 32), (4, 2, 16, 24, 40) }
        from tokenMajor in new[] { false, true }
        from bias in new[] { 0, 1, 2, 3 }
        select new object[] { shape.H, shape.Hkv, shape.Sq, shape.Sk, shape.D, tokenMajor, bias };

    [Theory]
    [MemberData(nameof(PlainShapes))]
    public void APlainB1Call_BuildsTheGraphMainBuilt(int h, int hKv, int sq, int sk, int d, bool tokenMajor, int biasKind)
    {
        foreach (DType dt in new[] { DType.F16, DType.BF16 })
        {
            Tensor q = Of(dt, 1, h, sq, d), k = Of(dt, 1, hKv, sk, d), v = Of(dt, 1, hKv, sk, d), o = Of(dt, 1, h, sq, d);
            AttentionOperand Op(Tensor t, int heads, int rows) => tokenMajor
                ? AttentionOperand.TokenMajor(Of(dt, rows, (long)heads * d), heads, d) : AttentionOperand.HeadMajor(t);
            CudnnBias bias = biasKind switch { 1 => new(1, 1, sq), 2 => new(1, 1, 1), 3 => new(1, h, sq), _ => CudnnBias.None };
            CudnnSdpa.PlanKey key = CudnnSdpa.PlanKey.Of(dt == DType.BF16 ? CudnnSdpa.SdpaDtype.BF16 : CudnnSdpa.SdpaDtype.F16, d,
                CudnnSdpa.View.Of(Op(q, h, sq)), CudnnSdpa.View.Of(Op(k, hKv, sk)), CudnnSdpa.View.Of(Op(v, hKv, sk)), CudnnSdpa.View.Of(Op(o, h, sq)), bias);
            key.Validate();
            AssertSameGraph(MainGraph(1, h, hKv, sq, sk, d, tokenMajor, key.CudnnDtype, bias.B, bias.H, bias.Sq), CudnnSdpa.GraphTensors(key),
                $"{(tokenMajor ? "token-major" : "head-major")} h{h}/{hKv} sq{sq} sk{sk} d{d} bias{biasKind} {dt}");
        }
    }

    [Fact]
    public void TheViewFactories_AgreeWithTheOperandFactories()
    {
        Assert.Equal(CudnnSdpa.View.HeadMajor(2, 8, 32, D), CudnnSdpa.View.Of(AttentionOperand.HeadMajor(Of(DType.F16, 2, 8, 32, D))));
        Assert.Equal(CudnnSdpa.View.TokenMajor(2, 8, 32, D), CudnnSdpa.View.Of(AttentionOperand.TokenMajor(Of(DType.F16, 2, 32, 8, D), 8, D)));
        Assert.Equal(CudnnSdpa.View.TokenMajor(1, 8, 32, D), CudnnSdpa.View.Of(AttentionOperand.TokenMajor(Of(DType.F16, 32, 8L * D), 8, D)));
    }

    // ── what the rows take: alignment and contiguity, with negative controls ─────────────────────────────────

    [Theory]
    [InlineData(0, true)]
    [InlineData(256, true)]
    [InlineData(512, true)]
    [InlineData(24, true)]      // 16-byte aligned, though not a multiple of the head dim
    [InlineData(4, false)]      // 8-byte aligned only
    [InlineData(1, false)]      // odd offset
    [InlineData(20, false)]     // an unaligned column of a packed row
    [InlineData(600, false)]    // 16-byte aligned but the heads run past the row's tensor
    public void APackedColumn_IsTakenOnlyOnA16ByteBoundaryInsideItsTensor(int column, bool takes)
    {
        AttentionOperand x = Col(Of(DType.F16, S, 768), column, 4);
        Assert.Equal(takes, CudnnSdpaRows.Takes(in x));
    }

    [Fact]
    public void AnOddOrUnalignedStride_IsRefused()
    {
        Assert.True(CudnnSdpaRows.Takes(Col(Of(DType.F16, S, 768), 0, 4)));
        AttentionOperand odd = Col(Of(DType.F16, S, 771), 0, 4);        // rows of 771 elements
        AttentionOperand half = Col(Of(DType.F16, S, 764), 0, 4);       // 764 % 8 = 4
        Assert.False(CudnnSdpaRows.Takes(in odd));
        Assert.False(CudnnSdpaRows.Takes(in half));
        AttentionOperand plain = AttentionOperand.TokenMajor(Of(DType.F16, S, 4L * D), 4, D);
        Assert.True(CudnnSdpaRows.Takes(in plain));
    }

    [Fact]
    public void AHeadDimThatIsNotContiguousOrAligned_IsRefused()
    {
        CudnnSdpa.View v = CudnnSdpa.View.TokenMajor(1, 4, S, D);
        Assert.True(v.Takes(D));
        Assert.False((v with { DimStride = 2 }).Takes(D));                      // head dim not contiguous
        Assert.False((v with { DimStride = 0 }).Takes(D));
        Assert.False(v.Takes(36));                                              // head dim not a multiple of 8
        Assert.False((v with { HeadStride = D - 8 }).Takes(D));                 // neighbouring heads overlap a head dim run
        Assert.False((v with { TokenStride = 0 }).Takes(D));
        Assert.False((v with { BatchStride = -v.BatchStride }).Takes(D));
        Assert.True((v with { S = 1, TokenStride = 8 }).Takes(D));              // the stride of a single row is unused
    }

    [Fact]
    public void AnOutputMustCoverItsWholeTensor()
    {
        AttentionCall fused = Fused(1, 4, 0, 256, 512);
        Assert.True(CudnnSdpaRows.ViewsFit(in fused));
        AttentionCall intoColumn = fused with { Output = Col(Of(DType.F16, S, 768), 256, 4) };
        Assert.False(CudnnSdpaRows.ViewsFit(in intoColumn));                    // would replace the other columns' buffer
        AttentionCall shifted = fused with { Output = fused.Output with { Offset = 8 } };
        Assert.False(CudnnSdpaRows.ViewsFit(in shifted));
        AttentionCall headMajorOut = fused with { Output = AttentionOperand.HeadMajor(Of(DType.F16, 1, 4, S, D)) };
        Assert.True(CudnnSdpaRows.ViewsFit(in headMajorOut));                   // any dense order of the whole tensor
    }

    [Fact]
    public void F32OperandsAreCastWhole_SoOnlyWholeTensorsFit()
    {
        AttentionCall plain = Fused(2, 4, 0, 256, 512) with { Numerics = default };
        Tensor Mk(params long[] dims) => Of(DType.F32, dims);
        AttentionCall f32Plain = new()
        {
            Query = AttentionOperand.TokenMajor(Mk(2, S, 4, D), 4, D), Key = AttentionOperand.TokenMajor(Mk(2, S, 4, D), 4, D),
            Value = AttentionOperand.TokenMajor(Mk(2, S, 4, D), 4, D), Output = AttentionOperand.TokenMajor(Mk(2, S, 4, D), 4, D), Scale = 1f,
            Route = AttentionRoute.TokenMajor,
        };
        Assert.True(CudnnSdpaRows.ViewsFit(in f32Plain));
        Tensor qkv = Mk(2L * S, 768);
        AttentionCall f32Packed = f32Plain with { Query = Col(qkv, 0, 4, 2), Key = Col(qkv, 256, 4, 2), Value = Col(qkv, 512, 4, 2) };
        Assert.False(CudnnSdpaRows.ViewsFit(in f32Packed));
        Assert.True(CudnnSdpaRows.ViewsFit(in plain));                          // the same view in F16 fits
    }

    [Fact]
    public void OperandsThatDisagreeOnGeometryOrDtype_DoNotFit()
    {
        AttentionCall ok = Fused(2, 4, 0, 256, 512);
        Assert.True(CudnnSdpaRows.ViewsFit(in ok));
        Assert.False(CudnnSdpaRows.ViewsFit(ok with { Key = ok.Key with { Batch = 1 } }));
        Assert.False(CudnnSdpaRows.ViewsFit(ok with { Value = ok.Value with { Tokens = S - 1 } }));
        Assert.False(CudnnSdpaRows.ViewsFit(ok with { Key = ok.Key with { Heads = 3 } }));
        Assert.False(CudnnSdpaRows.ViewsFit(ok with { Output = AttentionOperand.TokenMajor(Of(DType.BF16, 2, S, 4, D), 4, D) }));
    }

    // ── routing: a strided call plans on cuDNN when its permissions allow, else on the next row or none ──────

    private static AttentionCall TokenMajor(DType dt, int batch, int hq, int hkv, int d = D, Tensor? mask = null, bool allowF16 = true)
    {
        Tensor q = Of(dt, batch, S, hq, d), k = Of(dt, batch, S, hkv, d), v = Of(dt, batch, S, hkv, d), o = Of(dt, batch, S, hq, d);
        return AttentionCall.FromTokenMajor(o, q, k, v, mask, hq, hkv, d, 0.125f, allowF16);
    }

    [Fact]
    public void AFusedQkvColumnView_PlansOnTheTokenMajorCudnnRow_WhateverTheBatch()
    {
        CudaBackend backend = Backend();
        foreach (int batch in new[] { 1, 2 })
        {
            AttentionCall fused = Fused(batch, 4, 0, 256, 512);
            Assert.Null(fused.TermsGap());
            Assert.NotNull(fused.LegacyGap());
            AttentionPlan plan = backend.PlanAttention(in fused);
            Assert.Equal("cudnn.tokenmajor", plan.RowId);
            Assert.DoesNotContain("Permute", plan.Describe());
        }
    }

    [Fact]
    public void ABatchedTokenMajorCall_PlansOnCudnn_AndTheGroupedOneOnTheGroupedRow()
    {
        CudaBackend backend = Backend();
        AttentionCall equal = TokenMajor(DType.F16, 2, 24, 24, 128);
        Assert.Equal("cudnn.tokenmajor", backend.PlanAttention(in equal).RowId);
        AttentionCall gqa = TokenMajor(DType.F16, 2, 48, 12, 128);
        Assert.Equal("cudnn.tokenmajor.gqa", backend.PlanAttention(in gqa).RowId);
        // A fused [q|k|v] row of (48 + 2·12)·128 columns, the way Krea2's projection lays it out.
        Tensor qkv = Of(DType.F16, 2L * S, 72L * 128);
        AttentionCall fusedGqa = gqa with
        {
            Query = AttentionOperand.Packed(qkv, S, 72 * 128, 0, 48, 128, 2), Key = AttentionOperand.Packed(qkv, S, 72 * 128, 48 * 128, 12, 128, 2),
            Value = AttentionOperand.Packed(qkv, S, 72 * 128, 60 * 128, 12, 128, 2),
        };
        Assert.Equal("cudnn.tokenmajor.gqa", backend.PlanAttention(in fusedGqa).RowId);
    }

    [Fact]
    public void MixedLayouts_PlanOnTheHeadMajorCudnnRow()
    {
        CudaBackend backend = Backend();
        Tensor q = Of(DType.F16, 1, S, 4, D), k = Of(DType.F16, 1, 4, S, D), v = Of(DType.F16, 1, 4, S, D), o = Of(DType.F16, 1, S, 4, D);
        AttentionCall mixed = new()
        {
            Query = AttentionOperand.TokenMajor(q, 4, D), Key = AttentionOperand.HeadMajor(k), Value = AttentionOperand.HeadMajor(v),
            Output = AttentionOperand.TokenMajor(o, 4, D), Scale = 0.125f, Route = AttentionRoute.Dense,
        };
        Assert.Equal("cudnn.f16", backend.PlanAttention(in mixed).RowId);
    }

    [Fact]
    public void AViewNoCudnnRowTakes_PlansNoRow_AndNeverAStagingCopy()
    {
        CudaBackend backend = Backend();
        AttentionCall unaligned = Fused(1, 4, 4, 260, 516);
        AttentionPlan refused = backend.PlanAttention(in unaligned);
        Assert.Equal("none", refused.RowId);
        Assert.Contains("cudnn.tokenmajor (not admitted)", refused.Reason);
        Assert.Contains("permute.dense (Layout)", refused.Reason);

        // A whole-tensor layout with an offset is a view the permuting row cannot read either.
        AttentionCall plain = TokenMajor(DType.F16, 1, 4, 4);
        AttentionCall offset = plain with { Query = plain.Query with { Offset = 8 } };
        AttentionPlan offsetPlan = backend.PlanAttention(in offset);
        Assert.Equal("none", offsetPlan.RowId);
        Assert.Contains("permute.dense (Strides)", offsetPlan.Reason);

        AttentionCall intoColumn = Fused(1, 4, 0, 256, 512) with { Output = Col(Of(DType.F16, S, 768), 256, 4) };
        Assert.Equal("none", backend.PlanAttention(in intoColumn).RowId);

        AttentionCall f32Fused = Fused(1, 4, 0, 256, 512, DType.F32);
        Assert.Equal("none", backend.PlanAttention(in f32Fused).RowId);

        AttentionCall off = Fused(1, 4, 0, 256, 512);
        Assert.Equal("none", Backend(cudnn: false).PlanAttention(in off).RowId);
    }

    [Fact]
    public void ABatchedCallWithoutACudnnRow_IsRefusedBecauseThePermuteRowsReadOneSequence()
    {
        CudaBackend backend = Backend();
        AttentionCall f32NoGrant = TokenMajor(DType.F32, 2, 4, 4, allowF16: false);
        AttentionPlan plan = backend.PlanAttention(in f32NoGrant);
        Assert.Equal("none", plan.RowId);
        Assert.Contains("permute.dense (Batch)", plan.Reason);
        AttentionCall f32Granted = TokenMajor(DType.F32, 2, 4, 4);
        Assert.Equal("cudnn.tokenmajor", backend.PlanAttention(in f32Granted).RowId);
        AttentionCall gqa = TokenMajor(DType.F32, 2, 8, 2, allowF16: false);
        Assert.Contains("permute.widen.dense (Batch)", backend.PlanAttention(in gqa).Reason);
    }

    [Fact]
    public void APlainB1Call_KeepsItsRows()
    {
        CudaBackend backend = Backend();
        AttentionCall d128 = TokenMajor(DType.F16, 1, 4, 4, 128);
        Assert.Equal("cudnn.tokenmajor", backend.PlanAttention(in d128).RowId);
        AttentionCall d36 = TokenMajor(DType.F16, 1, 4, 4, 36);
        Assert.Equal("permute.dense", backend.PlanAttention(in d36).RowId);
        AttentionCall gqa = TokenMajor(DType.F16, 1, 48, 12, 128);
        Assert.Equal("cudnn.tokenmajor.gqa", backend.PlanAttention(in gqa).RowId);
        AttentionCall off = TokenMajor(DType.F16, 1, 4, 4, 128);
        Assert.Equal("permute.dense", Backend(cudnn: false).PlanAttention(in off).RowId);
    }

    [Fact]
    public void TheCudnnRowsDeclareTheViewsTheyTake_AndTheOthersDoNot()
    {
        CudaBackend backend = Backend();
        foreach ((string builder, string id, bool strided) in new[]
        {
            ("BuildDenseRows", "cudnn.f16", true), ("BuildDenseRows", "cudnn.f32.sat", false), ("BuildTokenMajorRows", "cudnn.tokenmajor", true),
            ("BuildTokenMajorGqaRows", "cudnn.tokenmajor.gqa", true), ("BuildTokenMajorRows", "flash.f16", false),
            ("BuildTokenMajorRows", "permute.dense", false), ("BuildTokenMajorGqaRows", "permute.widen.dense", false),
        })
        {
            AttentionKernelCapability c = CudnnSdpaBackendRowsTests.Rows(backend, builder).Single(r => r.Capability.Id == id).Capability;
            Assert.Equal(strided, c.StridedViews);
            if (strided) Assert.Equal(CudnnSdpaRows.AnyView, c.Layouts);
            Assert.Equal(id is "flash.f16" or "permute.dense" or "permute.widen.dense" ? 1 : int.MaxValue, c.MaxBatch);
        }
    }

    // ── token-major masks: validated at planning ──────────────────────────────────────────────────────────────

    private static Tensor Mask(params long[] dims) => Of(DType.F32, dims);

    public static IEnumerable<object[]> CudnnMasks() =>
    [
        [1, new long[] { 1, 1, 1, S }], [1, new long[] { 1, S }], [1, new long[] { 1, 1, S, S }], [1, new long[] { 1, 4, S, S }],
        [2, new long[] { 2, 1, S, S }], [2, new long[] { 2, 4, S, S }], [2, new long[] { 2, 1, 1, S }], [2, new long[] { 4, 1, S }],
    ];

    [Theory]
    [MemberData(nameof(CudnnMasks))]
    public void TheTokenMajorCudnnRow_TakesWhatTheHeadMajorRowTakes(int batch, long[] shape)
    {
        CudaBackend backend = Backend();
        AttentionCall call = TokenMajor(DType.F16, batch, 4, 4, mask: Mask(shape));
        Assert.Equal("cudnn.tokenmajor", backend.PlanAttention(in call).RowId);
        AttentionCall gqa = TokenMajor(DType.F16, batch, 4, 2, mask: Mask(shape));
        Assert.Equal("cudnn.tokenmajor.gqa", backend.PlanAttention(in gqa).RowId);
    }

    [Theory]
    [InlineData(new long[] { S, S }, "permute.dense")]
    [InlineData(new long[] { 4, S, S }, "permute.dense")]
    [InlineData(new long[] { 1, S, S }, "permute.dense")]
    [InlineData(new long[] { 1, 1, S, S }, "permute.dense")]
    [InlineData(new long[] { 1, 4, S, S }, "permute.dense")]
    [InlineData(new long[] { 1, S }, "none")]                 // key-only: refused where it was refused before
    [InlineData(new long[] { 1, 1, 1, S }, "none")]
    [InlineData(new long[] { 4, 1, S }, "none")]
    [InlineData(new long[] { 2, 1, S, S }, "none")]           // a batch of 2 against one sequence
    [InlineData(new long[] { 1, 1, S, S + 1 }, "none")]
    public void TheTokenMajorFallbacks_KeepTodaysMasks_WhenCudnnIsOff(long[] shape, string row)
    {
        CudaBackend backend = Backend(cudnn: false);
        AttentionCall call = TokenMajor(DType.F16, 1, 4, 4, mask: Mask(shape));
        Assert.Equal(row, backend.PlanAttention(in call).RowId);
    }

    [Theory]
    [InlineData(new long[] { 1, 1, 1, S }, "permute.widen.dense")]     // the head-major entry takes a key row
    [InlineData(new long[] { 1, S }, "permute.widen.dense")]
    [InlineData(new long[] { 1, 4, S, S }, "permute.widen.dense")]
    [InlineData(new long[] { 2, 1, S, S }, "none")]
    [InlineData(new long[] { 1, 3, S, S }, "none")]
    public void TheWideningFallback_KeepsTheHeadMajorEntrysMasks_WhenCudnnIsOff(long[] shape, string row)
    {
        CudaBackend backend = Backend(cudnn: false);
        AttentionCall call = TokenMajor(DType.F16, 1, 4, 2, mask: Mask(shape));
        Assert.Equal(row, backend.PlanAttention(in call).RowId);
    }

    [Fact]
    public void AMaskThatIsNotF32_IsRefusedByEveryTokenMajorRow()
    {
        using Tensor f16 = Of(DType.F16, 1, 1, S, S);
        AttentionCall call = TokenMajor(DType.F16, 1, 4, 4, mask: f16);
        Assert.Equal("none", Backend().PlanAttention(in call).RowId);
        Assert.Equal("none", Backend(cudnn: false).PlanAttention(in call).RowId);
    }

    [Fact]
    public void AKeyOnlyMask_OnF32WithoutThePermission_IsRefusedAsBefore()
    {
        AttentionCall call = TokenMajor(DType.F32, 1, 4, 4, mask: Mask(1, 1, 1, S), allowF16: false);
        Assert.Equal("none", Backend().PlanAttention(in call).RowId);
        AttentionCall granted = TokenMajor(DType.F32, 1, 4, 4, mask: Mask(1, 1, 1, S));
        Assert.Equal("cudnn.tokenmajor", Backend().PlanAttention(in granted).RowId);
    }
}
