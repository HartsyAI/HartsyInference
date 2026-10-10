using System.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.BlockScale;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>EXL3 host decode against exllamav3's own device kernels (committed fixture) and an fp64 dense-Hadamard oracle.</summary>
/// <remarks>Tolerances were fixed before the tests were written and are not to be loosened: the trellis stage is bit exact; the F32 rotation is within 1e-5 of
/// max|W| of an fp64 oracle; and it is within 2^-9 of max|W| of exllamav3's fused kernel, whose fp16 butterflies round at about 1.2e-3 on this fixture.</remarks>
public sealed class Exl3CodecTests : IDisposable
{
    private readonly Exl3FixtureData _fx = new();
    private readonly ITestOutputHelper _output;

    public Exl3CodecTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => _fx.Dispose();

    private float[] DecodeAll(QuantRecipe? recipe = null)
    {
        float[] m = new float[Exl3FixtureData.OutDim * Exl3FixtureData.InDim];
        Exl3Codec.DequantRows(_fx.Trellis, recipe ?? _fx.Recipe(), 0, Exl3FixtureData.OutDim, m);
        return m;
    }

    [Fact]
    public void TrellisStage_IsBitExactAgainstUpstreamReconstructTile()
    {
        Half[] got = new Half[Exl3FixtureData.InDim * Exl3FixtureData.OutDim];
        Exl3Codec.DecodeTrellis(_fx.Trellis, Exl3FixtureData.InDim, Exl3FixtureData.OutDim, got);

        Half[] expected = Exl3FixtureData.Halves(_fx.WHat);
        int mismatches = 0;
        for (int i = 0; i < got.Length; i++)
            if (BitConverter.HalfToUInt16Bits(got[i]) != BitConverter.HalfToUInt16Bits(expected[i])) mismatches++;
        Assert.Equal(0, mismatches);
    }

    [Fact]
    public void Decode_MatchesFp64DenseHadamardOracle()
    {
        int inDim = Exl3FixtureData.InDim, outDim = Exl3FixtureData.OutDim;
        double[] x = Exl3FixtureData.Halves(_fx.WHat).Select(h => (double)h).ToArray(); // [in, out]
        double[,] hm = new double[128, 128];
        double s = 1.0 / Math.Sqrt(128);
        for (int i = 0; i < 128; i++)
            for (int j = 0; j < 128; j++)
                hm[i, j] = s * (BitOperations.PopCount((uint)(i & j)) % 2 == 0 ? 1 : -1);

        // Y = H_in * X * H_out per 128x128 block as two dense matrix products, so it shares nothing with the butterfly.
        double[] y = new double[x.Length], tmp = new double[x.Length];
        for (int ib = 0; ib < inDim / 128; ib++)
            for (int o = 0; o < outDim; o++)
                for (int i = 0; i < 128; i++)
                {
                    double acc = 0;
                    for (int a = 0; a < 128; a++) acc += hm[i, a] * x[(ib * 128 + a) * outDim + o];
                    tmp[(ib * 128 + i) * outDim + o] = acc;
                }
        for (int ob = 0; ob < outDim / 128; ob++)
            for (int i = 0; i < inDim; i++)
                for (int o = 0; o < 128; o++)
                {
                    double acc = 0;
                    for (int b = 0; b < 128; b++) acc += tmp[i * outDim + ob * 128 + b] * hm[b, o];
                    y[i * outDim + ob * 128 + o] = acc;
                }
        Half[] suh = Exl3FixtureData.Halves(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "Exl3", "suh.f16")));
        Half[] svh = Exl3FixtureData.Halves(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "Exl3", "svh.f16")));
        double maxAbs = 0;
        for (int i = 0; i < inDim; i++)
            for (int o = 0; o < outDim; o++)
            {
                y[i * outDim + o] *= (double)suh[i] * (double)svh[o];
                maxAbs = Math.Max(maxAbs, Math.Abs(y[i * outDim + o]));
            }

        float[] m = DecodeAll();
        double worst = 0;
        for (int i = 0; i < inDim; i++)
            for (int o = 0; o < outDim; o++)
                worst = Math.Max(worst, Math.Abs(m[o * inDim + i] - y[i * outDim + o]));
        _output.WriteLine($"host F32 vs fp64 oracle: max abs {worst:E3}, relative to max|W| {worst / maxAbs:E3}");
        Assert.True(worst <= 1e-5 * maxAbs, $"max abs error {worst} exceeds 1e-5 * max|W| = {1e-5 * maxAbs}");
    }

    [Fact]
    public void RowWindow_DecodesTheSameValuesAsTheWholeMatrix()
    {
        float[] whole = DecodeAll();
        int inDim = Exl3FixtureData.InDim;
        float[] window = new float[128 * inDim];
        Exl3Codec.DequantRows(_fx.Trellis, _fx.Recipe(), 128, 128, window);

        Assert.Equal(whole.AsSpan(128 * inDim, 128 * inDim).ToArray(), window);
    }

    [Fact]
    public void RowWindow_OffAHadamardBlock_Refuses()
    {
        const int offset = 64, count = 128;
        float[] dest = new float[count * Exl3FixtureData.InDim];
        NotSupportedException ex = Assert.Throws<NotSupportedException>(
            () => Exl3Codec.DequantRows(_fx.Trellis, _fx.Recipe(), offset, count, dest));
        Assert.Contains($"[{offset}..{offset + count})", ex.Message);
    }

    [Fact]
    public void SliceCols_OnAHadamardBlock_DecodesTheMatchingColumnsExactly()
    {
        float[] whole = DecodeAll();
        int outTiles = Exl3FixtureData.OutDim / 16;
        // Input columns are the leading trellis axis, so the window's bytes are one contiguous run.
        int firstTile = 128 / 16, tiles = 128 / 16;
        byte[] slicedTrellis = _fx.Trellis.AsSpan(firstTile * outTiles * 64, tiles * outTiles * 64).ToArray();

        QuantRecipe sliced = _fx.Recipe().SliceCols(128, 128, "layers.0.ffn.experts.0.w1");
        float[] window = new float[Exl3FixtureData.OutDim * 128];
        Exl3Codec.DequantRows(slicedTrellis, sliced, 0, Exl3FixtureData.OutDim, window);

        Assert.Equal(128, sliced.LogicalCols);
        for (int o = 0; o < Exl3FixtureData.OutDim; o++)
            Assert.Equal(whole.AsSpan(o * Exl3FixtureData.InDim + 128, 128).ToArray(), window.AsSpan(o * 128, 128).ToArray());
    }

    [Theory]
    [InlineData(64, 128)]
    [InlineData(0, 0)]
    public void SliceCols_OffAHadamardBlock_RefusesNamingKeyAndRange(long offset, long count)
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => _fx.Recipe().SliceCols(offset, count, "layers.3.ffn.experts.7.w2"));
        Assert.Contains("layers.3.ffn.experts.7.w2", ex.Message);
        Assert.Contains($"[{offset}..{offset + count})", ex.Message);
    }

    [Fact]
    public void SliceRows_PartialOutputWindow_RefusesButWholeMatrixPasses()
    {
        QuantRecipe recipe = _fx.Recipe();

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => recipe.SliceRows(128, 128, "layers.1.ffn.experts.2.w3"));

        Assert.Contains("layers.1.ffn.experts.2.w3", ex.Message);
        Assert.Contains("[128..256)", ex.Message);
        Assert.Same(recipe, recipe.SliceRows(0, Exl3FixtureData.OutDim, "w"));
    }

    public static TheoryData<string> BadRecipes() => new() { "bits3", "mcg", "suhCount", "packedLength" };

    [Theory]
    [MemberData(nameof(BadRecipes))]
    public void Decode_RefusesARecipeOutsideTheSupportedLayout(string kind)
    {
        using Tensor wrongMcg = Exl3FixtureData.McgTensor(0x12345678);
        using Tensor shortSuh = new(new TensorShape(128), DType.F16);
        QuantRecipe good = _fx.Recipe();
        QuantRecipe bad = kind switch
        {
            "bits3" => good with { Exl3 = good.Exl3! with { Bits = 3 } },
            "mcg" => good with { Exl3 = good.Exl3! with { Mcg = wrongMcg } },
            "suhCount" => good with { Exl3 = good.Exl3! with { Suh = shortSuh } },
            _ => good,
        };
        byte[] packed = kind == "packedLength" ? _fx.Trellis[..^64] : _fx.Trellis;
        float[] dest = new float[Exl3FixtureData.OutDim * Exl3FixtureData.InDim];

        Exception ex = Assert.ThrowsAny<Exception>(() => Exl3Codec.DequantRows(packed, bad, 0, Exl3FixtureData.OutDim, dest));

        Assert.True(ex is NotSupportedException or ArgumentException, ex.ToString());
    }
}
