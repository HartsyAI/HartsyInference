using Xunit;
using HartsyInference.Core.Codecs;
using HartsyInference.Core.Tensors;
using HartsyInference.Video.Tokenizers;

namespace HartsyInference.Video.Tests;

/// <summary>Structural unit tests (CPU, no weights) for the Cosmos DV tokenizer's deterministic pieces: FSQ index
/// packing (bit-exact vs an independent reference), the mixed-radix basis / vocab, the invertible Haar wavelet
/// front-end, and the latent-grid geometry. The learned conv autoencoder + real-weight token parity vs
/// <c>encoder.jit</c> is env-gated (see <see cref="CosmosV2WGenerationTests"/>).</summary>
public unsafe class CosmosTokenizerTests
{
    private static readonly int[] DvLevels = [8, 8, 8, 5, 5, 5];

    [Fact]
    public void Fsq_VocabAndBasis_MatchDv8x16x16()
    {
        Assert.Equal(64000, Fsq.VocabSize(DvLevels));
        int[] basis = new int[DvLevels.Length];
        Fsq.Basis(DvLevels, basis);
        Assert.Equal(new[] { 1, 8, 64, 512, 2560, 12800 }, basis);
    }

    [Fact]
    public void Fsq_Quantize_BitExact_VsReference()
    {
        int b = 2, t = 37, d = DvLevels.Length;
        using Tensor z = new(new TensorShape(b, t, d), DType.F32);
        float* zp = (float*)z.DataPointer;
        // Deterministic pseudo-random spread across the tanh-bounded range.
        uint s = 0x1234567u;
        for (long i = 0; i < (long)b * t * d; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            zp[i] = ((s >> 8) * (1f / 16777216f) - 0.5f) * 6f;   // ~[-3, 3]
        }
        using Tensor codes = new(new TensorShape(b, t), DType.I32);
        Fsq.Quantize(codes, z, DvLevels);

        int* cp = (int*)codes.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ti = 0; ti < t; ti++)
            {
                int expected = RefQuantizeOne(zp + (long)(bi * t + ti) * d, DvLevels);
                Assert.Equal(expected, cp[bi * t + ti]);
                Assert.InRange(cp[bi * t + ti], 0, 63999);
            }
    }

    [Fact]
    public void Fsq_DequantizeToCodesToIndices_RoundTripsBitExact()
    {
        // The pack/unpack invariant: any index → grid code (Dequantize) → index (CodesToIndices, the no-tanh
        // inverse) must reproduce the original index exactly. (Quantize includes the tanh bound and is the
        // raw-latent encoder path — NOT the inverse of Dequantize, so it is not used here.)
        int b = 1, t = 128;
        using Tensor codes = new(new TensorShape(b, t), DType.I32);
        int* cp = (int*)codes.DataPointer;
        uint s = 0x9E3779B9u;
        for (int i = 0; i < b * t; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            cp[i] = (int)((s >> 8) % 64000);
        }
        using Tensor zHat = new(new TensorShape(b, t, DvLevels.Length), DType.F32);
        Fsq.Dequantize(zHat, codes, DvLevels);
        using Tensor codes2 = new(new TensorShape(b, t), DType.I32);
        Fsq.CodesToIndices(codes2, zHat, DvLevels);

        int* c2 = (int*)codes2.DataPointer;
        for (int i = 0; i < b * t; i++) Assert.Equal(cp[i], c2[i]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void HaarWavelet3D_ForwardInverse_RoundTrips(int levels)
    {
        int div = 1 << levels;   // each level halves T/H/W
        int b = 1, c = 3, tt = 2 * div, h = 2 * div, w = 4 * div;
        using Tensor x = new(new TensorShape([(long)b, c, tt, h, w]), DType.F32);
        float* xp = (float*)x.DataPointer;
        uint s = 0xABCDEF01u;
        long n = (long)b * c * tt * h * w;
        for (long i = 0; i < n; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            xp[i] = (s >> 8) * (1f / 16777216f) * 2f - 1f;
        }

        Tensor fwd = HaarWavelet3D.Forward(x, levels);
        Assert.Equal(c * (long)Math.Pow(8, levels), fwd.Shape[1]);
        Assert.Equal(tt / div, (int)fwd.Shape[2]);
        Assert.Equal(h / div, (int)fwd.Shape[3]);
        Assert.Equal(w / div, (int)fwd.Shape[4]);

        Tensor inv = HaarWavelet3D.Inverse(fwd, levels);
        Assert.Equal(x.Shape.ToString(), inv.Shape.ToString());
        float* ip = (float*)inv.DataPointer;
        float maxErr = 0f;
        for (long i = 0; i < n; i++) maxErr = Math.Max(maxErr, Math.Abs(ip[i] - xp[i]));
        Assert.True(maxErr < 1e-4f, $"Haar round-trip max error {maxErr} exceeds 1e-4.");
        fwd.Dispose();
        inv.Dispose();
    }

    [Fact]
    public void CosmosDvTokenizer_LatentGrid_MatchesCosmosGeometry()
    {
        using CosmosDvTokenizer tok = new(CosmosDvTokenizerConfig.Dv8x16x16);
        Assert.Equal(64000, tok.VocabSize);
        // 33-frame 1024×640 clip → [5, 40, 64] = 12,800 tokens (research-doc §5).
        (int t, int h, int w) = tok.LatentGrid(33, 640, 1024);
        Assert.Equal((5, 40, 64), (t, h, w));
        Assert.Equal(12800, t * h * w);
        // 9-frame video prefix → [2, 40, 64] = 5,120; 1-frame image prefix → [1, 40, 64] = 2,560.
        Assert.Equal((2, 40, 64), tok.LatentGrid(9, 640, 1024));
        Assert.Equal((1, 40, 64), tok.LatentGrid(1, 640, 1024));
    }

    [Fact]
    public void CosmosDvTokenizer_CodesToContinuous_ProducesGridValues()
    {
        using CosmosDvTokenizer tok = new(CosmosDvTokenizerConfig.Dv8x16x16);
        int t = 2, h = 3, w = 4, n = t * h * w;
        using Tensor codes = new(new TensorShape(1, n), DType.I32);
        int* cp = (int*)codes.DataPointer;
        for (int i = 0; i < n; i++) cp[i] = (i * 997) % 64000;

        using Tensor continuous = tok.CodesToContinuous(codes, t, h, w);
        Assert.Equal(new[] { 1L, 6, t, h, w }, new[] { continuous.Shape[0], continuous.Shape[1], continuous.Shape[2], continuous.Shape[3], continuous.Shape[4] });
        // Every decoded value lands on the FSQ grid within [-1, 1].
        float* dp = (float*)continuous.DataPointer;
        for (long i = 0; i < continuous.Shape.ElementCount; i++) Assert.InRange(dp[i], -1f, 1f);
    }

    // Independent reference FSQ index for one 6-dim vector (mirrors lucidrains / Cosmos FSQuantizer).
    private static int RefQuantizeOne(float* z, int[] levels)
    {
        const float eps = 1e-6f;
        int code = 0, place = 1;
        for (int d = 0; d < levels.Length; d++)
        {
            int L = levels[d];
            float halfL = (L - 1) * (1f + eps) / 2f;
            float offset = (L % 2 == 0) ? 0.5f : 0f;
            float shift = MathF.Atan(offset / halfL);
            float bounded = MathF.Tanh(z[d] + shift) * halfL - offset;
            int rounded = (int)MathF.Round(bounded, MidpointRounding.ToEven);
            int shifted = Math.Clamp(rounded + L / 2, 0, L - 1);
            code += shifted * place;
            place *= L;
        }
        return code;
    }
}
