using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tests for the tiled VAE encoder geometry and the shared <see cref="VaeTiling"/> primitives
/// (extract / linear-ramp blend / crop-concat) that both <see cref="VaeTiledEncoder"/> and
/// <see cref="VaeTiledDecoder"/> rely on. These run without any VAE checkpoint; full forward parity
/// against a real VAE is covered by the checkpoint-gated SSIM tests.</summary>
public sealed class VaeTiledEncoderTests
{
    private const float Tolerance = 1e-5f;

    // ── Geometry ─────────────────────────────────────────────────────────

    [Fact]
    public void TiledEncoder_PixelTileGeometry_Sd15()
    {
        // SD1.5: sample_size=512, 4 block stages → tile_latent=64. Encoder tiles in PIXEL space:
        // pixelTile = 64*8 = 512, step = 48*8 = 384, latent blend = 16.
        VaeConfig config = VaeConfig.Sd15;
        const float overlapFactor = 0.25f;

        int tileLatentSize = config.SampleSize / (int)Math.Pow(2, config.BlockOutChannels.Length - 1);
        int latentOverlapStep = (int)(tileLatentSize * (1.0f - overlapFactor));
        int latentBlendExtent = (int)(tileLatentSize * overlapFactor);
        int pixelTileSize = tileLatentSize * 8;
        int pixelOverlapStep = latentOverlapStep * 8;

        Assert.Equal(64, tileLatentSize);
        Assert.Equal(512, pixelTileSize);
        Assert.Equal(384, pixelOverlapStep);
        Assert.Equal(16, latentBlendExtent);
    }

    // ── ExtractTile ──────────────────────────────────────────────────────

    [Fact]
    public unsafe void ExtractTile_SlicesExpectedRegion()
    {
        // Source [1,1,4,4] = 0..15 row-major. Extract 2x2 at (1,1) → [5,6,9,10].
        Tensor src = new Tensor(new TensorShape(1, 1, 4, 4), DType.F32);
        Span<float> s = src.AsSpan<float>();
        for (int i = 0; i < 16; i++) s[i] = i;

        Tensor tile = VaeTiling.ExtractTile(src, batch: 1, channels: 1, startH: 1, startW: 1, tileH: 2, tileW: 2);
        ReadOnlySpan<float> t = tile.AsReadOnlySpan<float>();

        Assert.Equal(5f, t[0]);
        Assert.Equal(6f, t[1]);
        Assert.Equal(9f, t[2]);
        Assert.Equal(10f, t[3]);

        src.Dispose();
        tile.Dispose();
    }

    // ── BlendHorizontal ──────────────────────────────────────────────────

    [Fact]
    public unsafe void BlendHorizontal_LinearRamp()
    {
        // left=10, right=20, blendExtent=4 → right row becomes [10, 12.5, 15, 17.5].
        Tensor left = Constant(1, 1, 2, 4, 10f);
        Tensor right = Constant(1, 1, 2, 4, 20f);

        VaeTiling.BlendHorizontal(left, right, blendExtent: 4);
        ReadOnlySpan<float> r = right.AsReadOnlySpan<float>();

        Assert.InRange(r[0], 10f - Tolerance, 10f + Tolerance);
        Assert.InRange(r[1], 12.5f - Tolerance, 12.5f + Tolerance);
        Assert.InRange(r[2], 15f - Tolerance, 15f + Tolerance);
        Assert.InRange(r[3], 17.5f - Tolerance, 17.5f + Tolerance);

        left.Dispose();
        right.Dispose();
    }

    [Fact]
    public unsafe void BlendHorizontal_EqualValues_AreSeamless()
    {
        // A flat field must survive blending unchanged — this is the "no visible seam" guarantee.
        Tensor left = Constant(1, 3, 4, 8, 5f);
        Tensor right = Constant(1, 3, 4, 8, 5f);

        VaeTiling.BlendHorizontal(left, right, blendExtent: 4);
        ReadOnlySpan<float> r = right.AsReadOnlySpan<float>();
        foreach (float v in r) Assert.InRange(v, 5f - Tolerance, 5f + Tolerance);

        left.Dispose();
        right.Dispose();
    }

    // ── Concat ───────────────────────────────────────────────────────────

    [Fact]
    public unsafe void ConcatHorizontal_CropsAllButLastToRowLimit()
    {
        // Two [1,1,2,4] tiles, rowLimit=2 → width = min(2,4) + 4 = 6.
        Tensor a = Constant(1, 1, 2, 4, 1f);
        Tensor b = Constant(1, 1, 2, 4, 2f);

        Tensor result = VaeTiling.ConcatHorizontal([a, b], rowLimit: 2, batch: 1);
        Assert.Equal(6, (int)result.Shape[3]);
        Assert.Equal(2, (int)result.Shape[2]);

        a.Dispose();
        b.Dispose();
        result.Dispose();
    }

    [Fact]
    public unsafe void ConcatVertical_CropsAllButLastToRowLimit()
    {
        Tensor a = Constant(1, 1, 4, 3, 1f);
        Tensor b = Constant(1, 1, 4, 3, 2f);

        Tensor result = VaeTiling.ConcatVertical([a, b], rowLimit: 2, batch: 1);
        Assert.Equal(6, (int)result.Shape[2]); // min(2,4) + 4
        Assert.Equal(3, (int)result.Shape[3]);

        a.Dispose();
        b.Dispose();
        result.Dispose();
    }

    // ── Construction ─────────────────────────────────────────────────────

    [Fact]
    public void VaeTiledEncoder_WrapsEncoder_Constructs()
    {
        VaeEncoder encoder = new VaeEncoder(VaeConfig.Sd15);
        VaeTiledEncoder tiled = new VaeTiledEncoder(encoder);
        Assert.NotNull(tiled);
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private static unsafe Tensor Constant(int b, int c, int h, int w, float value)
    {
        Tensor t = new Tensor(new TensorShape(b, c, h, w), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = value;
        return t;
    }
}
