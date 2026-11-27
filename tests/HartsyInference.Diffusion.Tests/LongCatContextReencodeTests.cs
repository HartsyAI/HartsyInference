using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The 1.5 chain's context is the segment's last decoded pixel frames, re-encoded (use_vcond False in meituan-longcat/LongCat-Video @
/// 58eb336, pipeline 1467-1490). Tiny config with a deterministic stand-in VAE: the decoder maps each pixel frame to one latent frame, and the
/// encoder records the clip it was given, so the test can check which frames went in.</summary>
public sealed class LongCatContextReencodeTests
{
    private const int Channels = 4, SegmentFrames = 7, ContextLatents = 4, LatentH = 2, LatentW = 2;
    private const int PixelH = LatentH * 8, PixelW = LatentW * 8;

    // The RGB24 round trip of the decoded frames quantizes to 1/127.5 in [-1, 1]; the bound allows one step and rounding.
    private const float Tolerance = 0.01f;

    [Fact]
    public void TheContextIsTheEncodedLastDecodedFrames()
    {
        using CpuBackend backend = new();
        Random rng = new(11);
        float[] segment = new float[Channels * SegmentFrames * LatentH * LatentW];
        for (int i = 0; i < segment.Length; i++)
            segment[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        RecordingVae vae = new();

        float[] context = LongCatRecipeSupport.ReencodedContext(backend, vae, vae, segment, Channels, SegmentFrames, ContextLatents, LatentH, LatentW);

        // The encoder got the last 13 of the 25 decoded pixel frames (pixels 12…24), not the carried latents.
        Assert.Equal(13, vae.ClipFrames);
        Assert.Equal(PixelH, vae.ClipHeight);
        Assert.Equal(PixelW, vae.ClipWidth);
        Assert.Equal(Channels * ContextLatents * LatentH * LatentW, context.Length);

        using Tensor latent = new(new TensorShape([1L, Channels, SegmentFrames, LatentH, LatentW]), DType.F32);
        segment.AsSpan().CopyTo(latent.AsSpan<float>());
        using Tensor decoded = vae.Decode(backend, latent);
        ReadOnlySpan<float> pixels = decoded.AsReadOnlySpan<float>();
        int plane = PixelH * PixelW, decodedFrames = 1 + 4 * (SegmentFrames - 1);
        for (int ch = 0; ch < 3; ch++)
            for (int t = 0; t < vae.ClipFrames; t++)
                for (int e = 0; e < plane; e++)
                {
                    float expected = pixels[(ch * decodedFrames + (decodedFrames - vae.ClipFrames + t)) * plane + e];
                    float actual = vae.ClipValues[(ch * vae.ClipFrames + t) * plane + e];
                    Assert.True(MathF.Abs(expected - actual) <= Tolerance, $"clip frame {t}, channel {ch}, value {e}: {actual} vs decoded {expected}");
                }

        // The context is the encoder's latent of that clip: latent frame f takes the clip's pixel frame 4f.
        for (int c = 0; c < Channels; c++)
            for (int f = 0; f < ContextLatents; f++)
                for (int y = 0; y < LatentH; y++)
                    for (int x = 0; x < LatentW; x++)
                    {
                        float expected = vae.ClipValues[((c % 3) * vae.ClipFrames + 4 * f) * plane + (y * 8) * PixelW + x * 8];
                        Assert.Equal(expected, context[((c * ContextLatents + f) * LatentH + y) * LatentW + x]);
                    }
    }

    /// <summary>Deterministic VAE: the decoder maps pixel frame <c>p</c> to latent frame <c>(p + 3) / 4</c> (frame 0 to 0), and the encoder
    /// takes pixel frame <c>4f</c> of its clip. The encoder records the clip it was given (its frames, height and width, and its values).</summary>
    private sealed class RecordingVae : IWanVaeEncoder, IWanVaeDecoder
    {
        public float[] ClipValues { get; private set; } = [];
        public int ClipFrames { get; private set; }
        public int ClipHeight { get; private set; }
        public int ClipWidth { get; private set; }

        public Tensor Decode(IBackend backend, Tensor latent)
        {
            long frames = latent.Shape[2], h = latent.Shape[3], w = latent.Shape[4];
            long pixels = 1 + 4 * (frames - 1), height = h * 8, width = w * 8;
            Tensor clip = new(new TensorShape([1, 3, pixels, height, width]), DType.F32);
            ReadOnlySpan<float> source = latent.AsReadOnlySpan<float>();
            Span<float> target = clip.AsSpan<float>();
            int channels = (int)latent.Shape[1];
            for (int ch = 0; ch < 3; ch++)
                for (int p = 0; p < pixels; p++)
                {
                    long lat = p == 0 ? 0 : (p + 3) / 4;
                    for (int y = 0; y < height; y++)
                        for (int x = 0; x < width; x++)
                            target[(int)(((ch * pixels + p) * height + y) * width + x)] =
                                0.5f * source[(int)(((ch % channels * frames + lat) * h + y / 8) * w + x / 8)];
                }
            return clip;
        }

        public Tensor Encode(IBackend backend, Tensor rgbClip)
        {
            long frames = rgbClip.Shape[2], height = rgbClip.Shape[3], width = rgbClip.Shape[4];
            ReadOnlySpan<float> clip = rgbClip.AsReadOnlySpan<float>();
            ClipValues = clip.ToArray();
            ClipFrames = (int)frames;
            ClipHeight = (int)height;
            ClipWidth = (int)width;
            long latentFrames = (frames - 1) / 4 + 1, h = height / 8, w = width / 8;
            Tensor latent = new(new TensorShape([1, Channels, latentFrames, h, w]), DType.F32);
            Span<float> target = latent.AsSpan<float>();
            for (int c = 0; c < Channels; c++)
                for (int f = 0; f < latentFrames; f++)
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                            target[(int)(((c * latentFrames + f) * h + y) * w + x)] =
                                clip[(int)((((c % 3) * frames + 4 * f) * height + y * 8) * width + x * 8)];
            return latent;
        }

        public Tensor EncodeRgbFrame(IBackend backend, ReadOnlySpan<byte> rgb24, int width, int height) =>
            throw new NotSupportedException("The context test uses the clip path only.");

        public IEnumerable<Tensor> EnumerateWeights() => [];
    }
}
