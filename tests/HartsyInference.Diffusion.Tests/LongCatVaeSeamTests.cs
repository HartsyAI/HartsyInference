using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat VAE seam on stand-in encoder and decoder: the shapes the pipeline relies on (1 + 4·k frames in, the compressed grid out,
/// the decoded clip back), and the refusals for a bad clip, a bad latent, and an encoder or decoder that returns the wrong shape. The VAE's values
/// are not checked here; that needs the real weights.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatVaeSeamTests
{
    private const int Channels = 16;

    [Fact]
    public void AClipEncodesToTheCompressedGridAndDecodesBack()
    {
        using CpuBackend backend = new();
        StandIn vae = new();
        using Tensor clip = new(new TensorShape([1, 3, 13, 16, 16]), DType.F32);

        using Tensor latent = LongCatVaeSeam.EncodeClip(backend, vae, clip, Channels);
        Assert.Equal(new long[] { 1, Channels, 4, 2, 2 }, Shape(latent));

        using Tensor decoded = LongCatVaeSeam.DecodeLatents(backend, vae, latent, Channels);
        Assert.Equal(new long[] { 1, 3, 13, 16, 16 }, Shape(decoded));
    }

    [Fact]
    public void AClipWithANonCompressibleFrameCountIsRefused()
    {
        using CpuBackend backend = new();
        using Tensor clip = new(new TensorShape([1, 3, 12, 16, 16]), DType.F32);

        Assert.Throws<ArgumentException>(() => LongCatVaeSeam.EncodeClip(backend, new StandIn(), clip, Channels));
    }

    [Fact]
    public void ALatentWithTheWrongChannelCountIsRefused()
    {
        using CpuBackend backend = new();
        using Tensor latent = new(new TensorShape([1, 4, 4, 2, 2]), DType.F32);

        Assert.Throws<ArgumentException>(() => LongCatVaeSeam.DecodeLatents(backend, new StandIn(), latent, Channels));
    }

    [Fact]
    public void AnEncoderThatReturnsTheWrongShapeIsRefused()
    {
        using CpuBackend backend = new();
        using Tensor clip = new(new TensorShape([1, 3, 13, 16, 16]), DType.F32);
        StandIn wrong = new() { EncodedFrames = 3 };

        Assert.Throws<InvalidOperationException>(() => LongCatVaeSeam.EncodeClip(backend, wrong, clip, Channels));
    }

    [Fact]
    public void ADecoderThatReturnsTheWrongShapeIsRefused()
    {
        using CpuBackend backend = new();
        using Tensor latent = new(new TensorShape([1, Channels, 4, 2, 2]), DType.F32);
        StandIn wrong = new() { DecodedFrames = 12 };

        Assert.Throws<InvalidOperationException>(() => LongCatVaeSeam.DecodeLatents(backend, wrong, latent, Channels));
    }

    [Theory]
    [InlineData(12, 16)]
    [InlineData(13, 12)]
    public void AClipWhoseHeightOrWidthIsNotAMultipleOfEightIsRefused(int height, int width)
    {
        using CpuBackend backend = new();
        using Tensor clip = new(new TensorShape([1, 3, 13, height, width]), DType.F32);

        Assert.Throws<ArgumentException>(() => LongCatVaeSeam.EncodeClip(backend, new StandIn(), clip, Channels));
    }

    [Fact]
    public void AClipOfTheWrongRankBatchOrChannelCountIsRefused()
    {
        using CpuBackend backend = new();
        using Tensor rank4 = new(new TensorShape([1, 3, 13, 16]), DType.F32);
        using Tensor batch2 = new(new TensorShape([2, 3, 13, 16, 16]), DType.F32);
        using Tensor channels4 = new(new TensorShape([1, 4, 13, 16, 16]), DType.F32);

        Assert.Throws<ArgumentException>(() => LongCatVaeSeam.EncodeClip(backend, new StandIn(), rank4, Channels));
        Assert.Throws<ArgumentException>(() => LongCatVaeSeam.EncodeClip(backend, new StandIn(), batch2, Channels));
        Assert.Throws<ArgumentException>(() => LongCatVaeSeam.EncodeClip(backend, new StandIn(), channels4, Channels));
    }

    [Fact]
    public void ALatentOfTheWrongRankOrBatchIsRefused()
    {
        using CpuBackend backend = new();
        using Tensor rank4 = new(new TensorShape([1, Channels, 4, 2]), DType.F32);
        using Tensor batch2 = new(new TensorShape([2, Channels, 4, 2, 2]), DType.F32);

        Assert.Throws<ArgumentException>(() => LongCatVaeSeam.DecodeLatents(backend, new StandIn(), rank4, Channels));
        Assert.Throws<ArgumentException>(() => LongCatVaeSeam.DecodeLatents(backend, new StandIn(), batch2, Channels));
    }

    [Fact]
    public void ANonPositiveLatentChannelCountIsRefusedUpFront()
    {
        using CpuBackend backend = new();
        using Tensor clip = new(new TensorShape([1, 3, 13, 16, 16]), DType.F32);
        using Tensor latent = new(new TensorShape([1, Channels, 4, 2, 2]), DType.F32);

        Assert.Throws<ArgumentOutOfRangeException>(() => LongCatVaeSeam.EncodeClip(backend, new StandIn(), clip, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => LongCatVaeSeam.DecodeLatents(backend, new StandIn(), latent, -1));
    }

    private static long[] Shape(Tensor tensor)
    {
        long[] shape = new long[tensor.Shape.Rank];
        for (int i = 0; i < shape.Length; i++)
            shape[i] = tensor.Shape[i];
        return shape;
    }

    /// <summary>Returns tensors of the Wan VAE's shapes, without any weights.</summary>
    private sealed class StandIn : IWanVaeEncoder, IWanVaeDecoder
    {
        /// <summary>The frame count the encoder returns; the right value is the clip's 1 + (T−1)/4.</summary>
        public long EncodedFrames { get; init; } = -1;

        public Tensor Encode(IBackend backend, Tensor rgbClip)
        {
            long frames = EncodedFrames >= 0 ? EncodedFrames : (rgbClip.Shape[2] - 1) / 4 + 1;
            return new Tensor(new TensorShape([1, Channels, frames, rgbClip.Shape[3] / 8, rgbClip.Shape[4] / 8]), DType.F32);
        }

        /// <summary>The frame count the decoder returns; the right value is 1 + 4·(F−1).</summary>
        public long DecodedFrames { get; init; } = -1;

        public Tensor Decode(IBackend backend, Tensor latent)
        {
            long frames = DecodedFrames >= 0 ? DecodedFrames : 1 + 4 * (latent.Shape[2] - 1);
            return new Tensor(new TensorShape([1, 3, frames, latent.Shape[3] * 8, latent.Shape[4] * 8]), DType.F32);
        }

        public Tensor EncodeRgbFrame(IBackend backend, ReadOnlySpan<byte> rgb24, int width, int height) =>
            throw new NotSupportedException("The seam test uses the clip path only.");

        public IEnumerable<Tensor> EnumerateWeights() => [];
    }
}
