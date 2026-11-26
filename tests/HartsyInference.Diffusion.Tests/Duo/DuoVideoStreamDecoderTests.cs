using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>The Duo video stream decoder with a fake decoder: the windows it hands the decoder hold the right latents, the blocks come out
/// in order with the right crops, and the RGB mapping is the one derived from upstream. The H3 decoder is not run here; parity against
/// upstream is not run.</summary>
public sealed class DuoVideoStreamDecoderTests
{
    private const int Height = 2, Width = 2, Plane = Height * Width;

    [Fact]
    public void TheFirstChunkDecodesNothing()
    {
        List<float[]> windows = [];
        DuoVideoStreamDecoder decoder = new(FakeDecoder(windows));

        Assert.Empty(decoder.OnChunk(Chunk(0)));
        Assert.Empty(windows);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    public void EveryChunkPublishesOneBlockAndEndPublishesTheLastOnce(int chunks)
    {
        List<float[]> windows = [];
        DuoVideoStreamDecoder decoder = new(FakeDecoder(windows));
        List<DuoDecodedBlock> blocks = [];
        for (int c = 0; c < chunks; c++)
        {
            blocks.AddRange(decoder.OnChunk(Chunk(c)));
        }
        blocks.AddRange(decoder.End());

        Assert.Equal(Enumerable.Range(0, chunks), blocks.Select(b => b.Block));
        Assert.Equal(chunks, windows.Count);
        Assert.All(blocks, b => Assert.Equal(17, b.Frames));
        Assert.All(blocks, b => Assert.Equal(Height, b.Height));
    }

    [Fact]
    public void WindowsHoldTheRightLatentsAtEveryStep()
    {
        List<float[]> windows = [];
        DuoVideoStreamDecoder decoder = new(FakeDecoder(windows));
        decoder.OnChunk(Chunk(0));
        decoder.OnChunk(Chunk(1));
        decoder.OnChunk(Chunk(2));
        decoder.End();

        // Chunk 1: all of chunk 0, then chunk 1's first two latents (7 latents).
        Assert.Equal(Expected((0, 0, 5), (1, 0, 2)), windows[0]);
        // Chunk 2: chunks 0 and 1 whole, then chunk 2's first two latents (12 latents).
        Assert.Equal(Expected((0, 0, 5), (1, 0, 5), (2, 0, 2)), windows[1]);
        // End after three chunks: chunks 1 and 2 whole, then chunk 2's last two latents (12 latents).
        Assert.Equal(Expected((1, 0, 5), (2, 0, 5), (2, 3, 2)), windows[2]);
    }

    [Fact]
    public void EndAfterOneChunkUsesItsOwnLastTwoLatents()
    {
        List<float[]> windows = [];
        DuoVideoStreamDecoder decoder = new(FakeDecoder(windows));
        decoder.OnChunk(Chunk(0));
        List<DuoDecodedBlock> blocks = decoder.End().ToList();

        Assert.Equal(Expected((0, 0, 5), (0, 3, 2)), windows[0]);
        Assert.Single(blocks);
        Assert.Equal(0, blocks[0].Block);
    }

    [Fact]
    public void EachBlockIsTheCroppedFramesOfItsWindowInRgbAsDerivedFromUpstream()
    {
        List<float[]> windows = [];
        DuoVideoStreamDecoder decoder = new(FakeDecoder(windows));
        decoder.OnChunk(Chunk(0));
        DuoDecodedBlock first = decoder.OnChunk(Chunk(1)).Single();
        DuoDecodedBlock second = decoder.OnChunk(Chunk(2)).Single();

        // Block 0 is raw frames 0..16 of the 22-frame window; block 1 is raw frames 17..33 of the 39-frame window.
        Assert.Equal(ExpectedRgb(raw: 22, crop: 0, frames: 17), first.Rgb);
        Assert.Equal(ExpectedRgb(raw: 39, crop: 17, frames: 17), second.Rgb);
    }

    [Theory]
    [InlineData(-1f, 0)]
    [InlineData(-0.0001f, 127)]
    [InlineData(0f, 128)]
    [InlineData(1f, 255)]
    [InlineData(2f, 255)]
    [InlineData(-2f, 0)]
    public void RgbIsRoundedAndClamped(float x, byte expected)
    {
        Assert.Equal(expected, DuoVideoStreamDecoder.ToByte(x));
    }

    [Fact]
    public void AChunkOfTheWrongShapeIsRefused()
    {
        DuoVideoStreamDecoder decoder = new(FakeDecoder([]));
        Assert.Throws<ArgumentException>(() => decoder.OnChunk(new Tensor(new TensorShape(new long[] { 1, 24, 4, Height, Width }), DType.F32)));
    }

    [Fact]
    public void HistoryStaysBoundedOverALongStream()
    {
        List<float[]> windows = [];
        DuoVideoStreamDecoder decoder = new(FakeDecoder(windows));
        for (int c = 0; c < 12; c++)
        {
            decoder.OnChunk(Chunk(c));
            Assert.True(decoder.RetainedChunks <= 2, $"After chunk {c} the decoder holds {decoder.RetainedChunks} chunks.");
        }
        decoder.End();
        Assert.True(decoder.RetainedChunks <= 2);
    }

    [Fact]
    public void AWindowAtFourChunksSlidesToChunksOneAndTwo()
    {
        List<float[]> windows = [];
        DuoVideoStreamDecoder decoder = new(FakeDecoder(windows));
        for (int c = 0; c < 4; c++)
        {
            decoder.OnChunk(Chunk(c));
        }

        Assert.Equal(Expected((1, 0, 5), (2, 0, 5), (3, 0, 2)), windows[2]);
    }

    [Fact]
    public void AChunkOfAnotherTypeOrSizeIsRefused()
    {
        DuoVideoStreamDecoder decoder = new(FakeDecoder([]));
        decoder.OnChunk(Chunk(0));
        Assert.Throws<ArgumentException>(() => decoder.OnChunk(new Tensor(new TensorShape(new long[] { 1, 24, 5, Height, Width }), DType.F16)));
        Assert.Throws<ArgumentException>(() => decoder.OnChunk(new Tensor(new TensorShape(new long[] { 1, 24, 5, Height + 2, Width }), DType.F32)));
    }

    [Fact]
    public void ChunksAfterEndAndASecondEndAreRefused()
    {
        DuoVideoStreamDecoder decoder = new(FakeDecoder([]));
        decoder.OnChunk(Chunk(0));
        decoder.End();
        Assert.Throws<InvalidOperationException>(() => decoder.OnChunk(Chunk(1)));
        Assert.Throws<InvalidOperationException>(() => decoder.End());
    }

    [Fact]
    public void AFailedDecodeLeavesTheStreamAsItWasAndTheRetryPublishesTheRightBlock()
    {
        List<float[]> windows = [];
        bool fail = false;
        Func<Tensor, Tensor> inner = FakeDecoder(windows);
        DuoVideoStreamDecoder decoder = new(window => fail ? throw new InvalidOperationException("decoder down") : inner(window));
        decoder.OnChunk(Chunk(0));
        int retained = decoder.RetainedChunks;

        fail = true;
        Assert.Throws<InvalidOperationException>(() => decoder.OnChunk(Chunk(1)));
        Assert.Equal(retained, decoder.RetainedChunks);
        Assert.Empty(windows);

        fail = false;
        DuoDecodedBlock block = decoder.OnChunk(Chunk(1)).Single();
        Assert.Equal(0, block.Block);
        Assert.Equal(Expected((0, 0, 5), (1, 0, 2)), windows[0]);
    }

    [Fact]
    public void AFailedEndCanBeRetried()
    {
        List<float[]> windows = [];
        bool fail = false;
        Func<Tensor, Tensor> inner = FakeDecoder(windows);
        DuoVideoStreamDecoder decoder = new(window => fail ? throw new InvalidOperationException("decoder down") : inner(window));
        decoder.OnChunk(Chunk(0));
        decoder.OnChunk(Chunk(1));

        fail = true;
        Assert.Throws<InvalidOperationException>(() => decoder.End());
        fail = false;
        DuoDecodedBlock last = decoder.End().Single();
        Assert.Equal(1, last.Block);
        Assert.Equal(Expected((0, 0, 5), (1, 0, 5), (1, 3, 2)), windows[^1]);
        Assert.Throws<InvalidOperationException>(() => decoder.End());
    }

    private static Func<Tensor, Tensor> FakeDecoder(List<float[]> windows) => window =>
    {
        windows.Add(window.AsSpan<float>().ToArray());
        int frames = DuoVideoDecodePlan.RawFramesFor((int)window.Shape[2]);
        int h = (int)window.Shape[3], w = (int)window.Shape[4], plane = h * w;
        Tensor output = new(new TensorShape(new long[] { 1, 3, frames, h, w }), DType.F32);
        Span<float> o = output.AsSpan<float>();
        for (int c = 0; c < 3; c++)
        {
            for (int f = 0; f < frames; f++)
            {
                for (int p = 0; p < plane; p++)
                {
                    o[((c * frames) + f) * plane + p] = Pixel(f, c, p);
                }
            }
        }
        return output;
    };

    /// <summary>Chunk <paramref name="index"/> of <c>[1, 24, 5, 2, 2]</c>, valued so every latent is distinct and exact in F32.</summary>
    private static Tensor Chunk(int index)
    {
        Tensor chunk = new(new TensorShape(new long[] { 1, 24, 5, Height, Width }), DType.F32);
        Span<float> s = chunk.AsSpan<float>();
        for (int c = 0; c < 24; c++)
        {
            for (int t = 0; t < 5; t++)
            {
                for (int p = 0; p < Plane; p++)
                {
                    s[((c * 5) + t) * Plane + p] = LatentValue(index, c, t, p);
                }
            }
        }
        return chunk;
    }

    private static float LatentValue(int chunk, int c, int t, int p) => chunk * 1000 + c * 100 + t * 10 + p;

    private static float Pixel(int f, int c, int p) => -1f + 2f * ((f * 7 + c * 3 + p) % 11) / 10f;

    /// <summary>The window the decoder must receive for these <c>(chunk, first latent, count)</c> segments, in layout <c>[24][T][H][W]</c>.</summary>
    private static float[] Expected(params (int Chunk, int Start, int Count)[] segments)
    {
        int total = segments.Sum(s => s.Count);
        float[] result = new float[24 * total * Plane];
        for (int c = 0; c < 24; c++)
        {
            int t = 0;
            foreach ((int chunk, int start, int count) in segments)
            {
                for (int k = 0; k < count; k++, t++)
                {
                    for (int p = 0; p < Plane; p++)
                    {
                        result[((c * total) + t) * Plane + p] = LatentValue(chunk, c, start + k, p);
                    }
                }
            }
        }
        return result;
    }

    private static byte[] ExpectedRgb(int raw, int crop, int frames)
    {
        byte[] rgb = new byte[frames * Plane * 3];
        for (int f = 0; f < frames; f++)
        {
            for (int p = 0; p < Plane; p++)
            {
                for (int c = 0; c < 3; c++)
                {
                    rgb[((f * Plane) + p) * 3 + c] = DuoVideoStreamDecoder.ToByte(Pixel(crop + f, c, p));
                }
            }
        }
        return rgb;
    }
}
