using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Temporal VAE tiling on the CPU. The mock decode is linear in time, so any window misalignment or weight error
/// shows up as a value error against the full decode.</summary>
public sealed unsafe class VideoTemporalTilingTests
{
    private const int Batch = 2;
    private const int Channels = 3;
    private const int Height = 2;
    private const int Width = 3;
    private const int Compression = 8;

    [Theory]
    [InlineData(40, 13, 8, 4, 8)]   // five windows, the last one 8 latent frames
    [InlineData(21, 13, 8, 4, 8)]   // exact fit: two full windows
    [InlineData(22, 13, 8, 4, 8)]   // three windows, the last one 6 latent frames
    [InlineData(13, 13, 8, 4, 8)]   // exactly one window
    [InlineData(7, 13, 8, 4, 8)]    // shorter than one window
    [InlineData(30, 13, 8, 2, 8)]   // half-width ramp
    [InlineData(9, 4, 2, 1, 2)]     // small geometry, minimal overlap
    [InlineData(17, 5, 2, 2, 3)]    // blend equal to the full overlap, non-power-of-two compression
    public void TiledDecode_MatchesFullDecode_AtEveryFrame(int frames, int tile, int stride, int blend, int compression)
    {
        using Tensor latent = Latent(frames, Batch, Channels, Height, Width);
        using Tensor full = InterpolateDecode(latent, compression);

        using Tensor tiled = VideoTemporalTiling.Decode(latent, Options(tile, stride, blend, compression),
            window => InterpolateDecode(window, compression));

        AssertClose(full, tiled, 1e-5f);
    }

    [Theory]
    [InlineData(40, 13, 8, 4, 8)]
    [InlineData(21, 13, 8, 4, 8)]
    [InlineData(22, 13, 8, 4, 8)]
    [InlineData(13, 13, 8, 4, 8)]
    [InlineData(7, 13, 8, 4, 8)]
    [InlineData(30, 13, 8, 2, 8)]
    [InlineData(9, 4, 2, 1, 2)]
    [InlineData(17, 5, 2, 2, 3)]
    public void BlendWeights_SumToOne_AtEveryOutputFrame(int frames, int tile, int stride, int blend, int compression)
    {
        using Tensor latent = Latent(frames, 1, 1, 1, 1);

        using Tensor tiled = VideoTemporalTiling.Decode(latent, Options(tile, stride, blend, compression),
            window => Filled(window.Shape[0], window.Shape[1], compression * ((int)window.Shape[2] - 1) + 1,
                window.Shape[3], window.Shape[4], 1f));

        float* p = (float*)tiled.DataPointer;
        int outFrames = compression * (frames - 1) + 1;
        Assert.Equal(outFrames, tiled.Shape[2]);
        for (int g = 0; g < outFrames; g++)
        {
            Assert.True(MathF.Abs(p[g] - 1f) <= 1e-6f, $"frame {g} sums to {p[g]}");
        }
    }

    [Fact]
    public void ShortFinalWindow_IsDecodedAtItsOwnLength()
    {
        List<int> lengths = [];
        using Tensor latent = Latent(22, Batch, Channels, Height, Width);

        using Tensor tiled = VideoTemporalTiling.Decode(latent, Options(13, 8, 4, Compression), window =>
        {
            lengths.Add((int)window.Shape[2]);
            return InterpolateDecode(window, Compression);
        });

        Assert.Equal(new int[] { 13, 13, 6 }, lengths);
        Assert.Equal(Compression * 21 + 1, tiled.Shape[2]);
    }

    [Fact]
    public void Blend_IsLinearRampOverEachOverlap()
    {
        // Latent frame t holds t and each window decodes to its first latent value, so window k (latent start 8k, decoded
        // start 64k) is constant 8k. Windows k-1 and k overlap on decoded frames 64k .. 64k+31, where the ramp runs from
        // 8(k-1) to 8k in steps of 1/32.
        const int frames = 40;
        using Tensor latent = FrameIndexLatent(frames);

        using Tensor tiled = VideoTemporalTiling.Decode(latent, Options(13, 8, 4, Compression),
            window => ConstantFromFirstFrame(window, Compression));

        int outFrames = Compression * (frames - 1) + 1;
        Assert.Equal(outFrames, tiled.Shape[2]);
        float* p = (float*)tiled.DataPointer;
        for (int g = 0; g < outFrames; g++)
        {
            int k = g / 64;
            int x = g % 64;
            float expected = k == 0 ? 0f : x < 32 ? ((32 - x) * 8f * (k - 1) + x * 8f * k) / 32f : 8f * k;
            Assert.True(MathF.Abs(p[g] - expected) <= 1e-4f, $"frame {g}: expected {expected}, got {p[g]}");
        }
    }

    [Fact]
    public void FromUpstreamSampleFrames_ConvertsDecodedSpansToLatentFrames()
    {
        VideoTemporalTilingOptions options = VideoTemporalTilingOptions.FromUpstreamSampleFrames(96, 64, Compression);

        Assert.Equal(13, options.TileLatentFrames);
        Assert.Equal(8, options.StrideLatentFrames);
        Assert.Equal(4, options.BlendLatentFrames);
        Assert.Equal(Compression, options.CompressionRatio);
        Assert.Throws<ArgumentException>(() => VideoTemporalTilingOptions.FromUpstreamSampleFrames(100, 64, Compression));
    }

    [Fact]
    public void FromUpstreamSampleFrames_RefusesSettingsThatCannotBeBlended()
    {
        // 128/32 decoded frames: a 16-interval window advancing 4 leaves a 12-interval overlap, but the 12-interval blend
        // would reach back past one stride, so a frame could be ramped by three windows.
        Assert.Throws<ArgumentException>(() => VideoTemporalTilingOptions.FromUpstreamSampleFrames(128, 32, Compression));
        // 64/64: the stride equals the window, so consecutive windows do not overlap at all.
        Assert.Throws<ArgumentException>(() => VideoTemporalTilingOptions.FromUpstreamSampleFrames(64, 64, Compression));
    }

    [Fact]
    public void ThrowingDelegate_ReleasesEveryWindowItWasGiven()
    {
        List<Tensor> seen = [];
        using Tensor latent = Latent(40, Batch, Channels, Height, Width);

        Assert.Throws<InvalidOperationException>(() => VideoTemporalTiling.Decode(latent, Options(13, 8, 4, Compression), window =>
        {
            seen.Add(window);
            return seen.Count == 2 ? throw new InvalidOperationException("decode failed") : InterpolateDecode(window, Compression);
        }));

        Assert.Equal(2, seen.Count);
        Assert.All(seen, window => Assert.Throws<ObjectDisposedException>(() => window.AsReadOnlySpan<float>()));
        Assert.True(latent.AsReadOnlySpan<float>().Length > 0, "The caller's latent was released.");
    }

    [Theory]
    [InlineData(40)]   // several windows
    [InlineData(7)]    // one window, passed through whole
    public void RejectedDelegateResult_IsReleased(int frames)
    {
        Tensor? rejected = null;
        using Tensor latent = Latent(frames, Batch, Channels, Height, Width);

        Assert.Throws<InvalidOperationException>(() => VideoTemporalTiling.Decode(latent, Options(13, 8, 4, Compression), window =>
        {
            rejected = Filled(window.Shape[0], window.Shape[1], (int)window.Shape[2], window.Shape[3], window.Shape[4], 0f);
            return rejected;
        }));

        Assert.NotNull(rejected);
        Assert.Throws<ObjectDisposedException>(() => rejected!.AsReadOnlySpan<float>());
        Assert.True(latent.AsReadOnlySpan<float>().Length > 0, "The caller's latent was released.");
    }

    [Fact]
    public void SingleWindowDelegateReturningItsInput_LeavesTheCallersLatentAlone()
    {
        using Tensor latent = Latent(7, Batch, Channels, Height, Width);

        Assert.Throws<InvalidOperationException>(() => VideoTemporalTiling.Decode(latent, Options(13, 8, 4, 1), window => window));

        Assert.True(latent.AsReadOnlySpan<float>().Length > 0, "The caller's latent was released.");
    }

    [Theory]
    [InlineData(13, 13, 1)]   // stride equal to the tile leaves no overlap
    [InlineData(13, 8, 5)]    // blend wider than the four-interval overlap
    [InlineData(13, 8, 0)]    // blend must cover at least one interval
    [InlineData(13, 0, 1)]    // stride must advance
    public void InvalidGeometry_IsRejectedBeforeAnyDecode(int tile, int stride, int blend)
    {
        using Tensor latent = Latent(40, Batch, Channels, Height, Width);

        Assert.Throws<ArgumentException>(() => VideoTemporalTiling.Decode(latent, Options(tile, stride, blend, Compression),
            window => throw new InvalidOperationException("The decode delegate must not run on invalid geometry.")));
    }

    [Fact]
    public void DelegateReturningWrongFrameCount_IsRejected()
    {
        using Tensor latent = Latent(40, Batch, Channels, Height, Width);

        Assert.Throws<InvalidOperationException>(() => VideoTemporalTiling.Decode(latent, Options(13, 8, 4, Compression),
            window => Filled(window.Shape[0], window.Shape[1], (int)window.Shape[2], window.Shape[3], window.Shape[4], 0f)));
    }

    [Fact]
    public void DelegateReturningItsInput_IsRejected()
    {
        using Tensor latent = Latent(40, Batch, Channels, Height, Width);

        Assert.Throws<InvalidOperationException>(() => VideoTemporalTiling.Decode(latent, Options(13, 8, 4, 1),
            window => window));
    }

    private static VideoTemporalTilingOptions Options(int tile, int stride, int blend, int compression) => new()
    {
        TileLatentFrames = tile,
        StrideLatentFrames = stride,
        BlendLatentFrames = blend,
        CompressionRatio = compression,
    };

    private static Tensor Latent(int frames, int batch, int channels, int height, int width)
    {
        Tensor latent = new Tensor(new TensorShape([batch, channels, frames, height, width]), DType.F32);
        float* p = (float*)latent.DataPointer;
        for (long i = 0; i < latent.ElementCount; i++)
        {
            p[i] = (float)Math.Sin(i * 0.731 + 0.3);
        }
        return latent;
    }

    private static Tensor FrameIndexLatent(int frames)
    {
        Tensor latent = new Tensor(new TensorShape([1, 1, frames, 1, 1]), DType.F32);
        float* p = (float*)latent.DataPointer;
        for (int t = 0; t < frames; t++)
        {
            p[t] = t;
        }
        return latent;
    }

    private static Tensor Filled(long batch, long channels, int frames, long height, long width, float value)
    {
        Tensor filled = new Tensor(new TensorShape([batch, channels, frames, height, width]), DType.F32);
        float* p = (float*)filled.DataPointer;
        for (long i = 0; i < filled.ElementCount; i++)
        {
            p[i] = value;
        }
        return filled;
    }

    /// <summary>Linear-in-time decode: output frame j sits at latent position j / C of the window it is given. The
    /// full-latent call is the reference, so a window's frames must match the full decode's frames at the same positions.</summary>
    private static Tensor InterpolateDecode(Tensor window, int compression)
    {
        long groups = window.Shape[0] * window.Shape[1];
        int frames = (int)window.Shape[2];
        int plane = (int)(window.Shape[3] * window.Shape[4]);
        int outFrames = compression * (frames - 1) + 1;
        Tensor output = new Tensor(new TensorShape([window.Shape[0], window.Shape[1], outFrames, window.Shape[3], window.Shape[4]]), DType.F32);

        float* src = (float*)window.DataPointer;
        float* dst = (float*)output.DataPointer;
        for (long g = 0; g < groups; g++)
        {
            for (int j = 0; j < outFrames; j++)
            {
                int p0 = j / compression;
                int p1 = Math.Min(p0 + 1, frames - 1);
                float f = (j % compression) / (float)compression;
                for (int q = 0; q < plane; q++)
                {
                    float a = src[(g * frames + p0) * plane + q];
                    float b = src[(g * frames + p1) * plane + q];
                    dst[(g * outFrames + j) * plane + q] = (1f - f) * a + f * b;
                }
            }
        }
        return output;
    }

    private static Tensor ConstantFromFirstFrame(Tensor window, int compression)
    {
        float value = ((float*)window.DataPointer)[0];
        int outFrames = compression * ((int)window.Shape[2] - 1) + 1;
        return Filled(window.Shape[0], window.Shape[1], outFrames, window.Shape[3], window.Shape[4], value);
    }

    private static void AssertClose(Tensor expected, Tensor actual, float tolerance)
    {
        Assert.Equal(expected.Shape.Rank, actual.Shape.Rank);
        for (int d = 0; d < expected.Shape.Rank; d++)
        {
            Assert.Equal(expected.Shape[d], actual.Shape[d]);
        }

        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        float worst = 0f;
        long worstAt = -1;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            float diff = MathF.Abs(e[i] - a[i]);
            if (diff > worst)
            {
                worst = diff;
                worstAt = i;
            }
        }
        Assert.True(worst <= tolerance, $"max |diff| {worst} at element {worstAt} exceeds {tolerance}");
    }
}
