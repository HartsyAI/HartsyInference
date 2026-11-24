using HartsyInference.Audio.Models.Separation;
using Xunit;

namespace HartsyInference.Audio.Tests.Separation;

public sealed class MdxSeparatorTests
{
    private sealed class IdentityModel : IMdxSpectrogramModel
    {
        public int Calls { get; private set; }
        public List<float[]> Spectrograms { get; } = [];

        public void Run(float[] spectrogram, float[] prediction)
        {
            Calls++;
            Spectrograms.Add((float[])spectrogram.Clone());
            spectrogram.CopyTo(prediction, 0);
        }
    }

    /// <summary>A short-chunk configuration (chunk 1984, trim 128) so a few thousand samples span several overlapping chunks. The full one-sided spectrum
    /// (129 bins) and no zeroed bins make an identity network an exact round trip.</summary>
    private static MdxSettings Small() => new() { NFft = 256, Hop = 64, DimF = 129, Frames = 32, Compensate = 1f, ZeroLowBins = 0, SampleRate = 44_100 };

    private static float[] Tones(int n, double peak)
    {
        float[] x = new float[n];
        for (int i = 0; i < n; i++)
        {
            double t = i / 44_100.0;
            x[i] = (float)(peak * (0.5 * Math.Sin(2 * Math.PI * 2000 * t) + 0.3 * Math.Sin(2 * Math.PI * 5300 * t + 1) + 0.2 * Math.Sin(2 * Math.PI * 9100 * t + 2)));
        }
        return x;
    }

    private static double RelRms(float[] expected, float[] actual)
    {
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = expected[i] - actual[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(num / den);
    }

    [Fact]
    public void IdentityNetworkReturnsTheInputAcrossSeveralOverlappingChunks()
    {
        IdentityModel model = new();
        MdxSeparator separator = new(Small(), model);
        float[] left = Tones(9000, 0.5), right = Tones(9000, 0.4);

        float[][] stem = separator.Separate(left, right);

        Assert.True(model.Calls >= 4, $"expected several chunks, got {model.Calls}");
        Assert.Equal(left.Length, stem[0].Length);
        Assert.True(RelRms(left, stem[0]) < 2e-5, $"left relRMS {RelRms(left, stem[0]):E2}");
        Assert.True(RelRms(right, stem[1]) < 2e-5, $"right relRMS {RelRms(right, stem[1]):E2}");
    }

    [Fact]
    public void DroppingHighBinsAndLowBinsOnlyRemovesThatBand()
    {
        MdxSettings band = Small() with { DimF = 100, ZeroLowBins = 3 };
        MdxSeparator separator = new(band, new IdentityModel());
        float[] left = Tones(9000, 0.5);
        float[][] stem = separator.Separate(left, left);
        // Tones sit at bins 11.6, 30.8 and 52.9; the cut bins hold only the edge discontinuities.
        Assert.True(RelRms(left, stem[0]) < 3e-2, $"relRMS {RelRms(left, stem[0]):E2}");
    }

    [Fact]
    public void ChunkCountFollowsPaddingAndStep()
    {
        IdentityModel model = new();
        MdxSettings s = Small();
        MdxSeparator separator = new(s, model);
        int n = 9000;
        int gen = s.ChunkSize - 2 * s.Trim;
        int total = s.Trim + n + (gen + s.Trim - n % gen);
        int step = (int)(0.75 * s.ChunkSize);
        int expected = (total + step - 1) / step;
        int reported = 0;

        separator.Separate(Tones(n, 0.5), Tones(n, 0.5), progress: (done, count) => reported = count);

        Assert.Equal(expected, model.Calls);
        Assert.Equal(expected, reported);
    }

    [Fact]
    public void LoudMixIsNormalizedToThePeakAndQuietMixIsNotAmplified()
    {
        MdxSeparator loud = new(Small(), new IdentityModel());
        float[][] stem = loud.Separate(Tones(6000, 3.0), Tones(6000, 3.0));
        float max = 0;
        foreach (float[] c in stem) foreach (float v in c) max = Math.Max(max, Math.Abs(v));
        Assert.InRange(max, 0.88f, 0.9005f);

        MdxSeparator quiet = new(Small(), new IdentityModel());
        float[] input = Tones(6000, 0.4);
        float[][] kept = quiet.Separate(input, input);
        Assert.True(RelRms(input, kept[0]) < 2e-5);
    }

    [Fact]
    public void CompensationScalesTheStemBeforeItsPeakNormalization()
    {
        MdxSettings s = Small() with { Compensate = 1.25f };
        MdxSeparator separator = new(s, new IdentityModel());
        float[] input = Tones(6000, 0.4);
        float[][] stem = separator.Separate(input, input);
        float[] expected = new float[input.Length];
        for (int i = 0; i < expected.Length; i++) expected[i] = input[i] * 1.25f;
        Assert.True(RelRms(expected, stem[0]) < 2e-5);
    }

    [Fact]
    public void SpectrogramLayoutIsChannelsOfLeftRealImagThenRightRealImag()
    {
        // Real size: n_fft 7680 is not a power of two, so this also checks the mixed-radix transform against a direct DFT.
        MdxSettings s = new();
        IdentityModel model = new();
        MdxSeparator separator = new(s, model);
        int n = 80_000;
        float[] left = new float[n], right = new float[n];
        Random rng = new(5);
        for (int i = 0; i < n; i++) left[i] = 0.2f * (float)(rng.NextDouble() - 0.5);

        separator.Separate(left, right);

        float[] spec = model.Spectrograms[1];                        // a chunk away from the zero-padded start
        // Rebuild what chunk 1 holds: the peak-normalized, padded mix from step samples in, reflect-padded around the chunk.
        float peak = 0;
        foreach (float v in left) peak = Math.Max(peak, Math.Abs(v));
        float scale = peak > 0.9f ? 0.9f / peak : 1f;
        int step = (int)(0.75 * s.ChunkSize);
        float[] chunk = new float[s.ChunkSize];
        for (int i = 0; i < chunk.Length; i++)
        {
            int idx = step + i - s.Trim;
            chunk[i] = idx >= 0 && idx < n ? left[idx] * scale : 0f;
        }
        int frame = 17;
        double[] reals = new double[3], imags = new double[3];
        int[] bins = [3, 100, 3071];
        for (int b = 0; b < bins.Length; b++)
        {
            double re = 0, im = 0;
            for (int i = 0; i < s.NFft; i++)
            {
                int pos = frame * s.Hop + i - s.NFft / 2;               // centre=True reflect padding
                int len = chunk.Length;
                int period = 2 * (len - 1);
                int m = ((pos % period) + period) % period;
                double sample = chunk[m < len ? m : period - m];
                double w = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / s.NFft);
                double phase = -2 * Math.PI * bins[b] * i / s.NFft;
                re += sample * w * Math.Cos(phase);
                im += sample * w * Math.Sin(phase);
            }
            reals[b] = re;
            imags[b] = im;
        }
        double tolerance = 1e-3 * Math.Max(Math.Abs(reals[1]), Math.Abs(imags[1]));
        for (int b = 0; b < bins.Length; b++)
        {
            Assert.InRange(spec[(0 * s.DimF + bins[b]) * s.Frames + frame], reals[b] - tolerance, reals[b] + tolerance);
            Assert.InRange(spec[(1 * s.DimF + bins[b]) * s.Frames + frame], imags[b] - tolerance, imags[b] + tolerance);
        }
        for (int i = 2 * s.DimF * s.Frames; i < spec.Length; i++) Assert.Equal(0f, spec[i]);   // right channel was silent
        for (int c = 0; c < 4; c++)
            for (int k = 0; k < s.ZeroLowBins; k++)
                for (int f = 0; f < s.Frames; f++) Assert.Equal(0f, spec[(c * s.DimF + k) * s.Frames + f]);
    }

    [Fact]
    public void SymmetricHannMatchesNumpy()
    {
        Assert.Equal([1.0], MdxSeparator.SymmetricHann(1));
        double[] w = MdxSeparator.SymmetricHann(5);
        Assert.Equal([0.0, 0.5, 1.0, 0.5, 0.0], w.Select(v => Math.Round(v, 12)));
    }

    [Fact]
    public void MonoMixResamplesTo16kAndCrops()
    {
        MdxSeparator separator = new(Small(), new IdentityModel());
        float[] a = Tones(44_100, 0.5);
        float[] mono = separator.ToMono([a, a], 16_000, targetSamplesAtSourceRate: 22_050);
        Assert.Equal(8_000, mono.Length);
        float[] full = separator.ToMono([a, a], 16_000);
        Assert.Equal(16_000, full.Length);
        // A tone well below 8 kHz keeps its energy through the resampler.
        double rms = Math.Sqrt(full.Skip(2000).Take(12_000).Sum(v => (double)v * v) / 12_000);
        double inRms = Math.Sqrt(a.Skip(2000).Take(12_000).Sum(v => (double)v * v) / 12_000);
        Assert.InRange(rms / inRms, 0.8, 1.1);
    }

    [Fact]
    public void EmptyOrMismatchedInputIsRefused()
    {
        MdxSeparator separator = new(Small(), new IdentityModel());
        Assert.Throws<ArgumentException>(() => separator.Separate([], []));
        Assert.Throws<ArgumentException>(() => separator.Separate(new float[10], new float[11]));
    }
}
