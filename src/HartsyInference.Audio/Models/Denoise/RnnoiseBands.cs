namespace HartsyInference.Audio.Models.Denoise;

/// <summary>The band/DCT front-end shared by RNNoise's feature extraction and its gain application: triangular
/// band energies over the 481-bin spectrum, per-band pitch correlation, the DCT that turns band energies into
/// cepstral features, and the interpolation that spreads 32 band gains back over the bins.
///
/// <para>Every table here is closed-form (upstream generates them in <c>dump_rnnoise_tables.c</c> rather than
/// shipping data), so they are computed once at static init instead of embedded — there is nothing to drift out
/// of sync with a checkpoint.</para>
///
/// <para><b>Band edges are in FFT bins, not Hz</b>, and are only meaningful at 48 kHz with a 960-sample window:
/// bin <c>k</c> is <c>k * 48000/960 = 50·k</c> Hz, so the 34 edges span 0-20 kHz. Feeding a 16 kHz spectrum
/// through these edges would silently reinterpret every band, which is why the pipeline resamples rather than
/// retuning the edges.</para></summary>
public static class RnnoiseBands
{
    /// <summary>Bands the model scores, and the width of its gain output.</summary>
    public const int BandCount = 32;

    /// <summary>Non-negative-frequency bins in a 960-point transform.</summary>
    public const int FreqSize = 481;

    /// <summary>Features fed to the network: 32 cepstral + 32 pitch-correlation cepstral + 1 pitch period.</summary>
    public const int FeatureCount = 2 * BandCount + 1;

    /// <summary>Band edges in FFT bins. 34 entries — one extra on each end, because the triangular weighting
    /// spills energy into the neighbouring band and the first/last need somewhere to spill.</summary>
    private static readonly int[] BandEdges =
    [
        0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53, 60, 68, 77, 87, 98, 110, 124, 140, 157,
        176, 198, 223, 251, 282, 317, 356, 400
    ];

    /// <summary>Row-major <c>[BandCount, BandCount]</c> DCT-II basis, <c>cos((i+0.5)·j·π/32)</c> with the j=0
    /// column scaled by <c>sqrt(0.5)</c>.</summary>
    private static readonly float[] DctTable = BuildDctTable();

    /// <summary>Upstream's DCT output scale. The <c>22</c> is a fossil of the original 22-band model and does
    /// <b>not</b> track <see cref="BandCount"/>; it is part of the trained feature scaling, so it stays wrong-
    /// looking on purpose.</summary>
    private static readonly float DctScale = MathF.Sqrt(2f / 22f);

    private static float[] BuildDctTable()
    {
        float[] table = new float[BandCount * BandCount];
        for (int i = 0; i < BandCount; i++)
        {
            for (int j = 0; j < BandCount; j++)
            {
                double v = Math.Cos((i + 0.5) * j * Math.PI / BandCount);
                if (j == 0) v *= Math.Sqrt(0.5);
                table[i * BandCount + j] = (float)v;
            }
        }
        return table;
    }

    /// <summary>Builds the analysis/synthesis window: <c>sin(½π·sin²(½π(i+0.5)/N))</c> over the first
    /// <paramref name="frameSize"/> samples, mirrored across the second half. Power-complementary, so applying it
    /// on both analysis and synthesis at 50% overlap reconstructs unity without a normalization pass.</summary>
    public static float[] BuildWindow(int frameSize)
    {
        float[] window = new float[2 * frameSize];
        for (int i = 0; i < frameSize; i++)
        {
            double inner = Math.Sin(0.5 * Math.PI * (i + 0.5) / frameSize);
            float w = (float)Math.Sin(0.5 * Math.PI * inner * inner);
            window[i] = w;
            window[2 * frameSize - 1 - i] = w;
        }
        return window;
    }

    /// <summary>Triangular-weighted energy per band from a half-spectrum. Each bin's power is split between its
    /// band and the next in proportion to its position, so adjacent bands overlap rather than hard-partition.</summary>
    public static void ComputeBandEnergy(ReadOnlySpan<float> re, ReadOnlySpan<float> im, Span<float> bandEnergy)
        => Accumulate(re, im, re, im, bandEnergy);

    /// <summary>Per-band correlation between the signal spectrum and the pitch-shifted spectrum — the same
    /// triangular weighting as <see cref="ComputeBandEnergy"/>, but over the cross term rather than the power.</summary>
    public static void ComputeBandCorrelation(ReadOnlySpan<float> re, ReadOnlySpan<float> im,
        ReadOnlySpan<float> pitchRe, ReadOnlySpan<float> pitchIm, Span<float> bandCorrelation)
        => Accumulate(re, im, pitchRe, pitchIm, bandCorrelation);

    private static void Accumulate(ReadOnlySpan<float> re, ReadOnlySpan<float> im,
        ReadOnlySpan<float> pr, ReadOnlySpan<float> pi, Span<float> output)
    {
        if (output.Length < BandCount)
            throw new ArgumentException($"output must hold {BandCount} bands.", nameof(output));

        Span<float> sum = stackalloc float[BandCount + 2];
        sum.Clear();

        for (int band = 0; band < BandCount + 1; band++)
        {
            int start = BandEdges[band];
            int width = BandEdges[band + 1] - start;
            for (int j = 0; j < width; j++)
            {
                float frac = (float)j / width;
                int bin = start + j;
                float v = re[bin] * pr[bin] + im[bin] * pi[bin];
                sum[band] += (1f - frac) * v;
                sum[band + 1] += frac * v;
            }
        }

        // The outermost bands have no neighbour to receive their spill, so upstream folds it back at 2/3 weight.
        sum[1] = (sum[0] + sum[1]) * 2f / 3f;
        sum[BandCount] = (sum[BandCount] + sum[BandCount + 1]) * 2f / 3f;
        for (int band = 0; band < BandCount; band++) output[band] = sum[band + 1];
    }

    /// <summary>DCT-II over the 32 bands, with upstream's fixed output scale.</summary>
    public static void Dct(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length < BandCount || output.Length < BandCount)
            throw new ArgumentException($"DCT operates on {BandCount} bands.", nameof(input));
        for (int i = 0; i < BandCount; i++)
        {
            float sum = 0f;
            for (int j = 0; j < BandCount; j++) sum += input[j] * DctTable[j * BandCount + i];
            output[i] = sum * DctScale;
        }
    }

    /// <summary>Spreads per-band gains linearly across the bins between band centres. Bins below the first band
    /// and above the last are held flat, and bins past the final edge stay zero — the model does not score the
    /// 20-24 kHz tail, and letting it through unmodified would leak un-denoised noise.</summary>
    public static void InterpolateBandGain(ReadOnlySpan<float> bandGain, Span<float> binGain)
    {
        if (bandGain.Length < BandCount)
            throw new ArgumentException($"bandGain must hold {BandCount} bands.", nameof(bandGain));
        if (binGain.Length < FreqSize)
            throw new ArgumentException($"binGain must hold {FreqSize} bins.", nameof(binGain));

        binGain.Clear();
        for (int band = 1; band < BandCount; band++)
        {
            int start = BandEdges[band];
            int width = BandEdges[band + 1] - start;
            for (int j = 0; j < width; j++)
            {
                float frac = (float)j / width;
                binGain[start + j] = (1f - frac) * bandGain[band - 1] + frac * bandGain[band];
            }
        }
        for (int bin = 0; bin < BandEdges[1]; bin++) binGain[bin] = bandGain[0];
        for (int bin = BandEdges[BandCount]; bin < BandEdges[BandCount + 1]; bin++) binGain[bin] = bandGain[BandCount - 1];
    }
}
