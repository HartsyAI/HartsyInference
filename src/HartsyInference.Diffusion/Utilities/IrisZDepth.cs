namespace HartsyInference.Diffusion.Utilities;

/// <summary>Host-side depth math for Iris-Z (speridlabs/iris-z-sdk <c>iris_z/depth.py</c> and the ensemble helpers of
/// <c>iris_z/rgbd.py</c>). The model reads and writes depth as an image in <c>[-1, 1]</c>: relative log depth whose 2nd to
/// 98th percentile spans <c>[-1, skyCeiling]</c>, with sky at <c>+1</c>. A log depth plane is row-major, near = low.</summary>
public static class IrisZDepth
{
    /// <summary>A log depth at or above this value is sky.</summary>
    public const float SkyValue = 1f - 1e-6f;

    /// <summary>Where the colour scale stops along reversed turbo; the rest is near-black indigo that reads as sky.</summary>
    private const double FarColourEnd = 0.85;

    /// <summary>The decoded value above which a pixel is sky, for a build whose real depth stops at
    /// <paramref name="skyCeiling"/>.</summary>
    public static float SkyThreshold(float skyCeiling) => (skyCeiling + 1f) / 2f;

    /// <summary>Maps decoded values from <c>[-1, skyCeiling]</c> back to log depth in <c>[-1, 1]</c>, in place; sky becomes +1.</summary>
    public static void UnsqueezeFromCeiling(Span<float> encoded, float skyCeiling)
    {
        float threshold = SkyThreshold(skyCeiling);
        for (int i = 0; i < encoded.Length; i++)
        {
            float value = Math.Clamp(encoded[i], -1f, 1f);
            if (skyCeiling >= 1f)
            {
                encoded[i] = value;
                continue;
            }
            float real = Math.Clamp(-1f + (value + 1f) * 2f / (skyCeiling + 1f), -1f, 1f);
            encoded[i] = value > threshold ? 1f : real;
        }
    }

    /// <summary>The sky mask of a log depth plane: every pixel at <see cref="SkyValue"/> or above when the build has a
    /// sky ceiling below 1, none otherwise.</summary>
    public static bool[] SkyMask(ReadOnlySpan<float> logDepth, float skyCeiling)
    {
        bool[] mask = new bool[logDepth.Length];
        if (skyCeiling >= 1f)
            return mask;
        for (int i = 0; i < logDepth.Length; i++) mask[i] = logDepth[i] >= SkyValue;
        return mask;
    }

    /// <summary>Encodes a log depth plane as the image the model reads: the 2nd to 98th percentile of the non-sky pixels
    /// spans <c>[-1, skyCeiling]</c> and sky is +1. <paramref name="skyMask"/> may be empty for no sky.</summary>
    public static float[] EncodeLogDepth(ReadOnlySpan<float> logDepth, ReadOnlySpan<bool> skyMask, float skyCeiling)
    {
        if (logDepth.IsEmpty)
            throw new ArgumentException("The log depth plane is empty.", nameof(logDepth));
        if (!skyMask.IsEmpty && skyMask.Length != logDepth.Length)
            throw new ArgumentException($"A {skyMask.Length}-pixel sky mask for {logDepth.Length} depth pixels.", nameof(skyMask));
        (double low, double high) = Percentiles(logDepth, skyMask, 2.0, 98.0);
        double span = Math.Max(high - low, 1e-6);
        float[] encoded = new float[logDepth.Length];
        for (int i = 0; i < logDepth.Length; i++)
        {
            if (!skyMask.IsEmpty && skyMask[i])
            {
                encoded[i] = 1f;
                continue;
            }
            double unit = Math.Clamp(2.0 * (logDepth[i] - low) / span - 1.0, -1.0, 1.0);
            encoded[i] = (float)(-1.0 + (unit + 1.0) * (skyCeiling + 1.0) / 2.0);
        }
        return encoded;
    }

    /// <summary>Reads a grey depth map (values in <c>[0, 1]</c>) as log depth: bright is far, or near when
    /// <paramref name="disparity"/> is set.</summary>
    public static float[] LogDepthFromGray(ReadOnlySpan<float> gray, bool disparity)
    {
        float[] logDepth = new float[gray.Length];
        for (int i = 0; i < gray.Length; i++)
        {
            float log = MathF.Log(MathF.Max(gray[i], 1e-3f));
            logDepth[i] = disparity ? -log : log;
        }
        return logDepth;
    }

    /// <summary>Least-squares <c>scale·member + shift</c> onto <paramref name="reference"/>, over every pixel.</summary>
    public static float[] AlignAffine(ReadOnlySpan<float> member, ReadOnlySpan<float> reference)
    {
        if (member.Length != reference.Length || member.IsEmpty)
            throw new ArgumentException($"Cannot align {member.Length} pixels onto {reference.Length}.", nameof(member));
        double meanMember = 0, meanReference = 0;
        for (int i = 0; i < member.Length; i++)
        {
            meanMember += member[i];
            meanReference += reference[i];
        }
        meanMember /= member.Length;
        meanReference /= member.Length;
        double variance = 0, covariance = 0;
        for (int i = 0; i < member.Length; i++)
        {
            double centred = member[i] - meanMember;
            variance += centred * centred;
            covariance += centred * (reference[i] - meanReference);
        }
        double scale, shift;
        if (variance > 1e-12 * member.Length)
        {
            scale = covariance / variance;
            shift = meanReference - scale * meanMember;
        }
        else
        {
            // A constant member leaves the fit rank-deficient; this is the minimum-norm solution lstsq returns.
            scale = meanMember * meanReference / (meanMember * meanMember + 1.0);
            shift = meanReference / (meanMember * meanMember + 1.0);
        }
        float[] aligned = new float[member.Length];
        for (int i = 0; i < member.Length; i++) aligned[i] = (float)(scale * member[i] + shift);
        return aligned;
    }

    /// <summary>Combines several log depth estimates of one image: each is aligned onto
    /// <paramref name="members"/>[<paramref name="referenceIndex"/>], the per-pixel median is taken, and pixels most
    /// estimates call sky stay at +1.</summary>
    public static float[] Ensemble(IReadOnlyList<float[]> members, int referenceIndex)
    {
        ArgumentNullException.ThrowIfNull(members);
        if (members.Count == 0)
            throw new ArgumentException("An ensemble needs at least one estimate.", nameof(members));
        if ((uint)referenceIndex >= (uint)members.Count)
            throw new ArgumentOutOfRangeException(nameof(referenceIndex));
        if (members.Count == 1)
            return (float[])members[0].Clone();
        float[] reference = members[referenceIndex];
        float[][] aligned = new float[members.Count][];
        for (int m = 0; m < members.Count; m++)
            aligned[m] = m == referenceIndex ? reference : AlignAffine(members[m], reference);
        float[] result = new float[reference.Length];
        Span<float> column = members.Count <= 64 ? stackalloc float[members.Count] : new float[members.Count];
        for (int i = 0; i < result.Length; i++)
        {
            int skyVotes = 0;
            for (int m = 0; m < aligned.Length; m++)
            {
                column[m] = aligned[m][i];
                if (members[m][i] >= SkyValue) skyVotes++;
            }
            result[i] = 2 * skyVotes > members.Count ? 1f : Median(column);
        }
        return result;
    }

    /// <summary>Renders a log depth plane as RGB24: near is red, far is blue, sky is black, and the 1st to 99.5th
    /// percentile of the non-sky depth spans the scale. <paramref name="skyMask"/> may be empty for no sky.</summary>
    /// <remarks>The scale is the published polynomial fit of the turbo colour map, not matplotlib's table.</remarks>
    public static byte[] Colourize(ReadOnlySpan<float> logDepth, ReadOnlySpan<bool> skyMask)
    {
        if (logDepth.IsEmpty)
            throw new ArgumentException("The log depth plane is empty.", nameof(logDepth));
        (double low, double high) = Percentiles(logDepth, skyMask, 1.0, 99.5);
        double span = Math.Max(high - low, 1e-6);
        byte[] rgb = new byte[logDepth.Length * 3];
        for (int i = 0; i < logDepth.Length; i++)
        {
            if (!skyMask.IsEmpty && skyMask[i])
                continue;
            double position = Math.Clamp((logDepth[i] - low) / span, 0.0, 1.0) * FarColourEnd;
            Turbo(1.0 - position, out rgb[i * 3], out rgb[i * 3 + 1], out rgb[i * 3 + 2]);
        }
        return rgb;
    }

    /// <summary>Renders a log depth plane as grey RGB24 over its own range: bright is far, or near when
    /// <paramref name="disparity"/> is set. Sky is the far end.</summary>
    public static byte[] ToGray(ReadOnlySpan<float> logDepth, bool disparity)
    {
        if (logDepth.IsEmpty)
            throw new ArgumentException("The log depth plane is empty.", nameof(logDepth));
        float low = float.MaxValue, high = float.MinValue;
        foreach (float value in logDepth)
        {
            low = MathF.Min(low, value);
            high = MathF.Max(high, value);
        }
        float span = MathF.Max(high - low, 1e-6f);
        byte[] rgb = new byte[logDepth.Length * 3];
        for (int i = 0; i < logDepth.Length; i++)
        {
            float unit = (logDepth[i] - low) / span;
            byte level = (byte)MathF.Round(Math.Clamp(disparity ? 1f - unit : unit, 0f, 1f) * 255f);
            rgb[i * 3] = level;
            rgb[i * 3 + 1] = level;
            rgb[i * 3 + 2] = level;
        }
        return rgb;
    }

    /// <summary>The working size for a source image: about <paramref name="targetSide"/>² pixels, the closest aspect
    /// ratio, sides that are multiples of <paramref name="multiple"/>, and never larger than the source.</summary>
    public static (int Width, int Height) ChooseResolution(int sourceWidth, int sourceHeight, int targetSide, int multiple = 16,
        double areaTolerance = 0.2)
    {
        if (sourceWidth < multiple || sourceHeight < multiple)
            throw new ArgumentException($"A {sourceWidth}x{sourceHeight} image is smaller than {multiple} pixels on a side.");
        double sourceAspect = (double)sourceWidth / sourceHeight;
        long targetArea = Math.Min((long)targetSide * targetSide, (long)sourceWidth * sourceHeight);
        bool found = false;
        double bestAspect = 0;
        long bestAreaError = 0;
        (int Width, int Height) best = default;
        for (int height = multiple; height <= sourceHeight; height += multiple)
        {
            for (int width = multiple; width <= sourceWidth; width += multiple)
            {
                long areaError = Math.Abs((long)width * height - targetArea);
                if (areaError > areaTolerance * targetArea)
                    continue;
                double aspectError = Math.Round(Math.Abs(Math.Log((double)width / height / sourceAspect)), 6);
                if (!found || aspectError < bestAspect || (aspectError == bestAspect && areaError < bestAreaError))
                {
                    found = true;
                    bestAspect = aspectError;
                    bestAreaError = areaError;
                    best = (width, height);
                }
            }
        }
        if (!found)
            throw new ArgumentException($"No size near {targetSide}² fits {sourceWidth}x{sourceHeight}.");
        return best;
    }

    /// <summary><see cref="ChooseResolution"/> for a source that may be smaller than the target: it is enlarged first.</summary>
    public static (int Width, int Height) EnlargedSize(int sourceWidth, int sourceHeight, int targetSide)
    {
        long area = (long)sourceWidth * sourceHeight;
        if (area < (long)targetSide * targetSide)
        {
            double factor = Math.Sqrt((double)targetSide * targetSide / area);
            sourceWidth = (int)Math.Round(sourceWidth * factor, MidpointRounding.ToEven);
            sourceHeight = (int)Math.Round(sourceHeight * factor, MidpointRounding.ToEven);
        }
        return ChooseResolution(sourceWidth, sourceHeight, targetSide);
    }

    /// <summary>Resizes a log depth plane with its sky: depth bilinear, the sky mask nearest (as the reference resizes
    /// them). With a sky ceiling below 1 the returned plane names exactly the returned mask: sky pixels are +1 and every
    /// other pixel stays below <see cref="SkyValue"/>, so the blend along a sky edge cannot read as more sky.</summary>
    public static (float[] LogDepth, bool[] Sky) ResizeLogDepth(ReadOnlySpan<float> logDepth, int width, int height, int newWidth,
        int newHeight, float skyCeiling)
    {
        bool[] sourceSky = SkyMask(logDepth, skyCeiling);
        bool[] sky = sourceSky;
        if (newWidth != width || newHeight != height)
        {
            sky = new bool[newWidth * newHeight];
            for (int y = 0; y < newHeight; y++)
            {
                int sourceY = Math.Min((int)((y + 0.5) * height / newHeight), height - 1);
                for (int x = 0; x < newWidth; x++)
                    sky[y * newWidth + x] = sourceSky[sourceY * width + Math.Min((int)((x + 0.5) * width / newWidth), width - 1)];
            }
        }
        float[] resized = ResizeBilinear(logDepth, width, height, newWidth, newHeight);
        if (skyCeiling < 1f)
        {
            float belowSky = MathF.BitDecrement(SkyValue);
            for (int i = 0; i < resized.Length; i++) resized[i] = sky[i] ? 1f : MathF.Min(resized[i], belowSky);
        }
        return (resized, sky);
    }

    /// <summary>Bilinear resize of a row-major plane, sampling at pixel centres.</summary>
    public static float[] ResizeBilinear(ReadOnlySpan<float> plane, int width, int height, int newWidth, int newHeight)
    {
        if (plane.Length != width * height)
            throw new ArgumentException($"A {plane.Length}-pixel plane is not {width}x{height}.", nameof(plane));
        if (newWidth == width && newHeight == height)
            return plane.ToArray();
        float[] resized = new float[newWidth * newHeight];
        double scaleX = (double)width / newWidth, scaleY = (double)height / newHeight;
        for (int y = 0; y < newHeight; y++)
        {
            double sourceY = Math.Clamp((y + 0.5) * scaleY - 0.5, 0, height - 1);
            int y0 = (int)sourceY;
            int y1 = Math.Min(y0 + 1, height - 1);
            double fy = sourceY - y0;
            for (int x = 0; x < newWidth; x++)
            {
                double sourceX = Math.Clamp((x + 0.5) * scaleX - 0.5, 0, width - 1);
                int x0 = (int)sourceX;
                int x1 = Math.Min(x0 + 1, width - 1);
                double fx = sourceX - x0;
                double top = plane[y0 * width + x0] * (1 - fx) + plane[y0 * width + x1] * fx;
                double bottom = plane[y1 * width + x0] * (1 - fx) + plane[y1 * width + x1] * fx;
                resized[y * newWidth + x] = (float)(top * (1 - fy) + bottom * fy);
            }
        }
        return resized;
    }

    /// <summary>The <paramref name="lowPercent"/> and <paramref name="highPercent"/> percentiles (linear interpolation)
    /// of the non-sky pixels, or of every pixel when all are sky.</summary>
    private static (double Low, double High) Percentiles(ReadOnlySpan<float> values, ReadOnlySpan<bool> skyMask, double lowPercent,
        double highPercent)
    {
        float[] real = new float[values.Length];
        int count = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (skyMask.IsEmpty || !skyMask[i]) real[count++] = values[i];
        }
        if (count == 0)
        {
            values.CopyTo(real);
            count = values.Length;
        }
        Array.Sort(real, 0, count);
        return (Percentile(real.AsSpan(0, count), lowPercent), Percentile(real.AsSpan(0, count), highPercent));
    }

    private static double Percentile(ReadOnlySpan<float> sorted, double percent)
    {
        double position = percent / 100.0 * (sorted.Length - 1);
        int below = (int)Math.Floor(position);
        int above = Math.Min(below + 1, sorted.Length - 1);
        return sorted[below] + (sorted[above] - (double)sorted[below]) * (position - below);
    }

    private static float Median(Span<float> values)
    {
        values.Sort();
        int middle = values.Length / 2;
        return values.Length % 2 == 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2f;
    }

    private static void Turbo(double x, out byte r, out byte g, out byte b)
    {
        double red = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
        double green = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
        double blue = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
        r = (byte)Math.Round(Math.Clamp(red, 0.0, 1.0) * 255.0);
        g = (byte)Math.Round(Math.Clamp(green, 0.0, 1.0) * 255.0);
        b = (byte)Math.Round(Math.Clamp(blue, 0.0, 1.0) * 255.0);
    }
}
