namespace HartsyInference.Core.Backends.Attention;

/// <summary>The representative key of INT8 attention as Comfy Kitchen's kernel picks it at the pinned commit
/// (benchmarks/attention/protocol.json): one of a head's own key rows, subtracted from every key of that head before
/// rotation and quantization. Keys that share a large common component then quantize their differences instead of
/// the component. The subtracted row moves every score of a query row by the same amount, <c>q · k_a</c>, so the
/// softmax is unchanged in exact arithmetic.</summary>
public static class Int8RepresentativeKey
{
    /// <summary>Keys sampled, evenly from the first to the last valid key.</summary>
    public const int Samples = 9;

    /// <summary>How much larger the largest sampled magnitude may be after the subtraction than before it.</summary>
    public const float AbsMaxSlack = 1.125f;

    private const int Threads = 128;

    /// <summary>The key row sample <paramref name="sample"/> reads among <paramref name="validKeys"/> keys.</summary>
    public static int SampleRow(int sample, int validKeys) => sample * (validKeys - 1) / (Samples - 1);

    /// <summary>The representative key of one head's <paramref name="validKeys"/> rows of
    /// <paramref name="paddedHeadDim"/> channels, or -1 when subtracting the best candidate would not shrink the
    /// sampled keys.</summary>
    /// <remarks>
    /// <para>Nine rows are sampled. The candidate is the sample closest to the samples' mean, the lowest sample winning
    /// a tie. It is accepted when subtracting it lowers the samples' total energy and raises their largest magnitude by
    /// no more than <see cref="AbsMaxSlack"/>.</para>
    /// <para>The sums are taken in the shape the kernel takes them, as read from its source: 128 lanes, each over its
    /// own channel (and the channel 128 above it), reduced by a shuffle-down tree within each group of 32 lanes and
    /// then added across the four groups. A decision that sits within rounding of a tie has not been compared with a
    /// run of the kernel.</para>
    /// </remarks>
    public static int Detect(ReadOnlySpan<float> keys, int validKeys, int paddedHeadDim)
    {
        if (paddedHeadDim is not (64 or 128 or 256))
            throw new ArgumentOutOfRangeException(nameof(paddedHeadDim), paddedHeadDim, "The padded head dim is 64, 128 or 256.");
        if (validKeys <= 0 || keys.Length < (long)validKeys * paddedHeadDim)
            throw new ArgumentException($"{keys.Length} values do not hold {validKeys} keys of {paddedHeadDim} channels.", nameof(keys));
        Span<int> rows = stackalloc int[Samples];
        for (int s = 0; s < Samples; s++) rows[s] = SampleRow(s, validKeys);

        Span<float> lanes = stackalloc float[Threads];
        float absMax = 0f;
        lanes.Clear();
        for (int lane = 0; lane < Threads; lane++)
            for (int c = lane; c < paddedHeadDim; c += Threads)
                foreach (int row in rows)
                {
                    float x = keys[row * paddedHeadDim + c];
                    lanes[lane] += x * x;
                    absMax = MathF.Max(absMax, MathF.Abs(x));
                }
        float energy = Reduce(lanes);

        int best = 0;
        float bestDistance = float.PositiveInfinity;
        for (int a = 0; a < Samples; a++)
        {
            lanes.Clear();
            for (int lane = 0; lane < Threads; lane++)
                for (int c = lane; c < paddedHeadDim; c += Threads)
                {
                    float sum = 0f;
                    foreach (int row in rows) sum += keys[row * paddedHeadDim + c];
                    float d = Samples * keys[rows[a] * paddedHeadDim + c] - sum;
                    lanes[lane] += d * d;
                }
            float distance = Reduce(lanes);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = a;
            }
        }

        float centeredAbsMax = 0f;
        lanes.Clear();
        for (int lane = 0; lane < Threads; lane++)
            for (int c = lane; c < paddedHeadDim; c += Threads)
            {
                float anchor = keys[rows[best] * paddedHeadDim + c];
                foreach (int row in rows)
                {
                    float d = keys[row * paddedHeadDim + c] - anchor;
                    lanes[lane] += d * d;
                    centeredAbsMax = MathF.Max(centeredAbsMax, MathF.Abs(d));
                }
            }
        float centeredEnergy = Reduce(lanes);
        return centeredEnergy < energy && centeredAbsMax <= absMax * AbsMaxSlack ? rows[best] : -1;
    }

    /// <summary>The lanes' sum: a shuffle-down tree within each group of 32, then the four groups in order.</summary>
    private static float Reduce(Span<float> lanes)
    {
        float total = 0f;
        for (int group = 0; group < Threads; group += 32)
        {
            Span<float> warp = lanes.Slice(group, 32);
            for (int offset = 16; offset > 0; offset >>= 1)
                for (int i = 0; i < offset; i++) warp[i] += warp[i + offset];
            total += warp[0];
        }
        return total;
    }
}
