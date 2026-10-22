using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Fibo 2.0 text conditioning: Qwen3-VL hidden-state taps become one <c>[B, S, bundles · 3 · hidden]</c> tensor.
/// Every tap is normalised per token with a weightless RMS norm, and each bundle concatenates its three taps in the order
/// listed. The denoiser splits the last dimension: bundle 0 feeds the Perceiver, bundles 1..5 the injection blocks.</summary>
public static unsafe class Fibo2TextBundles
{
    /// <summary>Weightless RMS-norm epsilon, <c>x · rsqrt(mean(x²) + eps)</c>.</summary>
    public const float NormEps = 1e-6f;

    /// <summary>Qwen3-VL hidden-state indices (HF <c>hidden_states</c>, index k is the input of block k), ascending. The
    /// packed taps from <see cref="Models.TextEncoders.LlamaStyleEncoder.EncodeMultiLayer"/> follow this order.</summary>
    public static IReadOnlyList<int> TapLayers { get; } = [5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35];

    /// <summary>Tap layers of each bundle, in concatenation order. Bundles 0 and 3 are the same taps by design.</summary>
    public static IReadOnlyList<IReadOnlyList<int>> BundleTaps { get; } =
    [
        [9, 20, 31],
        [5, 16, 27],
        [7, 18, 29],
        [9, 20, 31],
        [11, 22, 33],
        [13, 24, 35],
    ];

    /// <summary>Injection layers the production bundle layout feeds: every bundle after the Perceiver's.</summary>
    public static int InjectionLayerCount => BundleTaps.Count - 1;

    /// <summary>Each bundle's taps as slots in the packed input (the position of its layer in <see cref="TapLayers"/>).</summary>
    private static readonly int[][] BundleSlots = BuildBundleSlots();

    /// <summary>Builds the bundles for the Fibo 2.0 tap layout. <paramref name="taps"/> is F32 <c>[B, S, TapLayers.Count · hidden]</c>,
    /// tap-major, as <c>LlamaStyleEncoder.EncodeMultiLayer</c> returns it. The result is F32 <c>[B, S, 6 · 3 · hidden]</c>.</summary>
    public static Tensor Build(Tensor taps, int hidden) =>
        Build(taps, hidden, TapLayers.Count, BundleSlots);

    /// <summary>Core of <see cref="Build(Tensor, int)"/> over explicit packed-tap slots, so the same code path can be checked on
    /// a small hand-computed case. <paramref name="bundleSlots"/> lists each bundle's slots in concatenation order; every bundle
    /// must have the same length.</summary>
    public static Tensor Build(Tensor taps, int hidden, int tapCount, IReadOnlyList<IReadOnlyList<int>> bundleSlots)
    {
        ArgumentNullException.ThrowIfNull(taps);
        ArgumentNullException.ThrowIfNull(bundleSlots);
        if (hidden <= 0 || tapCount <= 0)
            throw new ArgumentException("Hidden width and tap count must be positive.");
        bool shapeOk = taps.Shape.Rank == 3 && taps.Shape[0] >= 1 && taps.Shape[1] >= 1 && taps.Shape[2] == (long)tapCount * hidden;
        if (taps.DType != DType.F32 || !shapeOk)
            throw new ArgumentException($"Fibo 2.0 taps must be F32 [B, S, {tapCount * hidden}], got {taps.Shape} {taps.DType}.", nameof(taps));
        if (bundleSlots.Count == 0)
            throw new ArgumentException("At least one bundle is required.", nameof(bundleSlots));

        int tapsPerBundle = bundleSlots[0].Count;
        if (tapsPerBundle == 0)
            throw new ArgumentException("A bundle needs at least one tap.", nameof(bundleSlots));
        for (int b = 0; b < bundleSlots.Count; b++)
        {
            if (bundleSlots[b].Count != tapsPerBundle)
                throw new ArgumentException(
                    $"Bundle {b} has {bundleSlots[b].Count} taps; every bundle must have {tapsPerBundle}.", nameof(bundleSlots));
            foreach (int slot in bundleSlots[b])
            {
                if ((uint)slot >= (uint)tapCount)
                    throw new ArgumentOutOfRangeException(nameof(bundleSlots), slot, $"Tap slot must be in [0, {tapCount}).");
            }
        }

        int batch = (int)taps.Shape[0];
        int seq = (int)taps.Shape[1];
        int rowIn = tapCount * hidden;
        int bundleWidth = tapsPerBundle * hidden;
        int rowOut = bundleSlots.Count * bundleWidth;
        Tensor output = new(new TensorShape(batch, seq, rowOut), DType.F32);
        float* src = (float*)taps.DataPointer;
        float* dst = (float*)output.DataPointer;
        for (long row = 0; row < (long)batch * seq; row++)
        {
            float* rowSrc = src + row * rowIn;
            float* rowDst = dst + row * rowOut;
            for (int b = 0; b < bundleSlots.Count; b++)
            {
                for (int t = 0; t < tapsPerBundle; t++)
                {
                    float* tap = rowSrc + bundleSlots[b][t] * hidden;
                    float* outTap = rowDst + b * bundleWidth + t * hidden;
                    float inverseRms = InverseRms(tap, hidden);
                    for (int h = 0; h < hidden; h++)
                        outTap[h] = tap[h] * inverseRms;
                }
            }
        }
        return output;
    }

    /// <summary>The weightless RMS-norm scale <c>rsqrt(mean(x²) + eps)</c> of one token's tap.</summary>
    private static float InverseRms(float* tap, int hidden)
    {
        double sumSquares = 0;
        for (int h = 0; h < hidden; h++)
            sumSquares += (double)tap[h] * tap[h];
        return (float)(1.0 / Math.Sqrt(sumSquares / hidden + NormEps));
    }

    private static int[][] BuildBundleSlots()
    {
        int[][] slots = new int[BundleTaps.Count][];
        for (int b = 0; b < BundleTaps.Count; b++)
        {
            slots[b] = new int[BundleTaps[b].Count];
            for (int t = 0; t < BundleTaps[b].Count; t++)
                slots[b][t] = SlotOf(BundleTaps[b][t], b);
        }
        return slots;
    }

    private static int SlotOf(int layer, int bundle)
    {
        for (int slot = 0; slot < TapLayers.Count; slot++)
        {
            if (TapLayers[slot] == layer)
                return slot;
        }
        throw new InvalidOperationException($"Fibo 2.0 bundle {bundle} reads layer {layer}, which is not a tap.");
    }
}
