using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>The per-card colour regression (<c>hairdeck/models/colormodel.py</c>, route <c>reg1</c>): one pass at t = 0 of the
/// image DiT's network, with seven patch statistics added to the context, predicting 16 CIELAB D65 colours per card.</summary>
public sealed unsafe class HairDeckColorHead
{
    /// <summary>Colours per card along its length.</summary>
    public const int Bins = HairDeckColorCheckpointConverter.Bins;

    /// <summary>Patch edge of the statistics, in pixels of the model input.</summary>
    public const int PatchSize = 16;

    private const int PatchChannels = 7;
    private const float NormEps = 1e-6f;
    private const float MinMatteFraction = 1e-3f;

    private readonly HairDeckColorCheckpointConverter.Weights _w;
    private readonly HairDeckImageDit _image;

    public HairDeckColorHead(HairDeckColorCheckpointConverter.Weights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _w = weights;
        _image = new HairDeckImageDit(weights.Dit);
        if (weights.ColorOut.Weight.Shape[0] != 3 * Bins)
            throw new ArgumentException("HairDeck colour head must predict 16 Lab colours per card (route reg1).", nameof(weights));
    }

    /// <summary>The colour model's set DiT.</summary>
    public HairDeckSetDit Core => _image.Core;

    /// <summary>Context of one input: the DINOv3 tokens with the patch statistics of the crop added to their patch rows.
    /// <paramref name="image"/> is <c>[3, H, W]</c> in 0..255, <paramref name="matte"/> is <c>[H, W]</c> with 0 or 1.
    /// The caller disposes the result.</summary>
    public Tensor EncodeContext(IBackend backend, Tensor dinoTokens, Tensor image, Tensor matte)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(matte);
        if (image.Shape.Rank != 3 || image.Shape[0] != 3 || matte.Shape.Rank != 2 || matte.Shape[0] != image.Shape[1] || matte.Shape[1] != image.Shape[2])
            throw new ArgumentException($"HairDeck colour input must be [3, H, W] with a [H, W] matte, got {image.Shape} and {matte.Shape}.", nameof(image));
        int h = (int)image.Shape[1], w = (int)image.Shape[2];
        if (h % PatchSize != 0 || w % PatchSize != 0)
            throw new ArgumentException($"HairDeck colour input sides must be multiples of {PatchSize}.", nameof(image));
        int patches = h / PatchSize * (w / PatchSize);
        int contextPatches = _image.ContextTokens - 5;
        if (patches != contextPatches)
            throw new ArgumentException($"HairDeck colour input has {patches} patches, the context has {contextPatches}.", nameof(image));

        Tensor context = _image.EncodeContext(backend, dinoTokens);
        using Tensor features = PatchStatistics(image, matte, h, w, patches);
        using Tensor patchRows = HairDeckNumerics.Project(backend, features, _w.Patch);
        int dim = _image.Core.Dim;
        float* c = (float*)context.DataPointer;
        float* p = (float*)patchRows.DataPointer;
        for (long i = 0; i < (long)patches * dim; i++) c[5L * dim + i] += p[i];
        return context;
    }

    /// <summary>Standardised colours <c>[N, 48]</c> of the card tokens <paramref name="cards"/> <c>[N, 40]</c> (clean and
    /// sign-ruled) given the colour context; the caller disposes the result.</summary>
    public Tensor PredictStandardized(IBackend backend, Tensor cards, Tensor context)
    {
        Tensor c = Core.Conditioning(backend, 0f, context);
        Tensor h = Core.Embed(backend, cards);
        Tensor blocks = Core.RunBlocks(backend, h, c, context);
        h.Dispose();

        using Tensor act = HairDeckNumerics.Silu(backend, c);
        c.Dispose();
        using Tensor modulation = HairDeckNumerics.Project(backend, act, _w.ColorAda);
        Tensor[] chunks = HairDeckNumerics.Chunks(modulation, 2, Core.Dim);
        using Tensor shift = chunks[0];
        using Tensor scale = chunks[1];
        using Tensor normed = HairDeckNumerics.NormModulate(backend, blocks, scale, shift, NormEps);
        blocks.Dispose();
        return HairDeckNumerics.Project(backend, normed, _w.ColorOut);
    }

    /// <summary>CIELAB colours <c>[N, 16 * 3]</c> (bin-major, L a b per bin) of the cards; the caller disposes the result.</summary>
    public Tensor PredictLab(IBackend backend, Tensor cards, Tensor context)
    {
        using Tensor standardized = PredictStandardized(backend, cards, context);
        int n = (int)standardized.Shape[0];
        Tensor lab = HairDeckNumerics.Create(n, 3 * Bins);
        float* y = (float*)standardized.DataPointer;
        float* o = (float*)lab.DataPointer;
        float* mu = (float*)_w.Mu.DataPointer;
        float* sd = (float*)_w.Sd.DataPointer;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3 * Bins; k++)
            {
                int channel = k % 3;
                long at = (long)i * 3 * Bins + k;
                o[at] = y[at] * sd[channel] + mu[channel];
            }
        return lab;
    }

    /// <summary>Per patch: the mean RGB (0..1), the matte fraction, and the mean RGB under the matte divided by that fraction.</summary>
    private static Tensor PatchStatistics(Tensor image, Tensor matte, int h, int w, int patches)
    {
        Tensor features = HairDeckNumerics.Create(patches, PatchChannels);
        float* img = (float*)image.DataPointer;
        float* m = (float*)matte.DataPointer;
        float* f = (float*)features.DataPointer;
        int gridW = w / PatchSize;
        const float area = PatchSize * PatchSize;
        Span<double> meanRgb = stackalloc double[3];
        Span<double> matteRgb = stackalloc double[3];
        for (int p = 0; p < patches; p++)
        {
            int y0 = p / gridW * PatchSize, x0 = p % gridW * PatchSize;
            double frac = 0;
            meanRgb.Clear();
            matteRgb.Clear();
            for (int dy = 0; dy < PatchSize; dy++)
                for (int dx = 0; dx < PatchSize; dx++)
                {
                    int pix = (y0 + dy) * w + x0 + dx;
                    float mv = m[pix] > 0.5f ? 1f : 0f;
                    frac += mv;
                    for (int ch = 0; ch < 3; ch++)
                    {
                        float v = img[(long)ch * h * w + pix] / 255f;
                        meanRgb[ch] += v;
                        matteRgb[ch] += v * mv;
                    }
                }
            float fraction = (float)(frac / area);
            float denominator = MathF.Max(fraction, MinMatteFraction);
            for (int ch = 0; ch < 3; ch++)
            {
                f[(long)p * PatchChannels + ch] = (float)(meanRgb[ch] / area);
                f[(long)p * PatchChannels + 3] = fraction;
                f[(long)p * PatchChannels + 4 + ch] = (float)(matteRgb[ch] / area) / denominator;
            }
        }
        return features;
    }
}
