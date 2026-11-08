using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>The card VAE decoder with its point-count head (<c>hairdeck/cards/vae.py</c>): a latent of width 32 becomes the
/// control points of one card, each with xyz (root-relative), width and thickness, at the uniform arc-length fractions
/// <c>j / (n - 1)</c>. A card is decoded on its own <c>n</c> positions; the padded batch of the reference gives the same
/// values because padding keys are masked.</summary>
public sealed unsafe class HairDeckCardVae
{
    /// <summary>Fewest control points a decoded card has.</summary>
    public const int MinPoints = 3;

    private const int Heads = HairDeckCardVaeCheckpointConverter.Heads;
    private const float NormEps = 1e-5f;
    private const float AttentionEps = 1e-6f;

    private readonly HairDeckCardVaeCheckpointConverter.Weights _w;
    private readonly float[] _outputMu;
    private readonly float[] _outputSd;
    private readonly float[] _frequencies;
    private readonly float _attentionScale;

    /// <summary>Builds the decoder from its weights and the output standardisation (<c>stats.json</c>, five channels).</summary>
    public HairDeckCardVae(HairDeckCardVaeCheckpointConverter.Weights weights, ReadOnlySpan<float> outputMu, ReadOnlySpan<float> outputSd)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (outputMu.Length != HairDeckCardVaeCheckpointConverter.OutputChannels || outputSd.Length != HairDeckCardVaeCheckpointConverter.OutputChannels)
            throw new ArgumentException("HairDeck card VAE needs five output means and deviations.");
        _w = weights;
        _outputMu = outputMu.ToArray();
        _outputSd = outputSd.ToArray();
        _frequencies = new float[weights.Frequencies];
        for (int k = 0; k < _frequencies.Length; k++) _frequencies[k] = MathF.Pow(2f, k) * MathF.PI;
        _attentionScale = 1f / MathF.Sqrt(weights.Dim / Heads);
    }

    /// <summary>Latent width.</summary>
    public int LatentDim => _w.LatentDim;

    /// <summary>Most control points a card can have.</summary>
    public int MaxPoints => _w.MaxPoints;

    /// <summary>Point count of the card for latent <paramref name="latent"/>: <c>max(3, argmax(n_head(z)) + 1)</c>.</summary>
    public int PointCount(IBackend backend, ReadOnlySpan<float> latent)
    {
        ValidateLatent(latent);
        using Tensor z = HairDeckNumerics.FromArray(latent, 1, _w.LatentDim);
        return Count(backend, z);
    }

    /// <summary>Decodes latent <paramref name="latent"/> to <c>n</c> control points in world coordinates: the rows are
    /// (x, y, z, width, thickness) and the root <paramref name="root"/> is added to the xyz channels.</summary>
    public float[] DecodeCard(IBackend backend, ReadOnlySpan<float> latent, ReadOnlySpan<float> root)
    {
        ValidateLatent(latent);
        if (root.Length != 3) throw new ArgumentException("A card root is three values.", nameof(root));
        using Tensor z = HairDeckNumerics.FromArray(latent, 1, _w.LatentDim);
        int n = Count(backend, z);
        using Tensor cond = Conditioning(backend, z);
        float[] u = new float[n];
        for (int j = 0; j < n; j++) u[j] = MathF.Min(j / (float)Math.Max(n - 1, 1), 1f);
        using Tensor query = Fourier(u, n);
        Tensor h = HairDeckNumerics.Project(backend, query, _w.QueryIn);
        HairDeckNumerics.AddRowInPlace(h, cond);
        for (int i = 0; i < _w.Blocks.Count; i++)
        {
            Tensor next = Block(backend, _w.Blocks[i], h, cond, n);
            h.Dispose();
            h = next;
        }
        Tensor normed = HairDeckNumerics.LayerNorm(backend, h, _w.NormWeight, _w.NormBias, NormEps);
        h.Dispose();
        Tensor channels = HairDeckNumerics.Project(backend, normed, _w.Output);
        normed.Dispose();

        float[] points = new float[n * HairDeckCardVaeCheckpointConverter.OutputChannels];
        new ReadOnlySpan<float>(channels.DataPointer, points.Length).CopyTo(points);
        channels.Dispose();
        if (_w.Integrate) Integrate(points, u);
        Finish(points, root);
        return points;
    }

    private int Count(IBackend backend, Tensor z)
    {
        using Tensor hidden = HairDeckNumerics.Project(backend, z, _w.CountIn);
        using Tensor act = HairDeckNumerics.Silu(backend, hidden);
        using Tensor logits = HairDeckNumerics.Project(backend, act, _w.CountOut);
        float* p = (float*)logits.DataPointer;
        int best = 0;
        for (int i = 1; i < _w.MaxPoints; i++)
            if (p[i] > p[best]) best = i;
        return Math.Max(MinPoints, best + 1);
    }

    private Tensor Conditioning(IBackend backend, Tensor z)
    {
        using Tensor first = HairDeckNumerics.Project(backend, z, _w.FromLatentIn);
        using Tensor act = HairDeckNumerics.Silu(backend, first);
        return HairDeckNumerics.Project(backend, act, _w.FromLatentOut);
    }

    /// <summary>Fourier query features per position: the position itself, then sines and cosines at 2^k π.</summary>
    private Tensor Fourier(float[] u, int n)
    {
        int frequencies = _frequencies.Length;
        Tensor features = HairDeckNumerics.Create(n, 1 + 2 * frequencies);
        float* f = (float*)features.DataPointer;
        for (int j = 0; j < n; j++)
        {
            float* row = f + (long)j * (1 + 2 * frequencies);
            row[0] = u[j];
            for (int k = 0; k < frequencies; k++)
            {
                float a = u[j] * _frequencies[k];
                row[1 + k] = MathF.Sin(a);
                row[1 + frequencies + k] = MathF.Cos(a);
            }
        }
        return features;
    }

    private Tensor Block(IBackend backend, HairDeckCardVaeCheckpointConverter.Block b, Tensor h, Tensor cond, int n)
    {
        int dim = _w.Dim, headDim = dim / Heads;
        using Tensor act = HairDeckNumerics.Silu(backend, cond);
        using Tensor modulation = HairDeckNumerics.Project(backend, act, b.Ada);
        Tensor[] m = HairDeckNumerics.Chunks(modulation, 6, dim);
        try
        {
            Tensor attnIn = HairDeckNumerics.NormModulate(backend, h, m[1], m[0], NormEps);
            Tensor qkv = HairDeckNumerics.Project(backend, attnIn, b.Qkv);
            attnIn.Dispose();
            using Tensor qh = HairDeckNumerics.Create(1, Heads, n, headDim);
            using Tensor kh = HairDeckNumerics.Create(1, Heads, n, headDim);
            using Tensor vh = HairDeckNumerics.Create(1, Heads, n, headDim);
            HairDeckNumerics.SplitHeads(qkv, 0, qh, Heads, headDim);
            HairDeckNumerics.SplitHeads(qkv, dim, kh, Heads, headDim);
            HairDeckNumerics.SplitHeads(qkv, 2 * dim, vh, Heads, headDim);
            qkv.Dispose();
            using Tensor oh = HairDeckNumerics.Create(1, Heads, n, headDim);
            backend.ScaledDotProductAttention(oh, qh, kh, vh, null, _attentionScale);
            using Tensor merged = HairDeckNumerics.Create(n, dim);
            HairDeckNumerics.MergeHeads(oh, merged, Heads, headDim);
            Tensor attnOut = HairDeckNumerics.Project(backend, merged, b.Output);
            Tensor afterAttn = HairDeckNumerics.GatedAdd(h, attnOut, m[2]);
            attnOut.Dispose();

            Tensor mlpIn = HairDeckNumerics.NormModulate(backend, afterAttn, m[4], m[3], NormEps);
            Tensor hidden = HairDeckNumerics.Project(backend, mlpIn, b.MlpIn);
            mlpIn.Dispose();
            Tensor hiddenAct = HairDeckNumerics.GeluExact(backend, hidden);
            hidden.Dispose();
            Tensor mlpOut = HairDeckNumerics.Project(backend, hiddenAct, b.MlpOut);
            hiddenAct.Dispose();
            Tensor result = HairDeckNumerics.GatedAdd(afterAttn, mlpOut, m[5]);
            afterAttn.Dispose();
            mlpOut.Dispose();
            return result;
        }
        finally
        {
            foreach (Tensor part in m) part.Dispose();
        }
    }

    /// <summary>Trapezoid integral of the xyz channels along the card (positions from the root), in the normalised space.</summary>
    private static void Integrate(float[] points, float[] u)
    {
        int n = u.Length;
        int c = HairDeckCardVaeCheckpointConverter.OutputChannels;
        Span<double> sum = stackalloc double[3];
        Span<double> prev = stackalloc double[3];
        for (int k = 0; k < 3; k++) prev[k] = points[k];
        for (int j = 0; j < n; j++)
        {
            double du = j == 0 ? 0.0 : u[j] - u[j - 1];
            for (int k = 0; k < 3; k++)
            {
                double cur = points[(long)j * c + k];
                sum[k] += 0.5 * (cur + prev[k]) * du;
                prev[k] = cur;
                points[(long)j * c + k] = (float)sum[k];
            }
        }
    }

    /// <summary>Inverse standardisation, root translation, and non-negative width and thickness.</summary>
    private void Finish(float[] points, ReadOnlySpan<float> root)
    {
        int c = HairDeckCardVaeCheckpointConverter.OutputChannels;
        for (int j = 0; j < points.Length / c; j++)
        {
            for (int k = 0; k < c; k++)
            {
                float v = points[(long)j * c + k] * _outputSd[k] + _outputMu[k];
                if (k < 3) v += root[k];
                else v = MathF.Max(v, 0f);
                points[(long)j * c + k] = v;
            }
        }
    }

    private void ValidateLatent(ReadOnlySpan<float> latent)
    {
        if (latent.Length != _w.LatentDim)
            throw new ArgumentException($"HairDeck card latent must have {_w.LatentDim} values, got {latent.Length}.", nameof(latent));
    }
}
