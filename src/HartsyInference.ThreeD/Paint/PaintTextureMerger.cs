namespace HartsyInference.ThreeD.Paint;

/// <summary>Merges per-view bakes into one texture (<c>MeshRender.fast_bake_texture</c>): each view weighs <c>viewWeight * cos^exponent</c> per texel, a
/// view whose trusted texels are almost all painted already is skipped, and the texture is the weighted mean.</summary>
public sealed class PaintTextureMerger
{
    private readonly PaintTexelSurface _surface;
    private readonly PaintBakeOptions _options;
    private readonly float[] _sum;
    private readonly float[] _trust;

    /// <summary>Creates a merger for <paramref name="channels"/>-channel textures over <paramref name="surface"/>.</summary>
    public PaintTextureMerger(PaintTexelSurface surface, int channels, PaintBakeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(surface);
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels));
        _surface = surface;
        Channels = channels;
        _options = options ?? new PaintBakeOptions();
        _sum = new float[surface.Count * channels];
        _trust = new float[surface.Count];
    }

    /// <summary>Values per texel.</summary>
    public int Channels { get; }

    /// <summary>Adds a view; returns false when it was skipped as redundant.</summary>
    public bool Add(PaintViewBake view, double viewWeight)
    {
        ArgumentNullException.ThrowIfNull(view);
        if (view.Weights.Length != _trust.Length || view.Channels != Channels)
            throw new ArgumentException("The view bake does not match the surface or channel count.", nameof(view));
        int seen = 0, painted = 0;
        float[] weight = new float[_trust.Length];
        for (int i = 0; i < weight.Length; i++)
        {
            weight[i] = (float)(viewWeight * Math.Pow(view.Weights[i], _options.CosineExponent));
            if (weight[i] > 0f)
            {
                seen++;
                if (_trust[i] > 0f)
                    painted++;
            }
        }
        if (seen == 0 || (double)painted / seen > _options.SkipOverlap)
            return false;
        for (int i = 0; i < weight.Length; i++)
        {
            for (int c = 0; c < Channels; c++)
                _sum[i * Channels + c] += view.Colors[i * Channels + c] * weight[i];
            _trust[i] += weight[i];
        }
        return true;
    }

    /// <summary>The merged texture (row-major, row 0 first) and a mask of the texels any view painted.</summary>
    public (float[] Texture, bool[] Painted) Finish()
    {
        int texels = _surface.Width * _surface.Height;
        float[] texture = new float[texels * Channels];
        bool[] painted = new bool[texels];
        for (int i = 0; i < _trust.Length; i++)
        {
            int texel = _surface.Texels[i];
            float trust = MathF.Max(_trust[i], 1e-8f);
            for (int c = 0; c < Channels; c++)
                texture[texel * Channels + c] = _sum[i * Channels + c] / trust;
            painted[texel] = _trust[i] > 1e-8f;
        }
        return (texture, painted);
    }
}
