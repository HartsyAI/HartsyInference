using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Render;

/// <summary>The supersampled colour and depth of a render, and its reduction to the output picture.</summary>
internal sealed class SampleBuffer
{
    private readonly float[] _depth;
    private readonly byte[] _rgb;
    private readonly int _samples;

    public SampleBuffer(MeshRenderOptions options)
    {
        _samples = options.Samples;
        Width = options.Width * _samples;
        Height = options.Height * _samples;
        _depth = new float[Width * Height];
        Array.Fill(_depth, float.PositiveInfinity);
        _rgb = new byte[Width * Height * 3];
    }

    /// <summary>Width in samples.</summary>
    public int Width { get; }

    /// <summary>Height in samples.</summary>
    public int Height { get; }

    /// <summary>Whether a sample at <paramref name="depth"/> would be nearer than what is there, so a caller can skip shading a hidden one.</summary>
    public bool IsNearer(int x, int y, float depth) => depth < _depth[y * Width + x];

    /// <summary>Writes a sample when it is nearer than what is there.</summary>
    public bool TryWrite(int x, int y, float depth, byte r, byte g, byte b)
    {
        int i = y * Width + x;
        if (depth >= _depth[i])
            return false;
        _depth[i] = depth;
        _rgb[i * 3] = r;
        _rgb[i * 3 + 1] = g;
        _rgb[i * 3 + 2] = b;
        return true;
    }

    /// <summary>The picture: each pixel averages its samples; without a background its alpha is the share of samples that were drawn.</summary>
    public TextureImage Resolve(MeshRenderOptions options)
    {
        int w = options.Width, h = options.Height, s = _samples;
        int channels = options.Background is null ? 4 : 3;
        byte[] pixels = new byte[w * h * channels];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int covered = 0, r = 0, g = 0, b = 0;
                for (int sy = 0; sy < s; sy++)
                {
                    for (int sx = 0; sx < s; sx++)
                    {
                        int i = (y * s + sy) * Width + x * s + sx;
                        if (float.IsPositiveInfinity(_depth[i]))
                            continue;
                        covered++;
                        r += _rgb[i * 3];
                        g += _rgb[i * 3 + 1];
                        b += _rgb[i * 3 + 2];
                    }
                }
                int o = (y * w + x) * channels;
                float share = covered / (float)(s * s);
                if (options.Background is { } bg)
                {
                    pixels[o] = Mix(bg[0], covered, r, share);
                    pixels[o + 1] = Mix(bg[1], covered, g, share);
                    pixels[o + 2] = Mix(bg[2], covered, b, share);
                }
                else if (covered > 0)
                {
                    pixels[o] = (byte)(r / covered);
                    pixels[o + 1] = (byte)(g / covered);
                    pixels[o + 2] = (byte)(b / covered);
                    pixels[o + 3] = (byte)MathF.Round(share * 255f);
                }
            }
        }
        return new TextureImage(w, h, channels, pixels);
    }

    private static byte Mix(byte background, int covered, int sum, float share) =>
        (byte)MathF.Round(background * (1f - share) + (covered == 0 ? 0 : sum / (float)covered) * share);
}
