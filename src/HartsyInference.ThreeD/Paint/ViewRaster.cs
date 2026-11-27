namespace HartsyInference.ThreeD.Paint;

/// <summary>Result of <see cref="ViewRasterizer"/>: per pixel the covering triangle plus one (0 is background) and its perspective-correct
/// barycentric weights. Row 0 is NDC y = -1, as upstream's <c>custom_rasterizer</c> lays its images out.</summary>
public sealed class ViewRaster
{
    /// <summary>Creates an empty raster.</summary>
    public ViewRaster(int width, int height)
    {
        Width = width;
        Height = height;
        FaceIds = new int[checked(width * height)];
        Barycentric = new float[checked(width * height * 3)];
    }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>Triangle index plus one per pixel, or 0.</summary>
    public int[] FaceIds { get; }

    /// <summary>Weights of the triangle's three vertices, three per pixel; zero on background.</summary>
    public float[] Barycentric { get; }

    /// <summary>Interpolates a per-vertex attribute (<paramref name="channels"/> values per vertex, indexed like positions) at every pixel;
    /// background pixels get zeros.</summary>
    public float[] Interpolate(ReadOnlySpan<float> vertexData, ReadOnlySpan<int> indices, int channels)
    {
        float[] output = new float[Width * Height * channels];
        for (int p = 0; p < FaceIds.Length; p++)
        {
            int face = FaceIds[p] - 1;
            if (face < 0)
                continue;
            for (int c = 0; c < channels; c++)
            {
                float value = 0;
                for (int k = 0; k < 3; k++)
                    value += Barycentric[p * 3 + k] * vertexData[indices[face * 3 + k] * channels + c];
                output[p * channels + c] = value;
            }
        }
        return output;
    }
}
