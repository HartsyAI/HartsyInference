using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>FLUX.3 four-axis <c>(t, row, col, l)</c> RoPE position ids. Image tokens sit at <c>l = 0</c> with
/// row/column scaled by <see cref="Flux3Config.ImageRopeScale"/>; reference image <c>k</c> repeats that grid at
/// <c>l = −1−k</c>; text token <c>i</c> sits at <c>(0, 0, 0, i)</c>. Rotation math is <see cref="FluxRope"/>.</summary>
public static unsafe class Flux3PosEmbed
{
    /// <summary>The four RoPE axes.</summary>
    public const int NumAxes = 4;

    /// <summary>Ids for one image grid at axis-l value <paramref name="index"/>: <c>[h·w, 4]</c>.</summary>
    public static Tensor ImageIds(int height, int width, float index, float scale)
    {
        Tensor ids = Allocate(height * width);
        WriteImage((float*)ids.DataPointer, height, width, index, scale);
        return ids;
    }

    /// <summary>Ids for every reference grid in order, reference <c>k</c> at <c>l = −1−k</c>: <c>[Σh·w, 4]</c>.</summary>
    public static Tensor ReferenceIds(IReadOnlyList<(int Height, int Width)> shapes, float scale)
    {
        ArgumentNullException.ThrowIfNull(shapes);
        int total = 0;
        foreach ((int h, int w) in shapes) total += h * w;
        Tensor ids = Allocate(total);
        float* p = (float*)ids.DataPointer;
        for (int k = 0; k < shapes.Count; k++)
        {
            WriteImage(p, shapes[k].Height, shapes[k].Width, -1 - k, scale);
            p += (long)shapes[k].Height * shapes[k].Width * NumAxes;
        }
        return ids;
    }

    /// <summary>Ids for <paramref name="length"/> text tokens: <c>[length, 4]</c>.</summary>
    public static Tensor TextIds(int length)
    {
        Tensor ids = Allocate(length);
        float* p = (float*)ids.DataPointer;
        for (int i = 0; i < length; i++) p[i * NumAxes + 3] = i;
        return ids;
    }

    /// <summary>Concatenates id tables along the token axis in the given order.</summary>
    public static Tensor Concat(params Tensor[] parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        long rows = 0;
        foreach (Tensor part in parts) rows += part.Shape[0];
        Tensor ids = Allocate((int)rows);
        byte* dst = (byte*)ids.DataPointer;
        foreach (Tensor part in parts)
        {
            long bytes = part.Shape[0] * NumAxes * sizeof(float);
            Buffer.MemoryCopy((void*)part.DataPointer, dst, bytes, bytes);
            dst += bytes;
        }
        return ids;
    }

    private static Tensor Allocate(int rows)
    {
        Tensor ids = new Tensor(new TensorShape(rows, NumAxes), DType.F32);
        new Span<float>((void*)ids.DataPointer, rows * NumAxes).Clear();
        return ids;
    }

    private static void WriteImage(float* p, int height, int width, float index, float scale)
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                float* id = p + (long)(row * width + col) * NumAxes;
                id[0] = 0f;
                id[1] = row * scale;
                id[2] = col * scale;
                id[3] = index;
            }
        }
    }
}
