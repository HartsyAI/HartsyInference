using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>The 3D rotary encoding of Paint 2.1's multiview attention (<c>compute_discrete_voxel_indice</c> and <c>RotaryEmbedding.get_3d_rotary_pos_embed</c>):
/// each latent token takes the integer voxel of the mean surface position under it, and Q and K rotate by angles proportional to that voxel.</summary>
public static class HunyuanPaintRope
{
    private const double Theta = 10000.0;

    /// <summary>Pools position maps into voxel indices per token. <paramref name="positions"/> holds <paramref name="views"/> RGB images of
    /// <paramref name="pixels"/> squared bytes (row-major, three channels); a pixel is foreground when none of its channels is 255. Returns
    /// <c>views * grid * grid</c> triples ordered view, row, column. A cell with fewer than <c>(pixels / grid)^2 / 16</c> foreground pixels maps to voxel 0.</summary>
    public static int[] VoxelIndices(ReadOnlySpan<byte> positions, int views, int pixels, int grid, int voxelResolution)
    {
        if (positions.Length != views * pixels * pixels * 3)
            throw new ArgumentException("The position maps do not match views and size.", nameof(positions));
        if (grid <= 0 || pixels % grid != 0)
            throw new ArgumentException("The pixel size must be a multiple of the grid.", nameof(grid));
        int cell = pixels / grid, threshold = cell * cell / 16;
        int[] indices = new int[views * grid * grid * 3];
        for (int v = 0; v < views; v++)
            for (int gy = 0; gy < grid; gy++)
                for (int gx = 0; gx < grid; gx++)
                {
                    float sx = 0, sy = 0, sz = 0;
                    int count = 0;
                    for (int y = gy * cell; y < (gy + 1) * cell; y++)
                        for (int x = gx * cell; x < (gx + 1) * cell; x++)
                        {
                            int p = ((v * pixels + y) * pixels + x) * 3;
                            if (positions[p] == 255 || positions[p + 1] == 255 || positions[p + 2] == 255)
                                continue;
                            sx += (float)(Half)(positions[p] / 255f);
                            sy += (float)(Half)(positions[p + 1] / 255f);
                            sz += (float)(Half)(positions[p + 2] / 255f);
                            count++;
                        }
                    int token = ((v * grid + gy) * grid + gx) * 3;
                    if (count < threshold || count == 0)
                        continue;
                    indices[token] = Voxel(sx, count, voxelResolution);
                    indices[token + 1] = Voxel(sy, count, voxelResolution);
                    indices[token + 2] = Voxel(sz, count, voxelResolution);
                }
        return indices;
    }

    /// <summary>Cosine and sine rows <c>[tokens, headDim]</c> with every angle repeated for its interleaved pair: x and y take <c>headDim / 8 * 3</c>
    /// channels each and z <c>headDim / 8 * 2</c>, with frequencies <c>10000^(-2k / axisDim)</c> times the voxel index.</summary>
    public static (float[] Cos, float[] Sin) Tables(ReadOnlySpan<int> indices, int headDim)
    {
        if (indices.Length % 3 != 0)
            throw new ArgumentException("Indices must be voxel triples.", nameof(indices));
        int dimXy = headDim / 8 * 3, dimZ = headDim / 8 * 2;
        if (dimXy * 2 + dimZ != headDim || dimXy % 2 != 0 || dimZ % 2 != 0)
            throw new ArgumentException($"A head dimension of {headDim} does not split into three even axes.", nameof(headDim));
        int tokens = indices.Length / 3;
        float[] cos = new float[tokens * headDim], sin = new float[tokens * headDim];
        for (int t = 0; t < tokens; t++)
        {
            int column = 0;
            for (int axis = 0; axis < 3; axis++)
            {
                int dim = axis == 2 ? dimZ : dimXy;
                for (int k = 0; k < dim / 2; k++)
                {
                    double angle = indices[t * 3 + axis] * (1.0 / Math.Pow(Theta, 2.0 * k / dim));
                    float c = (float)Math.Cos(angle), s = (float)Math.Sin(angle);
                    cos[t * headDim + column] = cos[t * headDim + column + 1] = c;
                    sin[t * headDim + column] = sin[t * headDim + column + 1] = s;
                    column += 2;
                }
            }
        }
        return (cos, sin);
    }

    /// <summary>Uploads <see cref="Tables"/> as F32 tensors.</summary>
    public static (Tensor Cos, Tensor Sin) TableTensors(ReadOnlySpan<int> indices, int headDim)
    {
        (float[] cos, float[] sin) = Tables(indices, headDim);
        int tokens = indices.Length / 3;
        Tensor c = new(new TensorShape(tokens, headDim), DType.F32), s = new(new TensorShape(tokens, headDim), DType.F32);
        unsafe
        {
            cos.AsSpan().CopyTo(new Span<float>((float*)c.DataPointer, cos.Length));
            sin.AsSpan().CopyTo(new Span<float>((float*)s.DataPointer, sin.Length));
        }
        return (c, s);
    }

    // Upstream pools in half precision (`position.half()`): the values, the sum, the mean and the scaled index are each rounded to F16 (the sum accumulates
    // in F32 and is rounded once), then torch.round rounds half to even. Doing the same keeps every index equal to upstream's; in F32 about 4 % of the
    // finest-level indices land one voxel away.
    private static int Voxel(float sum, int count, int voxelResolution)
    {
        Half total = (Half)sum;
        Half mean = (Half)((float)total / count);
        float clamped = Math.Clamp((float)mean, 0f, 1f);
        Half scaled = (Half)(clamped * (voxelResolution - 1));
        return (int)Math.Round((float)scaled, MidpointRounding.ToEven);
    }
}
