using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.Idol.Budget;

/// <summary>The octree behind Budget3D (upstream <c>budget3d/octree.py</c>). Idol's 8192 latent tokens are a stratified 3D Sobol set whose occupancy is exactly 16 × 16 × 16 × 2: every 16³ voxel holds two
/// antipodal points, one per x-half. That makes the set a relabelling of a three-level octree with three cubic scales: one Sobol point (s = 1, 8192 tokens), one 16³ cell of two points (s = 2, 4096 tokens), and one
/// 8³ cube of eight cells and sixteen points (s = 4, 512 tokens). Everything here is host index arithmetic, done once per model load; the cell order is <c>(ix · 16 + iy) · 16 + iz</c> and the cube order
/// <c>(jx · 8 + jy) · 8 + jz</c>.</summary>
public static class BudgetOctree
{
    /// <summary>Sobol points, Idol's latent tokens.</summary>
    public const int Points = 8192;

    /// <summary>16³ cells, each holding two points.</summary>
    public const int Cells = 4096;

    /// <summary>8³ cubes, each holding eight cells.</summary>
    public const int Cubes = 512;

    /// <summary>The row of <c>bp_scale_emb</c> of each scale: s = 1, 2 and 4 use rows 0, 1 and 2.</summary>
    public static int ScaleRow(int scale) => scale switch
    {
        1 => 0,
        2 => 1,
        4 => 2,
        _ => throw new ArgumentOutOfRangeException(nameof(scale), scale, "Budget3D has the scales 1, 2 and 4."),
    };

    /// <summary>The Sobol/octree bijection of <paramref name="positions"/> (<c>8192 × 3</c> floats in [0, 1]): <c>ToOctree[k]</c> is the Sobol point on the octree slot <c>k = cell · 2 + half</c>, and
    /// <c>FromOctree[p]</c> is the slot of point <c>p</c>. Throws unless the 16³ × 2 occupancy holds exactly, because every count of the scale hierarchy depends on it.</summary>
    /// <exception cref="HartsyInferenceException">The positions are not an exact 16³ × 2 stratified set.</exception>
    public static (int[] ToOctree, int[] FromOctree) Permutation(ReadOnlySpan<float> positions)
    {
        if (positions.Length != Points * 3)
            throw new ArgumentException($"The token positions are {Points} × 3 floats; got {positions.Length} values.", nameof(positions));
        int[] toOctree = new int[Points], fromOctree = new int[Points];
        Array.Fill(toOctree, -1);
        for (int p = 0; p < Points; p++)
        {
            int ix = Math.Clamp((int)(positions[p * 3] * 16f), 0, 15), iy = Math.Clamp((int)(positions[p * 3 + 1] * 16f), 0, 15), iz = Math.Clamp((int)(positions[p * 3 + 2] * 16f), 0, 15);
            int half = (int)(positions[p * 3] * 32f) % 2;
            int key = (((ix * 16 + iy) * 16 + iz) * 2) + half;
            if (toOctree[key] >= 0)
                throw new HartsyInferenceException("The token positions are not an exact 16³ × 2 stratified Sobol set; the Budget3D scale hierarchy does not apply to this checkpoint.");
            toOctree[key] = p;
            fromOctree[p] = key;
        }

        return (toOctree, fromOctree);
    }

    /// <summary>The eight cells of every cube, <c>[512 × 8]</c>, in the order <c>(a, b, c)</c> over the x, y and z halves, which is the order of <see cref="CubeChildOffsets"/> and the refiner's slots.</summary>
    public static int[] CubeToCells()
    {
        int[] cells = new int[Cubes * 8];
        for (int j = 0; j < Cubes; j++)
        {
            int jx = j / 64, jy = j / 8 % 8, jz = j % 8, o = 0;
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++)
                        cells[j * 8 + o++] = ((2 * jx + a) * 16 + (2 * jy + b)) * 16 + (2 * jz + c);
        }

        return cells;
    }

    /// <summary>The centre of every 16³ cell in [0, 1]³, <c>[4096 × 3]</c>, in cell order.</summary>
    public static float[] CellCenters() => Centers(16);

    /// <summary>The centre of every 8³ cube in [0, 1]³, <c>[512 × 3]</c>, in cube order.</summary>
    public static float[] CubeCenters() => Centers(8);

    private static float[] Centers(int side)
    {
        float[] c = new float[side * side * side * 3];
        int n = 0;
        for (int x = 0; x < side; x++)
            for (int y = 0; y < side; y++)
                for (int z = 0; z < side; z++)
                {
                    c[n++] = (x + 0.5f) / side;
                    c[n++] = (y + 0.5f) / side;
                    c[n++] = (z + 0.5f) / side;
                }

        return c;
    }

    /// <summary>The centre of each child cell relative to its cube's centre, the cube being [−1, 1]³: <c>[8 × 3]</c>, the offsets <c>(a, b, c) − ½</c>.</summary>
    public static float[] CubeChildOffsets()
    {
        float[] o = new float[24];
        int n = 0;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
                for (int c = 0; c < 2; c++)
                {
                    o[n++] = a - 0.5f;
                    o[n++] = b - 0.5f;
                    o[n++] = c - 0.5f;
                }

        return o;
    }

    /// <summary>The two Sobol slots' offsets inside a cell, the cell being [−1, 1]³, <c>[2 × 3]</c>: only the x-half is known, so y and z stay at the centre.</summary>
    public static float[] CellChildOffsets() => [-0.5f, 0f, 0f, 0.5f, 0f, 0f];

    /// <summary>The cube holding <paramref name="cell"/>.</summary>
    public static int CubeOfCell(int cell)
    {
        int ix = cell / 256, iy = cell / 16 % 16, iz = cell % 16;
        return (ix / 2 * 8 + iy / 2) * 8 + iz / 2;
    }

    /// <summary>Per-cell and per-cube salience of a latent in octree order (upstream <c>detail_scores</c>): <c>log2(1 + sqrt(mean_c Var_points(z) + 1e-8) / 0.08)</c> over the two points a cell would merge
    /// and the sixteen a cube would, so ranking by it ranks by detail.</summary>
    /// <param name="latentInOctreeOrder"><c>[8192, channels]</c>, row <c>k</c> the latent of octree slot <c>k</c> (the latent permuted by <see cref="Permutation"/>'s <c>ToOctree</c>).</param>
    /// <param name="channels">Latent channels, 16.</param>
    public static (float[] CubeScores, float[] CellScores) DetailScores(ReadOnlySpan<float> latentInOctreeOrder, int channels)
    {
        if (latentInOctreeOrder.Length != Points * channels)
            throw new ArgumentException($"A latent in octree order is {Points} × {channels} floats; got {latentInOctreeOrder.Length}.", nameof(latentInOctreeOrder));
        float[] cell = new float[Cells], cube = new float[Cubes];
        int[] cubeCells = CubeToCells();
        // Population variance per channel, averaged over channels, in float32 like the reference's tensor ops.
        for (int c = 0; c < Cells; c++)
            cell[c] = Score(MeanVariance(latentInOctreeOrder, channels, [c * 2, c * 2 + 1]));
        int[] points = new int[16];
        for (int j = 0; j < Cubes; j++)
        {
            for (int o = 0; o < 8; o++)
            {
                points[o * 2] = cubeCells[j * 8 + o] * 2;
                points[o * 2 + 1] = cubeCells[j * 8 + o] * 2 + 1;
            }

            cube[j] = Score(MeanVariance(latentInOctreeOrder, channels, points));
        }

        return (cube, cell);
    }

    private static float MeanVariance(ReadOnlySpan<float> z, int channels, int[] rows)
    {
        float total = 0f;
        for (int ch = 0; ch < channels; ch++)
        {
            float mean = 0f;
            foreach (int r in rows)
                mean += z[r * channels + ch];
            mean /= rows.Length;
            float v = 0f;
            foreach (int r in rows)
            {
                float d = z[r * channels + ch] - mean;
                v += d * d;
            }

            total += v / rows.Length;
        }

        return total / channels;
    }

    private static float Score(float variance) => MathF.Log2(1f + MathF.Sqrt(variance + 1e-8f) / 0.08f);
}
