using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>P3-SAM's per-point features (<c>get_feat</c> in <c>P3-SAM/demo/auto_mask.py</c>): the points are centred and voxelised (<see cref="P3SamGridSample"/>),
/// run through the Sonata encoder with <c>[coord, color, normal]</c> as input (the colour is <c>1 / 255</c>, upstream feeds ones through
/// <c>NormalizeColor</c>), every stage's features are gathered back to the finest voxels through the pooling maps (<c>[stage 0 | stage 1 | ... | stage 4]</c>,
/// 48 + 96 + 192 + 384 + 512 = 1232 channels), and the <c>mlp</c> (1232, 512, 512, 512) maps them to 512 channels, which are copied to every original point
/// of the voxel.</summary>
public sealed class P3SamFeatures : IDisposable
{
    private const float Color = 1f / 255f;
    private readonly IBackend _backend;
    private readonly UtoniaEncoder _sonata;
    private readonly P3SamMlp _mlp;
    private readonly int _gathered;

    /// <summary>Builds the encoder from <paramref name="state"/> (the tensors of <c>p3sam.safetensors</c>).</summary>
    public P3SamFeatures(IBackend backend, IReadOnlyDictionary<string, Tensor> state, UtoniaConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        UtoniaConfig sonata = config ?? UtoniaConfig.Sonata;
        _backend = backend;
        _sonata = new UtoniaEncoder(backend, sonata, P3SamCheckpoint.SonataWeights(state, sonata));
        _mlp = new P3SamMlp(state, "mlp");
        _gathered = sonata.EncChannels.Sum();
        if (_mlp.In != _gathered)
            throw new HartsyInferenceException($"P3-SAM mlp expects {_mlp.In} input channels, the encoder stages give {_gathered}.");
    }

    /// <summary>Feature width of <see cref="Extract"/> (512).</summary>
    public int FeatureDim => _mlp.Out;

    /// <summary>Returns <c>[count, FeatureDim]</c> features for <paramref name="points"/> (<c>[count, 3]</c>, already normalised to the unit cube) and their
    /// face <paramref name="normals"/> (<c>[count, 3]</c>).</summary>
    public float[] Extract(double[] points, float[] normals, int count, P3SamVoxelPick pick, Random? random = null) =>
        ExtractDetailed(points, normals, count, pick, random).Features;

    /// <summary>Every intermediate of <see cref="Extract"/>: the voxel selection, each stage's features after its blocks (finest voxels first), the gathered
    /// channels, and the per-point result.</summary>
    internal sealed record Detail(P3SamGridSample.Result Grid, float[][] Stages, int[][] StageGrids, float[] Gathered, float[] Features);

    internal Detail ExtractDetailed(double[] points, float[] normals, int count, P3SamVoxelPick pick, Random? random = null)
    {
        if (normals.Length != count * 3)
            throw new ArgumentException("Normals must be [count, 3].", nameof(normals));
        P3SamGridSample.Result grid = P3SamGridSample.Run(points, count, pick, random);
        int rows = grid.Selected.Length;
        Tensor feat = new(new TensorShape(rows, 9), DType.F32);
        float[] coord = new float[rows * 3];
        unsafe
        {
            float* f = (float*)feat.DataPointer;
            for (int r = 0; r < rows; r++)
            {
                int source = grid.Selected[r];
                for (int a = 0; a < 3; a++)
                {
                    coord[r * 3 + a] = (float)grid.Shifted[source * 3 + a];
                    f[r * 9 + a] = coord[r * 3 + a];
                    f[r * 9 + 3 + a] = Color;
                    f[r * 9 + 6 + a] = normals[source * 3 + a];
                }
            }
        }

        UtoniaPoint raw = new(feat, coord, grid.GridCoord, new int[rows]);
        IReadOnlyList<UtoniaPoint> levels = _sonata.ForwardLevels(raw);
        try
        {
            float[][] stages = [.. levels.Select(ToArray)];
            int[][] stageGrids = [.. levels.Select(l => l.GridCoord)];
            float[] gathered = Gather(levels, rows);
            float[] voxelFeatures = _mlp.Forward(_backend, gathered, rows);
            int dim = _mlp.Out;
            float[] result = new float[(long)count * dim];
            for (int i = 0; i < count; i++)
                Array.Copy(voxelFeatures, (long)grid.Inverse[i] * dim, result, (long)i * dim, dim);
            return new Detail(grid, stages, stageGrids, gathered, result);
        }
        finally
        {
            foreach (UtoniaPoint level in levels)
                level.Dispose();
        }
    }

    private static unsafe float[] ToArray(UtoniaPoint level)
    {
        float[] values = new float[level.Feat.ElementCount];
        new ReadOnlySpan<float>((float*)level.Feat.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    /// <summary>Sonata's unpooling by concatenation in <c>get_feat</c>: each finest voxel's row is followed through the pooling maps and every stage's feature
    /// vector is appended, finest first.</summary>
    internal static unsafe float[] Gather(IReadOnlyList<UtoniaPoint> levels, int rows)
    {
        int total = levels.Sum(l => (int)l.Feat.Shape[1]);
        float[] result = new float[(long)rows * total];
        int[] row = new int[rows];
        for (int i = 0; i < rows; i++)
            row[i] = i;
        int column = 0;
        for (int s = 0; s < levels.Count; s++)
        {
            UtoniaPoint level = levels[s];
            int width = (int)level.Feat.Shape[1];
            if (s > 0)
            {
                int[] inverse = level.PoolingInverse ?? throw new HartsyInferenceException("A pooled level carries no pooling inverse.");
                for (int i = 0; i < rows; i++)
                    row[i] = inverse[row[i]];
            }

            float* feat = (float*)level.Feat.DataPointer;
            for (int i = 0; i < rows; i++)
                new ReadOnlySpan<float>(feat + (long)row[i] * width, width).CopyTo(result.AsSpan((int)((long)i * total + column), width));
            column += width;
        }

        return result;
    }

    /// <inheritdoc/>
    public void Dispose() => _sonata.Dispose();
}
