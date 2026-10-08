namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Two-dimensional rotary tables of the vision tower for one patch grid: cos and sin, row-major <c>[Patches, HeadDim]</c>, laid out for the half-split rotation.</summary>
/// <remarks>Matches upstream <c>get_vision_cos_sin</c> run at half the head width. Patch <c>(h, w)</c>, numbered row-major, has its row angles <c>h * invFreq[j]</c> in the first
/// quarter of the head and its column angles <c>w * invFreq[j]</c> in the second. Upstream rotates elements <c>i</c> and <c>i + HeadDim / 2</c> of a head by one shared angle, so that half is repeated
/// and the table is what <see cref="HartsyInference.Core.Backends.IBackend.ApplyRopeSingle"/> reads. The angles are built in float32 like upstream.
/// This is not the interleaved one-dimensional table <see cref="DeepSeekV41RopeTable"/> holds.</remarks>
public sealed class DeepSeekV41VisionRopeTable
{
    private DeepSeekV41VisionRopeTable(int gridHeight, int gridWidth, int headDim, float[] cos, float[] sin)
    {
        GridHeight = gridHeight;
        GridWidth = gridWidth;
        HeadDim = headDim;
        Cos = cos;
        Sin = sin;
    }

    /// <summary>Patch rows of the grid.</summary>
    public int GridHeight { get; }

    /// <summary>Patch columns of the grid.</summary>
    public int GridWidth { get; }

    /// <summary>Patches in the grid.</summary>
    public int Patches => GridHeight * GridWidth;

    /// <summary>Per-head width the table spans; the rotated half is <c>HeadDim / 2</c> wide.</summary>
    public int HeadDim { get; }

    /// <summary>Cosines, <c>[Patches, HeadDim]</c>.</summary>
    public float[] Cos { get; }

    /// <summary>Sines, <c>[Patches, HeadDim]</c>.</summary>
    public float[] Sin { get; }

    /// <summary>Builds the table for a <paramref name="gridHeight"/> x <paramref name="gridWidth"/> patch grid.</summary>
    /// <param name="gridHeight">Patch rows.</param>
    /// <param name="gridWidth">Patch columns.</param>
    /// <param name="headDim">Per-head width, a positive multiple of 4.</param>
    /// <param name="theta">Rope base (<c>vision_rope_theta</c>).</param>
    public static DeepSeekV41VisionRopeTable Build(int gridHeight, int gridWidth, int headDim, double theta)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(gridHeight, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(gridWidth, 1);
        if (headDim < 4 || headDim % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "The head width must be a positive multiple of 4.");
        if (!(theta > 0.0) || !double.IsFinite(theta))
            throw new ArgumentOutOfRangeException(nameof(theta), theta, "The rope base must be a positive finite number.");
        int half = headDim / 2, pairs = half / 2;
        float[] invFreq = new float[pairs];
        for (int j = 0; j < pairs; j++) invFreq[j] = 1f / MathF.Pow((float)theta, 2f * j / half);

        float[] cos = new float[checked(gridHeight * gridWidth * headDim)];
        float[] sin = new float[cos.Length];
        for (int h = 0; h < gridHeight; h++)
            for (int w = 0; w < gridWidth; w++)
            {
                int row = (h * gridWidth + w) * headDim;
                for (int j = 0; j < pairs; j++)
                {
                    float rowAngle = h * invFreq[j], columnAngle = w * invFreq[j];
                    cos[row + j] = cos[row + half + j] = MathF.Cos(rowAngle);
                    sin[row + j] = sin[row + half + j] = MathF.Sin(rowAngle);
                    cos[row + pairs + j] = cos[row + half + pairs + j] = MathF.Cos(columnAngle);
                    sin[row + pairs + j] = sin[row + half + pairs + j] = MathF.Sin(columnAngle);
                }
            }
        return new DeepSeekV41VisionRopeTable(gridHeight, gridWidth, headDim, cos, sin);
    }
}
