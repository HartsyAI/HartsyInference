using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis rotary positional embedding for Qwen-Image 2.1 (<c>EmbedND</c> + <c>apply_rope1</c> in ComfyUI's
/// <c>qwen_image21/model.py</c>). The per-axis frequency math is identical to <see cref="QwenImageRope"/> — same
/// <c>[16,56,56]</c> split, same θ=10000, same axis-major pair order, same <c>(2i, 2i+1)</c> rotation — but the
/// <b>positions differ</b>, so the two cannot share a table.
///
/// <para>2.1 walks one running counter over the whole sequence: a text run of <c>n</c> tokens takes positions
/// <c>pos … pos+n−1</c> on all three axes and advances <c>pos</c> by <c>n</c>; the image then takes that
/// <c>pos</c> on the sequence axis and <b>centered</b> row/column indices on the other two. v1 instead pins images
/// to sequence axis 0 and starts text at <c>max(H/2, W/2)</c>, which is why this is a separate class, not a flag.</para>
///
/// <para>Text and image tables are built separately because the two halves of the sequence are evaluated at
/// different times: the text prefix runs once per prompt, the image rows once per denoise step (see
/// <see cref="QwenImage21Transformer"/>). Centering is <c>r − (h − h/2)</c> with integer division; the half-token
/// term ComfyUI adds for a reference grid of the opposite parity is zero for the target image, which is the only
/// image in a text-to-image sequence.</para>
///
/// <para>The angle is <c>position / Pow(theta, 2k/axisDim)</c>, a division by the denominator, so the tables use
/// <see cref="RopeAngleOrder.DivideByDenominator"/>. Multiplying by the reciprocal would change the last bit.</para></summary>
public sealed class QwenImage21Rope : IDisposable
{
    private static readonly RopeTableSettings DeviceTableSettings = new()
    {
        AngleOrder = RopeAngleOrder.DivideByDenominator,
        Layout = RopeTableLayout.InterleavedPair,
        Preload = true,
        RetainedSignatures = 2,
    };

    private const int TextSlot = 0;
    private const int ImageSlot = 1;

    private readonly int[] _axesDim;
    private readonly int _headDim;
    private readonly RopeTableCache _deviceTables;

    // Cached per backend, exactly as QwenImageRope does and for the same reason: each DiT shard keeps its own tables, and
    // a build on one backend never frees tables another backend still reads.
    private readonly RopeHeldTables<int> _textTables;
    private readonly RopeHeldTables<(int, int, int)> _imageTables;

    public QwenImage21Rope(int[]? axesDim = null, int theta = 10000)
    {
        _axesDim = axesDim ?? [16, 56, 56];
        if (_axesDim.Length != 3)
            throw new ArgumentException("QwenImage21Rope requires exactly 3 axes (sequence, height, width).", nameof(axesDim));
        foreach (int d in _axesDim)
        {
            if (d % 2 != 0)
                throw new ArgumentException($"Each RoPE axis must be even; got {d}.", nameof(axesDim));
        }
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = _axesDim, Theta = theta });
        _headDim = plan.HeadDim;
        _deviceTables = new RopeTableCache(plan, DeviceTableSettings);
        _textTables = new RopeHeldTables<int>(_deviceTables, TextSlot, QwenImage21PosEmbed.NumAxes);
        _imageTables = new RopeHeldTables<(int, int, int)>(_deviceTables, ImageSlot, QwenImage21PosEmbed.NumAxes);
    }

    /// <summary>Sum of the per-axis dimensions; equals the attention head dim.</summary>
    public int HeadDim => _headDim;

    /// <summary>Per-token <c>(sequence, height, width)</c> positions for the text prefix: token <c>i</c> sits at
    /// <c>i</c> on all three axes.</summary>
    public static (double Seq, double Height, double Width)[] TextPositions(int textLen) =>
        ToTuples(QwenImage21PosEmbed.TextRows(textLen));

    /// <summary>Per-token positions for the target image, row-major over <paramref name="imgH"/>×<paramref name="imgW"/>.
    /// The sequence axis is the constant <paramref name="textLen"/> (the counter's value after the text run); the
    /// other two are centered on the grid.</summary>
    public static (double Seq, double Height, double Width)[] ImagePositions(int textLen, int imgH, int imgW) =>
        ToTuples(QwenImage21PosEmbed.ImageRows(textLen, imgH, imgW));

    /// <summary>Builds (or returns the cached) text-prefix tables, <c>[textLen, headDim]</c> F32 in the
    /// <see cref="IBackend.WanRopeInterleaved"/> convention (pair <c>i</c>'s angle duplicated at <c>2i</c> and
    /// <c>2i+1</c>).</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildTextTables(IBackend backend, int textLen) =>
        _textTables.GetOrBuild(backend, textLen, () => QwenImage21PosEmbed.TextRows(textLen));

    /// <summary>Builds (or returns the cached) target-image tables, <c>[imgH·imgW, headDim]</c>. Keyed on the text
    /// length too, since that sets the image rows' sequence-axis position.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildImageTables(IBackend backend, int textLen, int imgH, int imgW) =>
        _imageTables.GetOrBuild(backend, (textLen, imgH, imgW), () => QwenImage21PosEmbed.ImageRows(textLen, imgH, imgW));

    /// <summary>Releases the held tables and the cache. Tables still under a lease are freed by their last release.</summary>
    public void Dispose()
    {
        _textTables.Dispose();
        _imageTables.Dispose();
        _deviceTables.Dispose();
    }

    private static (double Seq, double Height, double Width)[] ToTuples(float[] rows)
    {
        (double Seq, double Height, double Width)[] positions = new (double Seq, double Height, double Width)[rows.Length / QwenImage21PosEmbed.NumAxes];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = (rows[i * 3], rows[i * 3 + 1], rows[i * 3 + 2]);
        return positions;
    }
}
