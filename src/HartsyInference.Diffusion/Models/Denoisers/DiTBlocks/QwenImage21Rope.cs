using HartsyInference.Core.Backends;
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
/// image in a text-to-image sequence.</para></summary>
public sealed unsafe class QwenImage21Rope : IDisposable
{
    private readonly int[] _axesDim;
    private readonly int _theta;
    private readonly int _headDim;

    // Cached per backend, exactly as QwenImageRope does and for the same reason: two backends staging one host
    // tensor through the transfer helper trips its pinning state, so each card builds its own few-MB copy.
    private readonly Dictionary<IBackend, (int Key, Tensor Cos, Tensor Sin)> _textTables = new();
    private readonly Dictionary<IBackend, ((int Text, int H, int W) Key, Tensor Cos, Tensor Sin)> _imageTables = new();

    public QwenImage21Rope(int[]? axesDim = null, int theta = 10000)
    {
        _axesDim = axesDim ?? [16, 56, 56];
        if (_axesDim.Length != 3)
            throw new ArgumentException("QwenImage21Rope requires exactly 3 axes (sequence, height, width).", nameof(axesDim));
        _theta = theta;
        _headDim = 0;
        foreach (int d in _axesDim)
        {
            if (d % 2 != 0)
                throw new ArgumentException($"Each RoPE axis must be even; got {d}.", nameof(axesDim));
            _headDim += d;
        }
    }

    /// <summary>Sum of the per-axis dimensions; equals the attention head dim.</summary>
    public int HeadDim => _headDim;

    /// <summary>Per-token <c>(sequence, height, width)</c> positions for the text prefix: token <c>i</c> sits at
    /// <c>i</c> on all three axes.</summary>
    public static (double Seq, double Height, double Width)[] TextPositions(int textLen)
    {
        (double, double, double)[] positions = new (double, double, double)[textLen];
        for (int i = 0; i < textLen; i++)
            positions[i] = (i, i, i);
        return positions;
    }

    /// <summary>Per-token positions for the target image, row-major over <paramref name="imgH"/>×<paramref name="imgW"/>.
    /// The sequence axis is the constant <paramref name="textLen"/> (the counter's value after the text run); the
    /// other two are centered on the grid.</summary>
    public static (double Seq, double Height, double Width)[] ImagePositions(int textLen, int imgH, int imgW)
    {
        (double, double, double)[] positions = new (double, double, double)[imgH * imgW];
        double hCenter = imgH - imgH / 2;
        double wCenter = imgW - imgW / 2;
        for (int r = 0; r < imgH; r++)
        {
            for (int c = 0; c < imgW; c++)
                positions[r * imgW + c] = (textLen, r - hCenter, c - wCenter);
        }
        return positions;
    }

    /// <summary>Builds (or returns the cached) text-prefix tables, <c>[textLen, headDim]</c> F32 in the
    /// <see cref="IBackend.WanRopeInterleaved"/> convention (pair <c>i</c>'s angle duplicated at <c>2i</c> and
    /// <c>2i+1</c>).</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildTextTables(IBackend backend, int textLen)
    {
        lock (_textTables)
        {
            if (_textTables.TryGetValue(backend, out (int Key, Tensor Cos, Tensor Sin) entry) && entry.Key == textLen)
                return (entry.Cos, entry.Sin);
            entry.Cos?.Dispose();
            entry.Sin?.Dispose();
            (Tensor cos, Tensor sin) = Build(TextPositions(textLen));
            _textTables[backend] = (textLen, cos, sin);
            return (cos, sin);
        }
    }

    /// <summary>Builds (or returns the cached) target-image tables, <c>[imgH·imgW, headDim]</c>. Keyed on the text
    /// length too, since that sets the image rows' sequence-axis position.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildImageTables(IBackend backend, int textLen, int imgH, int imgW)
    {
        (int, int, int) key = (textLen, imgH, imgW);
        lock (_imageTables)
        {
            if (_imageTables.TryGetValue(backend, out ((int, int, int) Key, Tensor Cos, Tensor Sin) entry) && entry.Key == key)
                return (entry.Cos, entry.Sin);
            entry.Cos?.Dispose();
            entry.Sin?.Dispose();
            (Tensor cos, Tensor sin) = Build(ImagePositions(textLen, imgH, imgW));
            _imageTables[backend] = (key, cos, sin);
            return (cos, sin);
        }
    }

    private (Tensor Cos, Tensor Sin) Build((double Seq, double Height, double Width)[] positions)
    {
        Tensor cos = new Tensor(new TensorShape(positions.Length, _headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(positions.Length, _headDim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        for (int s = 0; s < positions.Length; s++)
            FillToken(cp + (long)s * _headDim, sp + (long)s * _headDim, positions[s]);
        return (cos, sin);
    }

    /// <summary>Writes one token's interleaved cos/sin row. Axis-major (<c>EmbedND</c> concatenates each axis's pairs
    /// in order), angle <c>pos · theta^(−2k/axisDim)</c>, each pair's angle duplicated across both of its slots.</summary>
    private void FillToken(float* cos, float* sin, (double Seq, double Height, double Width) position)
    {
        Span<double> axes = [position.Seq, position.Height, position.Width];
        int slot = 0;
        for (int axis = 0; axis < 3; axis++)
        {
            int axisDim = _axesDim[axis];
            for (int k = 0; k < axisDim / 2; k++)
            {
                double angle = axes[axis] / Math.Pow(_theta, (double)(2 * k) / axisDim);
                float c = (float)Math.Cos(angle);
                float s = (float)Math.Sin(angle);
                cos[slot] = c;
                cos[slot + 1] = c;
                sin[slot] = s;
                sin[slot + 1] = s;
                slot += 2;
            }
        }
    }

    public void Dispose()
    {
        lock (_textTables)
        {
            foreach ((_, Tensor cos, Tensor sin) in _textTables.Values) { cos.Dispose(); sin.Dispose(); }
            _textTables.Clear();
        }
        lock (_imageTables)
        {
            foreach ((_, Tensor cos, Tensor sin) in _imageTables.Values) { cos.Dispose(); sin.Dispose(); }
            _imageTables.Clear();
        }
    }
}
