using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Dinov3;

/// <summary>DINOv3 patch RoPE tables for <see cref="HartsyInference.Core.Backends.IBackend.ApplyRopeSingle"/>.
/// Two axes (row, column) each own <c>headDim/4</c> frequencies: channel pair <c>p</c> of the first <c>headDim/2</c>
/// channels reads axis <c>p / (headDim/4)</c> at frequency <c>p % (headDim/4)</c>. The angles are tiled over the head,
/// so rotate-half partners <c>(i, i + headDim/2)</c> share one angle. Prefix tokens (CLS and storage) get the identity
/// rotation, so the rotary op leaves them unchanged.</summary>
internal static class Dinov3RopeTables
{
    private const double RopeBase = 100.0;
    private const float TwoPi = 6.2831855f;

    /// <summary>Builds <c>[batch, prefix + gridH·gridW, headDim]</c> cos and sin tables; the caller disposes both.
    /// <paramref name="roundPeriodsToBf16"/> follows the reference the checkpoint is compared with, see <see cref="Periods"/>.</summary>
    public static unsafe (Tensor Cos, Tensor Sin) Build(int batch, int prefix, int gridH, int gridW, int headDim, bool roundPeriodsToBf16 = true,
        bool normalizeByLongSide = false)
    {
        int tokens = prefix + gridH * gridW;
        int half = headDim / 2, quarter = headDim / 4;
        float[] periods = Periods(headDim, roundPeriodsToBf16);
        int longSide = Math.Max(gridH, gridW);
        float[] coordH = AxisCoords(gridH, normalizeByLongSide ? longSide : gridH), coordW = AxisCoords(gridW, normalizeByLongSide ? longSide : gridW);

        Tensor cos = new(new TensorShape(batch, tokens, headDim), DType.F32);
        Tensor sin = new(new TensorShape(batch, tokens, headDim), DType.F32);
        float* cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        for (int t = 0; t < tokens; t++)
        {
            for (int b = 0; b < batch; b++)
            {
                long row = ((long)b * tokens + t) * headDim;
                if (t < prefix)
                {
                    for (int c = 0; c < headDim; c++)
                    {
                        cp[row + c] = 1f;
                        sp[row + c] = 0f;
                    }
                    continue;
                }
                int patch = t - prefix, h = patch / gridW, w = patch % gridW;
                for (int p = 0; p < half; p++)
                {
                    float coord = p < quarter ? coordH[h] : coordW[w];
                    float angle = TwoPi * coord / periods[p % quarter];
                    float c0 = MathF.Cos(angle), s0 = MathF.Sin(angle);
                    cp[row + p] = c0; sp[row + p] = s0;
                    cp[row + p + half] = c0; sp[row + p + half] = s0;
                }
            }
        }
        return (cos, sin);
    }

    /// <summary>Frequencies <c>base^(2k / (headDim/2))</c>, k in [0, headDim/4), in float32 (the checkpoint carries no periods of
    /// its own). Rounded to bfloat16 by default, as the reference HairDeck loader does; Hugging Face's <c>DINOv3ViTModel</c> keeps
    /// <c>inv_freq</c> in float32 even when its weights load as bfloat16, so a checkpoint compared with it takes <paramref name="roundToBf16"/> false.</summary>
    public static float[] Periods(int headDim, bool roundToBf16 = true)
    {
        int quarter = headDim / 4;
        float[] periods = new float[quarter];
        for (int k = 0; k < quarter; k++)
        {
            float period = MathF.Pow((float)RopeBase, 2f * k / (headDim / 2f));
            periods[k] = roundToBf16 ? RoundToBf16(period) : period;
        }
        return periods;
    }

    /// <summary>Cell centres of <paramref name="n"/> cells mapped to [-1, 1] in float32, as the reference computes them. <paramref name="norm"/> is the
    /// length they are divided by: <paramref name="n"/> itself per axis, or the longer grid side when both axes share one scale (timm's <c>max</c> mode, which
    /// maps the shorter axis into a sub-range of [-1, 1]).</summary>
    private static float[] AxisCoords(int n, int norm)
    {
        float[] coords = new float[n];
        for (int i = 0; i < n; i++)
            coords[i] = 2f * ((i + 0.5f) / norm) - 1f;
        return coords;
    }

    /// <summary>Round-to-nearest-even float32 to bfloat16 and back (torch's <c>.to(torch.bfloat16)</c>).</summary>
    internal static float RoundToBf16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint lsb = (bits >> 16) & 1u;
        bits = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
        return BitConverter.UInt32BitsToSingle(bits);
    }
}
