using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Builds MiniMax-H3's 3-axis rotary tables in the layout <see cref="Core.Backends.IBackend.ApplyRopeSingle"/>
/// consumes: NEOX split-half pairing over the first <c>rotaryDim</c> head dims, cos/sin stored at <c>headDim</c>
/// stride. The rotated width is <c>3 * 2 * invFreqLen</c> (t, h, w axes, each duplicated across the pair halves) —
/// for the reference config that is 96 of 128 dims, leaving the tail unrotated.</summary>
public static class MiniMaxH3Rope
{
    /// <summary>Head dims the rotary covers: t/h/w each contribute <paramref name="invFreqLen"/> angles, duplicated
    /// across the split-half pair.</summary>
    public static int RotaryDim(int invFreqLen) => 3 * invFreqLen * 2;

    /// <summary>Reference inverse frequencies when the checkpoint's buffer is unavailable (structural tests only) —
    /// the real values ship as <c>rope.inv_freq</c> and must be used when present.</summary>
    public static float[] DefaultInvFreq(int invFreqLen, double theta = 10000.0)
    {
        float[] inv = new float[invFreqLen];
        for (int i = 0; i < invFreqLen; i++)
        {
            inv[i] = (float)(1.0 / Math.Pow(theta, (2.0 * i) / (2.0 * invFreqLen)));
        }
        return inv;
    }

    /// <summary>Cos/sin tables shaped <c>[seq, headDim]</c> for <paramref name="positionIds"/> (row-major
    /// <c>[seq, 3]</c> in double). Angles are formed per axis as <c>pos * invFreq</c> and laid out
    /// <c>[t | h | w]</c>, then duplicated into the upper pair half so index <c>i</c> and <c>i + rotaryDim/2</c>
    /// carry the same angle — which is what makes the split-half rotation a true 2-D rotation. Dims beyond
    /// <c>rotaryDim</c> are left zero and are never read.</summary>
    public static (Tensor Cos, Tensor Sin) BuildTables(double[] positionIds, float[] invFreq, int headDim)
    {
        ArgumentNullException.ThrowIfNull(positionIds);
        ArgumentNullException.ThrowIfNull(invFreq);
        int seq = positionIds.Length / 3;
        int rotaryDim = RotaryDim(invFreq.Length);
        if (rotaryDim > headDim)
        {
            throw new ArgumentException(
                $"MiniMax-H3 rotary width {rotaryDim} (3 axes x {invFreq.Length} freqs x 2) exceeds head dim {headDim}.");
        }
        int half = rotaryDim / 2;
        int perAxis = invFreq.Length;

        Tensor cos = new Tensor(new TensorShape(seq, headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(seq, headDim), DType.F32);
        unsafe
        {
            float* pc = (float*)cos.DataPointer;
            float* ps = (float*)sin.DataPointer;
            for (int s = 0; s < seq; s++)
            {
                long row = (long)s * headDim;
                for (int axis = 0; axis < 3; axis++)
                {
                    // Angles are accumulated in double: the position grids span thousands of units and the
                    // area-normalized coordinates are fractional, so float here drifts on long sequences.
                    double pos = positionIds[s * 3 + axis];
                    for (int i = 0; i < perAxis; i++)
                    {
                        int slot = axis * perAxis + i;
                        double angle = pos * invFreq[i];
                        float c = (float)Math.Cos(angle);
                        float sn = (float)Math.Sin(angle);
                        pc[row + slot] = c;
                        ps[row + slot] = sn;
                        // Duplicate into the upper half so the pair (slot, slot + half) rotates together.
                        pc[row + slot + half] = c;
                        ps[row + slot + half] = sn;
                    }
                }
            }
        }
        return (cos, sin);
    }
}
