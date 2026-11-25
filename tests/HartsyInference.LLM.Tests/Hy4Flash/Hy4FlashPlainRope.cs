using HartsyInference.LLM.Hy4Flash;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>Plain interleaved rotary rows for the tests: every slot carries its position on all three M-RoPE planes, so the rows reduce to plain RoPE.</summary>
internal static class Hy4FlashPlainRope
{
    /// <param name="halfDim">Pairs per row, half the rotated width.</param>
    /// <param name="length">Positions covered.</param>
    /// <param name="theta">RoPE base.</param>
    public static Hy4FlashMropeRows Create(int halfDim, int length, double theta)
    {
        Hy4FlashMropeRows rows = new(halfDim, length, theta, [halfDim, 0, 0]);
        for (int pos = 0; pos < length; pos++) rows.SetPosition(pos, pos, pos, pos);
        return rows;
    }
}
