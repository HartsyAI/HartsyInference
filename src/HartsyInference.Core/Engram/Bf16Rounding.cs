namespace HartsyInference.Core.Engram;

/// <summary>Float32 to bf16 with round-to-nearest-even, which is what a torch <c>.to(torch.bfloat16)</c> does.</summary>
internal static class Bf16Rounding
{
    public static ushort FromSingle(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
            return (ushort)((bits >> 16) | 0x0040u); // quiet the NaN so truncation cannot turn it into infinity
        return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
    }
}
