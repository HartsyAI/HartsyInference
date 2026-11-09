namespace HartsyInference.LLM.Transformer.Compression;

/// <summary>Rotary positions of compressed latents: each latent takes the position of its group's first token.</summary>
public static class CompressedRopePositions
{
    /// <summary>Positions <c>(firstGroup + i) * ratio</c> for <paramref name="count"/> consecutive groups.</summary>
    public static int[] GroupStart(int firstGroup, int count, int ratio)
    {
        int[] r = new int[count];
        for (int i = 0; i < count; i++) r[i] = (firstGroup + i) * ratio;
        return r;
    }
}
