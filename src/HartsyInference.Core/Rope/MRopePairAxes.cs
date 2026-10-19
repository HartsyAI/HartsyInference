namespace HartsyInference.Core.Rope;

/// <summary>Which position axis (0 temporal, 1 height, 2 width) each rotary frequency pair of a head reads under
/// multimodal RoPE.</summary>
public static class MRopePairAxes
{
    /// <summary>Maps <paramref name="pairs"/> frequency pairs (half the rotary width) onto the axes of
    /// <paramref name="section"/> (pairs per axis; llama.cpp's trailing zero entry is ignored), laid out by
    /// <paramref name="layout"/>. The sections must sum to <paramref name="pairs"/>.</summary>
    public static byte[] Build(ReadOnlySpan<int> section, int pairs, MRopeLayout layout)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(pairs);
        int axes = section.Length;
        while (axes > 0 && section[axes - 1] == 0) axes--;
        if (axes is < 1 or > 3)
            throw new ArgumentException($"An M-RoPE section names 1 to 3 axes, got {section.Length} entries.", nameof(section));
        int sum = 0;
        for (int a = 0; a < axes; a++)
        {
            if (section[a] < 0)
                throw new ArgumentOutOfRangeException(nameof(section), $"M-RoPE section {a} is {section[a]}.");
            sum += section[a];
        }
        if (sum != pairs)
            throw new ArgumentException($"M-RoPE sections sum to {sum}; the rotary width has {pairs} frequency pairs.", nameof(section));
        byte[] map = new byte[pairs];
        if (layout == MRopeLayout.Sectioned)
        {
            int j = 0;
            for (int a = 0; a < axes; a++)
            {
                for (int i = 0; i < section[a]; i++) map[j++] = (byte)a;
            }
            return map;
        }
        for (int j = 0; j < pairs; j++)
        {
            int axis = j % 3;
            map[j] = axis != 0 && axis < axes && j < 3 * section[axis] ? (byte)axis : (byte)0;
        }
        return map;
    }
}
