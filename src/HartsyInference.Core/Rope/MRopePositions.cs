namespace HartsyInference.Core.Rope;

/// <summary>Multimodal RoPE position ids for one unpadded sequence (transformers' <c>get_rope_index</c> for Qwen-VL):
/// a text token advances all three axes together; an image's placeholder run gets <c>(start + t, start + row,
/// start + col)</c> over its merged grid; the text after it resumes at <c>start + max(T, H, W)</c>. Pair assignment to
/// axes is <see cref="MRopePairAxes"/>; the cos/sin rows are <see cref="MRopeTable"/>.</summary>
public static class MRopePositions
{
    /// <summary>Axis-major positions <c>[3 · seq]</c> (temporal, height, width) for <paramref name="tokenIds"/>, whose runs
    /// of <paramref name="imageTokenId"/> hold <paramref name="grids"/> in order. <paramref name="nextPosition"/> is where a
    /// following token would sit.</summary>
    public static int[] Build(ReadOnlySpan<int> tokenIds, int imageTokenId, ReadOnlySpan<MRopeImageGrid> grids, out int nextPosition)
    {
        int seq = tokenIds.Length;
        int[] positions = new int[3 * seq];
        int current = 0, image = 0, s = 0;
        while (s < seq)
        {
            if (tokenIds[s] != imageTokenId)
            {
                positions[s] = positions[seq + s] = positions[2 * seq + s] = current++;
                s++;
                continue;
            }
            if (image >= grids.Length)
                throw new ArgumentException($"The sequence holds more image placeholder runs than the {grids.Length} grid(s) given.", nameof(grids));
            MRopeImageGrid grid = grids[image++];
            if (grid.T < 1 || grid.H < 1 || grid.W < 1)
                throw new ArgumentOutOfRangeException(nameof(grids), $"An image grid must be positive, got {grid}.");
            if (s + grid.Tokens > seq)
                throw new ArgumentException($"Image {image - 1} needs {grid.Tokens} placeholders but the sequence ends first.", nameof(tokenIds));
            for (int i = 0; i < grid.Tokens; i++)
            {
                if (tokenIds[s + i] != imageTokenId)
                    throw new ArgumentException($"Image {image - 1} has {i} placeholder(s); its grid needs {grid.Tokens}.", nameof(tokenIds));
                int t = i / (grid.H * grid.W), row = i / grid.W % grid.H, col = i % grid.W;
                positions[s + i] = current + t;
                positions[seq + s + i] = current + row;
                positions[2 * seq + s + i] = current + col;
            }
            s += grid.Tokens;
            if (s < seq && tokenIds[s] == imageTokenId)
                throw new ArgumentException($"Image {image - 1}'s placeholder run is longer than its {grid.Tokens}-token grid.", nameof(tokenIds));
            current += Math.Max(grid.T, Math.Max(grid.H, grid.W));
        }
        if (image != grids.Length)
            throw new ArgumentException($"{grids.Length} image grid(s) for {image} placeholder run(s).", nameof(grids));
        nextPosition = current;
        return positions;
    }
}
