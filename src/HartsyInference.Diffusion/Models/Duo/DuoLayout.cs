namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Where Intent positions go: <see cref="Old"/> leaves every row of a block at the block's time, <see cref="New"/> counts each token forward and packs the shared spans to end at -1.</summary>
public enum DuoRopePolicy
{
    Old,
    New,
}

/// <summary>The packed Duo sequence for one forward: groups, per-row positions and AdaLN indices, and the row lists the media and Intent live in. Built by <see cref="DuoLayoutBuilder"/>.</summary>
public sealed class DuoLayout
{
    internal DuoLayout(IReadOnlyList<DuoGroup> groups, int liveRows, int length, double[] positions, int[] tags, int[] timeIds,
        int[] videoRows, int[] audioRows, int[] textRows, int[] textSourceRows, IReadOnlyList<int> noisyGroups)
    {
        Groups = groups;
        LiveRows = liveRows;
        Length = length;
        Positions = positions;
        Tags = tags;
        TimeIds = timeIds;
        VideoRows = videoRows;
        AudioRows = audioRows;
        TextRows = textRows;
        TextSourceRows = textSourceRows;
        NoisyGroups = noisyGroups;
    }

    /// <summary>Every group in row order, padding last.</summary>
    public IReadOnlyList<DuoGroup> Groups { get; }

    /// <summary>Rows that carry data, before the padding to a multiple of 128.</summary>
    public int LiveRows { get; }

    /// <summary>Total packed rows, a multiple of 128.</summary>
    public int Length { get; }

    /// <summary>Per-row (t, h, w), row-major <c>[Length, 3]</c>, in the 25 ms clock; text rows and shared rows as the policy sets them.</summary>
    public double[] Positions { get; }

    /// <summary>Modality tag per row: video 0, text 1, audio 2, padding -1.</summary>
    public int[] Tags { get; }

    /// <summary>Index into the unique-times vector per row: 0 for the clean slot and every non-noisy row, k for the k-th noisy group (1-based, in layout order).</summary>
    public int[] TimeIds { get; }

    /// <summary>Packed row of each video row, in group order.</summary>
    public int[] VideoRows { get; }

    /// <summary>Packed row of each audio row, in group order.</summary>
    public int[] AudioRows { get; }

    /// <summary>Packed row of each Intent row, in layout order.</summary>
    public int[] TextRows { get; }

    /// <summary>For each entry of <see cref="TextRows"/>, the row of the conversation's Intent states it came from (sample order, then each sample's shared and block spans in input order).</summary>
    public int[] TextSourceRows { get; }

    /// <summary>Indices into <see cref="Groups"/> of the noisy groups, in the order their time ids were assigned.</summary>
    public IReadOnlyList<int> NoisyGroups { get; }

    /// <summary>The adaLN table row of a packed row for the 3-tag modulation layout: <c>TimeIds * 3 + max(Tags, 0)</c>.</summary>
    public int AdalnRow(int row) => TimeIds[row] * 3 + Math.Max(Tags[row], 0);
}
