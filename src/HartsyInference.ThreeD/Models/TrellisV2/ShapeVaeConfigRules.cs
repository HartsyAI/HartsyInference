namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>Validation shared by the encoder and decoder configs.</summary>
internal static class ShapeVaeConfigRules
{
    internal const string ConvNeXtBlock = "SparseConvNeXtBlock3d";
    internal const string SpatialToChannelBlock = "SparseResBlockS2C3d";
    internal const string ChannelToSpatialBlock = "SparseResBlockC2S3d";

    /// <summary>Children per voxel on a factor-2 grid (2³).</summary>
    internal const int ChildGroup = 8;

    internal static void ValidateLevels(string owner, int[] modelChannels, int[] numBlocks, string[] blockType, bool[] useCheckpoint)
    {
        int levels = modelChannels.Length;
        if (levels < 1) throw new ArgumentException($"{owner} must define at least one level.");
        if (numBlocks.Length != levels) throw new ArgumentException($"{owner} num_blocks has {numBlocks.Length} entries for {levels} levels.");
        if (blockType.Length != levels) throw new ArgumentException($"{owner} block_type has {blockType.Length} entries for {levels} levels.");
        if (useCheckpoint.Length != levels)
            throw new ArgumentException($"{owner} block_args has {useCheckpoint.Length} entries for {levels} levels.");
        for (int i = 0; i < levels; i++)
        {
            if (modelChannels[i] <= 0) throw new ArgumentException($"{owner} level {i} has {modelChannels[i]} channels; channels must be positive.");
            if (numBlocks[i] < 0) throw new ArgumentException($"{owner} level {i} has {numBlocks[i]} blocks; block counts cannot be negative.");
            if (blockType[i] != ConvNeXtBlock)
                throw new ArgumentException($"{owner} level {i} block type '{blockType[i]}' is not implemented; only {ConvNeXtBlock} is.");
        }
    }

    internal static void ValidateTransitionTypes(string owner, string[] types, int expectedCount, string expected)
    {
        if (types.Length != expectedCount)
            throw new ArgumentException($"{owner} transition types have {types.Length} entries for {expectedCount} transitions.");
        foreach (string type in types)
            if (type != expected) throw new ArgumentException($"{owner} transition type '{type}' is not implemented; only {expected} is.");
    }
}
