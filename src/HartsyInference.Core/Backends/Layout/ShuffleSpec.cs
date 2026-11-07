namespace HartsyInference.Core.Backends.Layout;

/// <summary>The geometry of one depth-to-space or space-to-depth move (<see cref="PixelShuffles"/>). The dense grid has
/// <see cref="Batch"/> × <see cref="Frames"/> × <see cref="Height"/> × <see cref="Width"/> cells of
/// <see cref="Channels"/>·<see cref="FactorT"/>·<see cref="FactorH"/>·<see cref="FactorW"/> packed channels, nested per
/// <see cref="Order"/>; the unpacked side has <see cref="Channels"/> channels on a grid scaled by the factors.</summary>
/// <param name="PackedLayout">Layout of the packed (small grid, many channels) tensor.</param>
/// <param name="UnpackedLayout">Layout of the unpacked (large grid, few channels) tensor.</param>
/// <param name="ChannelRepeat">Packed channels per stored channel: stored channel <c>packed / ChannelRepeat</c> (DupUp3D's <c>repeat_interleave</c>). 1 for a plain shuffle.</param>
/// <param name="DropLeadingFrames">Unpacked frames discarded from the front (a causal shuffle's duplicated frames); frames that would land before 0 are not written.</param>
/// <param name="DropDuplicateFrame">Drops unpacked frame index 1 and shifts the later frames down by one (SeedVR2's duplicated half of frame 0).</param>
/// <param name="ChunkStartFrame">First packed frame of this call on the full temporal axis, when the unpacked tensor spans several calls.</param>
/// <param name="UnpackedFrames">Frames of the unpacked tensor; -1 derives it from this call's frames and drops.</param>
public readonly record struct ShuffleSpec(
    ShuffleLayout PackedLayout,
    ShuffleLayout UnpackedLayout,
    int Batch,
    int Frames,
    int Height,
    int Width,
    int Channels,
    int FactorT,
    int FactorH,
    int FactorW,
    ChannelOrder Order,
    int ChannelRepeat = 1,
    int DropLeadingFrames = 0,
    bool DropDuplicateFrame = false,
    int ChunkStartFrame = 0,
    int UnpackedFrames = -1)
{
    /// <summary>Packed channels per grid cell.</summary>
    public long PackedChannels => (long)Channels * FactorT * FactorH * FactorW;

    /// <summary>Channels the packed tensor stores per cell (packed channels over <see cref="ChannelRepeat"/>).</summary>
    public long StoredChannels => PackedChannels / ChannelRepeat;

    /// <summary>Frames of the unpacked tensor.</summary>
    public int UnpackedFrameCount => UnpackedFrames >= 0 ? UnpackedFrames : Frames * FactorT - DropLeadingFrames - (DropDuplicateFrame ? 1 : 0);

    /// <summary>Extent of one packed axis: the channel count or that axis's factor.</summary>
    public int Extent(ShuffleAxis axis) => axis switch
    {
        ShuffleAxis.Channel => Channels,
        ShuffleAxis.Time => FactorT,
        ShuffleAxis.Height => FactorH,
        _ => FactorW,
    };

    /// <summary>A channels-first shuffle of a <c>[B, C·fT·fH·fW, T, H, W]</c> tensor into <c>[B, C, T·fT, H·fH, W·fW]</c>.</summary>
    public static ShuffleSpec ChannelsFirst(int batch, int frames, int height, int width, int channels, int factorT, int factorH, int factorW, ChannelOrder order)
        => new(ShuffleLayout.ChannelsFirst, ShuffleLayout.ChannelsFirst, batch, frames, height, width, channels, factorT, factorH, factorW, order);

    /// <summary>The DiT token grid: packed tokens <c>[B, hP·wP, C·p²]</c> against an image <c>[B, C, hP·p, wP·p]</c>.</summary>
    public static ShuffleSpec TokenGrid(int batch, int hPacked, int wPacked, int channels, int patch, bool innerChannelFastest)
        => new(ShuffleLayout.Tokens, ShuffleLayout.ChannelsFirst, batch, 1, hPacked, wPacked, channels, 1, patch, patch, innerChannelFastest ? ChannelOrder.HWCT : ChannelOrder.CHWT);
}
