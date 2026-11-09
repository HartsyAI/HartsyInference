namespace HartsyInference.Core.Backends.Layout;

/// <summary>How a packed channel index of a pixel shuffle nests the four <see cref="ShuffleAxis"/> values, outermost first:
/// <c>packed = ((a0·e1 + a1)·e2 + a2)·e3 + a3</c> where <c>e_k</c> is the extent of axis <c>k</c> (the channel count or a
/// factor). A factor of 1 makes its axis's position irrelevant.</summary>
public readonly struct ChannelOrder
{
    private readonly ShuffleAxis _a0;
    private readonly ShuffleAxis _a1;
    private readonly ShuffleAxis _a2;
    private readonly ShuffleAxis _a3;

    public ChannelOrder(ShuffleAxis outermost, ShuffleAxis second, ShuffleAxis third, ShuffleAxis innermost)
    {
        int mask = (1 << (int)outermost) | (1 << (int)second) | (1 << (int)third) | (1 << (int)innermost);
        if (mask != 0b1111)
            throw new ArgumentException($"A channel order names each of Channel, Time, Height and Width once; got {outermost}, {second}, {third}, {innermost}.", nameof(outermost));
        _a0 = outermost;
        _a1 = second;
        _a2 = third;
        _a3 = innermost;
    }

    /// <summary>The axis at nesting position <paramref name="position"/> (0 outermost, 3 innermost).</summary>
    public ShuffleAxis this[int position] => position switch
    {
        0 => _a0,
        1 => _a1,
        2 => _a2,
        _ => _a3,
    };

    /// <summary>Channel outermost, then time, height, width: torch <c>PixelShuffle</c>, TRELLIS, DupUp3D, LTX-2.5.</summary>
    public static ChannelOrder CTHW => new(ShuffleAxis.Channel, ShuffleAxis.Time, ShuffleAxis.Height, ShuffleAxis.Width);

    /// <summary>Channel, then the horizontal sub-position, then the vertical one: the Wan 2.2 VAE unpatchify (<c>oc = c·p² + r·p + q</c>).</summary>
    public static ChannelOrder CWHT => new(ShuffleAxis.Channel, ShuffleAxis.Width, ShuffleAxis.Height, ShuffleAxis.Time);

    /// <summary>Spatial sub-position, then temporal, channel innermost: the SeedVR2 MAGViT shuffle.</summary>
    public static ChannelOrder HWTC => new(ShuffleAxis.Height, ShuffleAxis.Width, ShuffleAxis.Time, ShuffleAxis.Channel);

    /// <summary>Patch row, patch column, channel innermost: the DiT token packing <c>(ph·p + pw)·C + c</c> (Z-Image, Lumina 2, SD3, Oasis).</summary>
    public static ChannelOrder HWCT => new(ShuffleAxis.Height, ShuffleAxis.Width, ShuffleAxis.Channel, ShuffleAxis.Time);

    /// <summary>Channel outermost, then patch row and column: the DiT token packing <c>(c·p + ph)·p + pw</c> (Krea 2, diffusers, Cosmos embedding).</summary>
    public static ChannelOrder CHWT => new(ShuffleAxis.Channel, ShuffleAxis.Height, ShuffleAxis.Width, ShuffleAxis.Time);

    public override string ToString() => $"{_a0}, {_a1}, {_a2}, {_a3}";
}
