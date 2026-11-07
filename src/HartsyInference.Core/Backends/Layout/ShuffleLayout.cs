namespace HartsyInference.Core.Backends.Layout;

/// <summary>How a tensor of a pixel shuffle stores its grid.</summary>
public enum ShuffleLayout
{
    /// <summary><c>[B, C, T, H, W]</c> (or <c>[B, C, H, W]</c> for one frame): channels before the grid.</summary>
    ChannelsFirst,

    /// <summary><c>[B·T·H·W, C]</c> rows (any leading shape with that element count): one token per grid cell, channels last.</summary>
    Tokens,
}
