namespace HartsyInference.Core.Backends.Layout;

/// <summary>One of the four axes a packed channel of a pixel shuffle encodes: the channel itself and the sub-position within the temporal, vertical and horizontal factor.</summary>
public enum ShuffleAxis
{
    Channel,
    Time,
    Height,
    Width,
}
