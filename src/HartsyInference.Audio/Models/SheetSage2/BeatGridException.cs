namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>The decoded beats do not form a grid that measures can be laid out on.</summary>
public sealed class BeatGridException : AbcRebuildException
{
    public BeatGridException(string message) : base(message) { }

    public BeatGridException(string message, Exception innerException) : base(message, innerException) { }
}
