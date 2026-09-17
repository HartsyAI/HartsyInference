namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>A chord label that has no ABC spelling. Refusing beats rewriting it as something else: a chord
/// silently flattened to major is a wrong transcription that nothing downstream can detect.</summary>
public sealed class ChordSymbolException : AbcRebuildException
{
    public ChordSymbolException(string message) : base(message) { }

    public ChordSymbolException(string message, Exception innerException) : base(message, innerException) { }
}
