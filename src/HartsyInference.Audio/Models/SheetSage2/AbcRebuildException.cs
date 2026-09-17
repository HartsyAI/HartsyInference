using HartsyInference.Core.Exceptions;

namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>A decoded SheetSage2 transcription that cannot be written as ABC.
///
/// <para>Not sealed: the three subclasses name which part of the reconstruction refused, matching the released
/// implementation's own hierarchy so a caller can tell a broken beat grid from a broken chord symbol.</para></summary>
public class AbcRebuildException : HartsyInferenceException
{
    public AbcRebuildException(string message) : base(message) { }

    public AbcRebuildException(string message, Exception innerException) : base(message, innerException) { }
}
