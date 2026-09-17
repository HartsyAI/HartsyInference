namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>A melody line that cannot be placed on the decoded subbeat grid — a note shorter than one subbeat,
/// or two notes quantized onto the same one.</summary>
public sealed class MelodyVoiceException : AbcRebuildException
{
    public MelodyVoiceException(string message) : base(message) { }

    public MelodyVoiceException(string message, Exception innerException) : base(message, innerException) { }
}
