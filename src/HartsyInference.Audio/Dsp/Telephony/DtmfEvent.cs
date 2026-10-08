namespace HartsyInference.Audio.Dsp.Telephony;

/// <summary>One DTMF tone heard in audio: the key, when its tone began and the sample at which the detector had
/// confirmed it. Both offsets count samples pushed into the detector since construction or the last reset.</summary>
/// <param name="Digit">The key: 0-9, <c>*</c>, <c>#</c> or A-D.</param>
/// <param name="StartSample">Offset of the first block that carried the tone (the tone began within one block before it).</param>
/// <param name="ConfirmedSample">Offset of the end of the block that confirmed it, <c>StartSample</c> plus the
/// minimum tone duration plus the block length at the earliest.</param>
public readonly record struct DtmfEvent(char Digit, long StartSample, long ConfirmedSample);
