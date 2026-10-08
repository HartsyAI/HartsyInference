namespace HartsyInference.Audio.Dsp.Telephony;

/// <summary>What kind of call-progress evidence a <see cref="CallProgressClassifier"/> found. These are signals with a
/// confidence, not verdicts: tone kinds are reliable, the speech and music kinds are heuristics.</summary>
public enum CallProgressKind
{
    /// <summary>Ringing heard from the far end (440+480 Hz 2 s on / 4 s off, or 425 Hz 1 s on / 4 s off). Reliable.</summary>
    RingbackTone,

    /// <summary>Busy signal (480+620 Hz or 425 Hz, about 0.5 s on / 0.5 s off). Reliable once two cycles have been heard.</summary>
    BusyTone,

    /// <summary>Fast busy / reorder / congestion (480+620 Hz or 425 Hz, about 0.25 s on / 0.25 s off). Reliable.</summary>
    FastBusyTone,

    /// <summary>A special-information-tone triple (three rising tones ahead of an "intercept" recording: number
    /// unobtainable, circuits busy). Reliable when all three tones are heard; the first one is also reported as
    /// <see cref="Beep"/>.</summary>
    SitTone,

    /// <summary>Steady dial tone (350+440 Hz, or 425 Hz for several seconds).</summary>
    DialTone,

    /// <summary>A short single tone of 700-2200 Hz, the voicemail "record now" beep among other things. The tone is
    /// reliable; what it means is not (see <see cref="MachineGreeting"/>).</summary>
    Beep,

    /// <summary>Evidence that an answering machine or recording answered: a beep after a long greeting (strong), or long
    /// uninterrupted speech from the far end with no turn-taking (heuristic).</summary>
    MachineGreeting,

    /// <summary>Evidence of a live person: a short utterance followed by waiting, or speech that answers ours (heuristic).</summary>
    HumanSpeech,

    /// <summary>Several seconds of non-speech audio that is neither silence nor a tone: hold music, or noise (heuristic).</summary>
    HoldMusic,

    /// <summary>Several seconds of silence.</summary>
    PromptSilence,
}

/// <summary>Why a <see cref="CallProgressEvent"/> was raised.</summary>
public enum CallProgressReason
{
    /// <summary>A tone burst pattern with the expected on/off timing.</summary>
    ToneCadence,

    /// <summary>A sequence of tones (the SIT triple).</summary>
    ToneSequence,

    /// <summary>A steady tone.</summary>
    SteadyTone,

    /// <summary>A beep followed a long stretch of far-end speech.</summary>
    BeepAfterGreeting,

    /// <summary>Far-end speech ran long with no pause that would invite a reply.</summary>
    LongUninterruptedSpeech,

    /// <summary>A short far-end utterance, then silence (a person saying "hello" and waiting).</summary>
    ShortUtteranceThenSilence,

    /// <summary>Far-end speech began soon after the local side spoke, and was short.</summary>
    TurnTaking,

    /// <summary>Sustained non-speech sound.</summary>
    SustainedNonSpeech,

    /// <summary>Sustained silence.</summary>
    SustainedSilence,
}

/// <summary>One finding of a <see cref="CallProgressClassifier"/>. Offsets count samples pushed into the classifier since
/// construction or the last reset. The same kind can be raised again with a higher <see cref="Confidence"/> as more
/// evidence arrives.</summary>
/// <param name="Kind">What was found.</param>
/// <param name="Confidence">0-1. Tones 0.6-0.95; speech and music heuristics never exceed 0.7 (0.9 for a beep after a
/// greeting).</param>
/// <param name="Reason">The evidence.</param>
/// <param name="StartSample">Where the evidence began.</param>
/// <param name="ConfirmedSample">When the classifier decided.</param>
/// <param name="FrequencyHz">The tone's frequency for <see cref="CallProgressKind.Beep"/>, else 0.</param>
/// <param name="DurationMs">Length of the evidence at the decision (a beep's length, the speech so far).</param>
public readonly record struct CallProgressEvent(CallProgressKind Kind, float Confidence, CallProgressReason Reason,
    long StartSample, long ConfirmedSample, float FrequencyHz, int DurationMs);
