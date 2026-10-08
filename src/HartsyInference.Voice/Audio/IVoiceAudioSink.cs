using HartsyInference.Audio.Dsp.Telephony;
using HartsyInference.Voice.Turns;

namespace HartsyInference.Voice.Audio;

/// <summary>Where the audio thread hands what it decided. Called on the audio thread, once per utterance, barge-in or
/// failure, never per frame; implementations only queue work for other threads.</summary>
internal interface IVoiceAudioSink
{
    /// <summary>An utterance to answer.</summary>
    void OnUtterance(VoiceTurnInput utterance);

    /// <summary>Speech over the reply that never became a barge-in, <paramref name="samples"/> long.</summary>
    void OnUtteranceDiscarded(int samples);

    /// <summary>The caller barged in on <paramref name="turnId"/>; the flush and cancellation are already requested.</summary>
    void OnBargeIn(int turnId, long detectNs);

    /// <summary>A DTMF tone was heard in the audio. Default: ignored, so a sink that does not care keeps compiling.</summary>
    void OnInbandDtmf(DtmfEvent tone, long detectNs)
    {
    }

    /// <summary>The far end's audio was classified. Default: ignored.</summary>
    void OnCallProgress(CallProgressEvent finding, long detectNs)
    {
    }

    /// <summary>The audio thread failed and has stopped.</summary>
    void OnAudioFault(Exception error);
}
