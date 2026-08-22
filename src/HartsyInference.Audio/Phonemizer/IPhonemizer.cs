namespace HartsyInference.Audio.Phonemizer;

/// <summary>Converts written text into phonemes for a TTS pipeline; the swap point for the phonemization backend — the bundled <see cref="Espeak.EspeakPhonemizer"/> is a pure-C# port of espeak-ng, but a native binding or any other engine can implement this interface instead. <paramref name="language"/> is an espeak-style voice/language name such as <c>"en-us"</c>, <c>"cmn"</c>, or <c>"de"</c>.</summary>
public interface IPhonemizer
{
    /// <summary>Returns the phonemization as an IPA string (the representation most TTS models were trained on).</summary>
    string PhonemizeToIpa(string text, string language);

    /// <summary>Returns the phonemization as the raw per-phoneme mnemonics, for callers that map mnemonics directly to model tokens rather than going through IPA.</summary>
    IReadOnlyList<string> PhonemizeToMnemonics(string text, string language);

    /// <summary>Phonemizes <paramref name="text"/> to IPA and maps it to model token ids via <paramref name="idMap"/> (e.g. a Piper voice's phoneme id map).</summary>
    int[] PhonemizeToIds(string text, string language, PhonemeIdMap idMap);
}
