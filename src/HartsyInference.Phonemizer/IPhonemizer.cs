namespace HartsyInference.Phonemizer;

/// <summary>Converts written text into phonemes for a TTS pipeline. This is the swap point for the phonemization
/// backend: the bundled <see cref="Espeak.EspeakPhonemizer"/> is a pure-C# port of espeak-ng, but a native
/// espeak-ng binding or any other engine can implement this interface instead, and model code only ever depends on
/// this contract. <paramref name="language"/> is an espeak-style voice/language name such as <c>"en-us"</c>,
/// <c>"cmn"</c>, or <c>"de"</c>.</summary>
public interface IPhonemizer
{
    /// <summary>Returns the phonemization as an IPA string (the representation most TTS models were trained on).</summary>
    string PhonemizeToIpa(string text, string language);

    /// <summary>Returns the phonemization as the raw per-phoneme mnemonics the engine produced, for callers that map
    /// mnemonics directly to model tokens rather than going through IPA.</summary>
    IReadOnlyList<string> PhonemizeToMnemonics(string text, string language);
}
