namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Output of running a phoneme's bytecode program (espeak <c>PHONEME_DATA</c>), restricted to the fields the
/// phonemizer needs: the <c>pd_param</c> set-by-program parameters (notably the allophone change/insert/append codes)
/// and the IPA name emitted by <c>i_IPA_NAME</c>. The audio-synthesis fields (formants, wave addresses) are ignored.</summary>
internal sealed class EspeakPhonemeData
{
    /// <summary>Program-set parameters, indexed by the <c>i_*</c> param ids (e.g. <see cref="EspeakProgram.ParamChangePhoneme"/>).</summary>
    public int[] Param { get; } = new int[EspeakProgram.NumParam];

    /// <summary>IPA name set by <c>i_IPA_NAME</c>; empty when the phoneme has none (then the mnemonic is converted).</summary>
    public string IpaString { get; set; } = string.Empty;

    public void Reset()
    {
        Array.Clear(Param);
        IpaString = string.Empty;
    }
}
