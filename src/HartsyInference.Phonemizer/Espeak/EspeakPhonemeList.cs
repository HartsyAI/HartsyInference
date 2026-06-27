namespace HartsyInference.Phonemizer.Espeak;

/// <summary>Builds the clause phoneme list from per-word phoneme codes and runs espeak's change-phonemes pass
/// (<c>MakePhonemeList</c> in phonemelist.c): allophone selection via the bytecode programs (CHANGEPHONEME / INSERT /
/// REPLACE_NEXT) followed by consecutive-unstressed-syllable reduction. The result is rendered to IPA by
/// <see cref="EspeakPhonemeRenderer"/>. Audio-only behaviours (vowel pauses, consonant doubling) are omitted.</summary>
internal sealed class EspeakPhonemeList
{
    // stress_phonemes[] indexed by stress level (matches EspeakStress); inverted here to recover the level.
    private static readonly byte[] StressPhonemes = [3, 2, 4, 5, 6, 7, 26];

    private readonly EspeakPhonemeTable _phon;
    private readonly EspeakPhonemeInterpreter _interp;
    private readonly int _stressFlags;
    private readonly EspeakPhoneme _pause;

    public EspeakPhonemeList(EspeakPhonemeTable phonemeTable, EspeakPhonemeInterpreter interpreter, int stressFlags)
    {
        _phon = phonemeTable;
        _interp = interpreter;
        _stressFlags = stressFlags;
        _pause = phonemeTable.TryGet(1, out EspeakPhoneme p) ? p : default;
    }

    private static int LevelOf(byte code)
    {
        int ix = Array.IndexOf(StressPhonemes, code);
        return ix < 0 ? -1 : ix;
    }

    /// <summary>Builds the phoneme list for one clause from each word's stressed phoneme codes, then applies the
    /// change-phonemes pass and reduction. Guard pause entries bracket the list so context conditions are safe.</summary>
    public List<EspeakPhonemeListEntry> Build(IReadOnlyList<IReadOnlyList<byte>> words)
    {
        List<EspeakPhonemeListEntry> list = [PauseEntry(), PauseEntry()]; // leading guards (indices 0,1)

        for (int w = 0; w < words.Count; w++)
        {
            int pendingStress = EspeakProgram.StressUnstressed;
            bool wordStart = true;
            int wordStartIndex = list.Count;
            int wordMax = 0;
            List<int> wordVowelEntries = [];

            foreach (byte code in words[w])
            {
                if (!_phon.TryGet(code, out EspeakPhoneme ph)) continue;
                if (ph.Type == EspeakPhoneme.TypeStress)
                {
                    int lvl = LevelOf(code);
                    if (lvl >= 0) pendingStress = lvl; // stress markers become stresslevel on the next vowel
                    continue;
                }

                EspeakPhonemeListEntry e = new(code, ph) { Type = ph.Type };
                if (wordStart) { e.SourceIx = w + 1; wordStart = false; }
                if (ph.Type == EspeakPhoneme.TypeVowel)
                {
                    e.SynthFlags |= EspeakProgram.SflagSyllable;
                    e.StressLevel = pendingStress;
                    if (pendingStress > wordMax) wordMax = pendingStress;
                    pendingStress = EspeakProgram.StressUnstressed;
                    wordVowelEntries.Add(list.Count);
                }
                list.Add(e);
            }
            foreach (int vi in wordVowelEntries) list[vi].WordStress = wordMax;
            if (wordStartIndex < list.Count) _ = wordStartIndex; // word emitted
        }

        list.Add(PauseEntry());
        list.Add(PauseEntry()); // trailing guards

        ChangePhonemesPass(list);
        return list;
    }

    private EspeakPhonemeListEntry PauseEntry() => new(1, _pause) { Type = EspeakPhoneme.TypePause };

    // Port of the MakePhonemeList change-phonemes loop + consecutive-unstressed reduction.
    private void ChangePhonemesPass(List<EspeakPhonemeListEntry> list)
    {
        EspeakPhonemeData phdata = new();
        int unstressCount = 0;

        for (int j = 2; j < list.Count - 2; j++)
        {
            EspeakPhonemeListEntry e = list[j];
            if (e.Type == EspeakPhoneme.TypePause) { unstressCount = 0; continue; }

            _interp.Interpret(list, j, EspeakProgram.ControlChangePhonemes, tr: true, phdata);

            int alt = phdata.Param[EspeakProgram.ParamReplaceNextPhoneme];
            if (alt > 0 && _phon.TryGet(alt, out EspeakPhoneme rep))
            {
                list[j + 1].Ph = rep; list[j + 1].PhCode = alt; list[j + 1].Type = rep.Type;
            }

            alt = phdata.Param[EspeakProgram.ParamChangePhoneme];
            if (alt > 0 && _phon.TryGet(alt, out EspeakPhoneme chg))
            {
                EspeakPhoneme old = e.Ph;
                e.Ph = chg; e.PhCode = alt; e.Type = chg.Type;
                if (chg.Type == EspeakPhoneme.TypeVowel)
                {
                    e.SynthFlags |= EspeakProgram.SflagSyllable;
                    if (old.Type != EspeakPhoneme.TypeVowel) e.StressLevel = 0;
                }
                else e.SynthFlags &= ~EspeakProgram.SflagSyllable;
                _interp.Interpret(list, j, EspeakProgram.ControlChangePhonemes, tr: true, phdata); // re-interpret (no 2nd change)
            }

            if (e.Type == EspeakPhoneme.TypeVowel)
            {
                if (e.StressLevel <= 1)
                {
                    unstressCount++;
                    if ((_stressFlags & 0x08) != 0)
                    {
                        // Reduce sequences of consecutive unstressed vowels in unstressed words to diminished.
                        for (int k = j + 1; list[k].Type != EspeakPhoneme.TypePause; k++)
                        {
                            if (list[k].Type != EspeakPhoneme.TypeVowel) continue;
                            if (list[k].StressLevel <= 1)
                            {
                                if (e.WordStress < 4) e.StressLevel = 0;
                                if (list[k].WordStress < 4) list[k].StressLevel = 0;
                            }
                            break;
                        }
                    }
                    else if ((unstressCount > 1) && ((unstressCount & 1) == 0))
                    {
                        // Alternate-syllable reduction, except a final unstressed vowel of a stressed word.
                        bool noDim = e.WordStress > 3 && list[j + 1].SourceIx != 0;
                        if (noDim) unstressCount = 1;
                        else e.StressLevel = 0;
                    }
                }
                else unstressCount = 0;
            }

            // Append a linking phoneme (e.g. the rhotic ɹ that a rhotic vowel inserts before a following vowel).
            int append = phdata.Param[EspeakProgram.ParamAppendPhoneme];
            if (append > 0 && _phon.TryGet(append, out EspeakPhoneme ap))
            {
                list.Insert(j + 1, new EspeakPhonemeListEntry(append, ap) { Type = ap.Type });
                j++; // don't reinterpret the inserted phoneme as a change site
            }
        }
    }
}
