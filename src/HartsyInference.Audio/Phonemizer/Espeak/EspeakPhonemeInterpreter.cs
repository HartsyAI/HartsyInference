using System.Text;

namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Interprets a phoneme's bytecode program from <c>phonindex</c> (ported from <c>InterpretPhoneme</c> + <c>InterpretCondition</c> in espeak-ng synthdata.c); over a phoneme-list context it evaluates the program's conditions (prev/next phoneme type, stress, word boundaries) and produces the allophone-change parameters (CHANGE/INSERT/REPLACE/APPEND) and the IPA name. Audio-synthesis instructions are walked for control flow but their data is ignored.</summary>
internal sealed class EspeakPhonemeInterpreter
{
    private readonly EspeakPhonemeTable _phon;
    private readonly ushort[] _prog;
    private readonly EspeakPhoneme _pause; // sentinel for out-of-range list access

    public EspeakPhonemeInterpreter(EspeakPhonemeTable phonemeTable, EspeakPhonemeIndex index)
    {
        _phon = phonemeTable;
        _prog = index.Program;
        _pause = phonemeTable.TryGet(1, out EspeakPhoneme p) ? p : default; // code 1 is the pause/NULL phoneme
    }

    /// <summary>Runs the program for <paramref name="list"/>[<paramref name="pos"/>] (with neighbours as context), filling <paramref name="phdata"/>; <paramref name="tr"/> false skips the language-dependent stress ChangeIf (matches espeak passing <c>tr == NULL</c> for the IPA render pass).</summary>
    public void Interpret(IReadOnlyList<EspeakPhonemeListEntry> list, int pos, int control, bool tr, EspeakPhonemeData phdata)
    {
        phdata.Reset();
        EspeakPhoneme ph = At(list, pos).Ph;
        phdata.Param[EspeakProgram.ParamSetLength] = ph.StdLength;

        if (ph.Program == 0) return;

        int prog = ph.Program;
        int endFlag = 0;
        Span<int> returnStack = stackalloc int[10];
        int nReturn = 0;
        List<byte> ipa = new();

        int steps = 0;
        while (endFlag != 1)
        {
            if (++steps > 4096 || prog < 0 || prog >= _prog.Length) break; // guard against malformed jumps
            int instn = _prog[prog];
            int instn2 = (instn >> 8) & 0xf;
            switch (instn >> 12)
            {
                case 0:
                    int data = instn & 0xff;
                    if (instn2 == 0)
                    {
                        if (data == EspeakProgram.InstnReturn) endFlag = 1;
                        // INSTN_CONTINUE / unknown: fall through (no-op)
                    }
                    else if (instn2 == EspeakProgram.ParamAppendIfNextVowel)
                    {
                        if (At(list, pos + 1).Type == EspeakPhoneme.TypeVowel)
                            phdata.Param[EspeakProgram.ParamAppendPhoneme] = data;
                    }
                    else if (instn2 == EspeakProgram.ParamAddLength)
                    {
                        if ((data & 0x80) != 0) data = -(0x100 - data);
                        phdata.Param[EspeakProgram.ParamSetLength] += data;
                    }
                    else if (instn2 == EspeakProgram.ParamIpaName)
                    {
                        // espeak writes ipa_string from index 0, so a later i_IPA_NAME overwrites an earlier one
                        // (e.g. a rhotic vowel becomes non-rhotic when a separate ɹ follows).
                        ipa.Clear();
                        for (int ix = 0; ix < data && ix < 16; ix += 2)
                        {
                            prog++;
                            ipa.Add((byte)(_prog[prog] >> 8));
                            ipa.Add((byte)(_prog[prog] & 0xff));
                        }
                    }
                    else if (instn2 < EspeakProgram.NumParam)
                    {
                        phdata.Param[instn2] = data;
                        if (instn2 == EspeakProgram.ParamChangePhoneme && (control & EspeakProgram.ControlChangePhonemes) != 0)
                            endFlag = 1; // ChangePhoneme in PhonemeList mode: exit
                    }
                    break;
                case 1:
                    if (!tr) break; // language-dependent ChangeIf, skipped in render pass
                    if (instn2 < 8 && StressCondition(list, pos, instn2 & 7, 1))
                    {
                        phdata.Param[EspeakProgram.ParamChangePhoneme] = instn & 0xff;
                        endFlag = 1;
                    }
                    break;
                case 2:
                case 3:
                    {
                        bool orFlag = false, truth = true;
                        int cur = instn;
                        int condGuard = 0;
                        while (prog >= 0 && prog < _prog.Length && (cur & EspeakProgram.InstnConditionMask) == EspeakProgram.InstnConditionTag)
                        {
                            if (++condGuard > 64) break; // guard against malformed condition chains
                            bool truth2 = InterpretCondition(list, pos, prog, control);
                            prog += NumInstnWords(prog);
                            if (prog < 0 || prog >= _prog.Length) break;
                            if (_prog[prog] == EspeakProgram.InstnNot) { truth2 = !truth2; prog++; }
                            if (prog < 0 || prog >= _prog.Length) break;
                            truth = orFlag ? (truth || truth2) : (truth && truth2);
                            orFlag = (cur & EspeakProgram.InstnOr) != 0;
                            cur = _prog[prog];
                        }
                        if (!truth)
                        {
                            if ((cur & EspeakProgram.InstnJumpFalseMask) == EspeakProgram.InstnJumpFalse)
                                prog += cur & 0xff;
                            else
                            {
                                prog += NumInstnWords(prog);
                                if ((_prog[prog] & EspeakProgram.InstnElseMask) == EspeakProgram.InstnElseTag) prog++;
                            }
                        }
                        prog--;
                    }
                    break;
                case 6:
                    int jt = instn2 >> 1;
                    if (jt == 0) prog += (instn & 0xff) - 1;
                    else if (jt == 5 || jt == 6) prog += 11; // SwitchOnVowelType: skip the vowel-transition block
                    break;
                case 9:
                    prog++; // 2-word: data in prog[1]
                    if (instn2 == 1 && nReturn < returnStack.Length)
                    {
                        int target = ((instn & 0xf) << 16) + _prog[prog];
                        returnStack[nReturn++] = prog;
                        prog = target - 1;
                    }
                    break;
                case 10:
                    prog += 3; // Vowelin/Vowelout transition data
                    break;
                case 11: case 12: case 13: case 14: case 15:
                    {
                        int kind = (instn >> 12) - 11;
                        prog++;
                        if (_prog[prog + 1] != EspeakProgram.InstnContinue)
                        {
                            if (kind < 2) { endFlag = 1; if ((_prog[prog + 1] >> 12) == 0xf) endFlag = 2; }
                            else if (kind == 4) endFlag--;
                        }
                    }
                    break;
            }

            if (endFlag == 1 && nReturn > 0) { endFlag = 0; prog = returnStack[--nReturn]; }
            prog++;
        }

        phdata.IpaString = DecodeIpa(ipa);
    }

    private static string DecodeIpa(List<byte> bytes)
    {
        int n = bytes.Count;
        while (n > 0 && bytes[n - 1] == 0) n--;
        return n == 0 ? string.Empty : Encoding.UTF8.GetString(bytes.ToArray(), 0, n);
    }

    private EspeakPhonemeListEntry At(IReadOnlyList<EspeakPhonemeListEntry> list, int i)
        => i >= 0 && i < list.Count
            ? list[i]
            // Out-of-range acts as a clause boundary: SourceIx != 0 stops the boundary-seeking scans
            // (isFinalVowel / isAfterStress / CountVowelPosition / nextVowel) from running off the list end.
            : new EspeakPhonemeListEntry(1, _pause) { Type = EspeakPhoneme.TypePause, SourceIx = 1 };

    // Port of InterpretCondition. progPos points at the condition instruction word.
    private bool InterpretCondition(IReadOnlyList<EspeakPhonemeListEntry> list, int pos, int progPos, int control)
    {
        int instn = _prog[progPos] & 0xfff;
        int data = instn & 0xff;
        int instn2 = instn >> 8;

        if (instn2 < 14)
        {
            int which = instn2 % 7;
            if (which == 6) { progPos++; which = _prog[progPos]; }

            if (which == 4 && At(list, pos + 1).SourceIx != 0) return false; // nextPhW
            if (which == 5 && At(list, pos).SourceIx != 0) return false;     // prevPhW
            if (which == 6 && (At(list, pos + 1).SourceIx != 0 || At(list, pos + 2).SourceIx != 0)) return false;

            int pl = pos;
            bool checkEndType = false;
            switch (which)
            {
                case 0: case 5: pl = pos - 1; checkEndType = true; break;          // prevPh
                case 1: break;                                                      // thisPh
                case 2: case 4: pl = pos + 1; break;                                // nextPh
                case 3: case 6: pl = pos + 2; break;                                // next2Ph
                case 7:                                                             // nextVowel
                    for (int w = 1; ; w++)
                    {
                        if (At(list, pos + w).SourceIx != 0) return false;
                        if (At(list, pos + w).Type == EspeakPhoneme.TypeVowel) { pl = pos + w; break; }
                    }
                    break;
                case 8: return false; // prevVowel in word (worddata) — not tracked; conservative
                case 9:
                    for (int ix = 1; ix <= 3; ix++) if (At(list, pos + ix).SourceIx != 0) return false;
                    pl = pos + 3; break;
                case 10:
                    if (At(list, pos).SourceIx != 0 || At(list, pos - 1).SourceIx != 0) return false;
                    pl = pos - 2; checkEndType = true; break;
            }

            if ((which == 0 || which == 5) && At(list, pl).PhCode == 1) pl--; // skip deleted (NULL) phoneme
            EspeakPhoneme ph = At(list, pl).Ph;

            if (instn2 < 7)
            {
                if (_phon.TryGet(data, out EspeakPhoneme dph) && dph.Mnemonic == ph.Mnemonic) return true;
                if (checkEndType && ph.Type == EspeakPhoneme.TypeVowel) return data == ph.EndType;
                return data == ph.StartType;
            }

            data = instn & 0x1f;
            return (instn & 0xe0) switch
            {
                EspeakProgram.ConditionIsPhonemeType => ph.Type == data,
                EspeakProgram.ConditionIsPlace => ((ph.PhFlags >> 16) & 0xf) == data,
                EspeakProgram.ConditionIsPhflagSet => (ph.PhFlags & (1u << data)) != 0,
                EspeakProgram.ConditionIsOther => OtherCondition(list, pos, pl, ph, data, control),
                _ => false,
            };
        }
        return false;
    }

    private bool OtherCondition(IReadOnlyList<EspeakPhonemeListEntry> list, int pos, int pl, EspeakPhoneme ph, int data, int control)
    {
        _ = control;
        switch (data)
        {
            case EspeakProgram.StressDiminished:
            case EspeakProgram.StressUnstressed:
            case EspeakProgram.StressNotStressed:
            case EspeakProgram.StressSecondary:
            case EspeakProgram.StressPrimary:
                return StressCondition(list, pos, data, 0);
            case EspeakProgram.IsBreak:
                return ph.Type == EspeakPhoneme.TypePause || (At(list, pos).SynthFlags & EspeakProgram.SflagNextPause) != 0;
            case EspeakProgram.IsWordStart:
                return At(list, pos).SourceIx != 0;
            case EspeakProgram.IsWordEnd:
                return At(list, pos + 1).SourceIx != 0 || At(list, pos + 1).Type == EspeakPhoneme.TypePause;
            case EspeakProgram.IsAfterStress:
                if (At(list, pos).SourceIx != 0) return false;
                for (int i = pos - 1; ; i--)
                {
                    if ((At(list, i).StressLevel & 0xf) >= 4) return true;
                    if (At(list, i).SourceIx != 0) break;
                }
                return false;
            case EspeakProgram.IsNotVowel:
                return ph.Type != EspeakPhoneme.TypeVowel;
            case EspeakProgram.IsFinalVowel:
                for (int i = pl + 1; ; i++)
                {
                    if (At(list, i).SourceIx != 0) return true;
                    if (At(list, i).Type == EspeakPhoneme.TypeVowel) return false;
                }
            case EspeakProgram.IsVoiced:
                return ph.Type == EspeakPhoneme.TypeVowel || ph.Type == EspeakPhoneme.TypeLiquid || (ph.PhFlags & EspeakPhoneme.FlagVoiced) != 0;
            case EspeakProgram.IsFirstVowel:
                return CountVowelPosition(list, pl) == 1;
            case EspeakProgram.IsSecondVowel:
                return CountVowelPosition(list, pl) == 2;
            case EspeakProgram.IsTranslationGiven:
                return (At(list, pos).SynthFlags & EspeakProgram.SflagDictionary) != 0;
            default:
                return false;
        }
    }

    private bool StressCondition(IReadOnlyList<EspeakPhonemeListEntry> list, int pos, int condition, int control)
    {
        ReadOnlySpan<int> level = [1, 2, 4, 15];
        int pl;
        if (At(list, pos).Type == EspeakPhoneme.TypeVowel) pl = pos;
        else if (At(list, pos + 1).Type == EspeakPhoneme.TypeVowel) pl = pos + 1;
        else return false;

        int stress = At(list, pl).StressLevel & 0xf;
        if (condition == EspeakProgram.StressPrimary) return stress >= At(list, pl).WordStress;
        if (condition == EspeakProgram.StressSecondary) return stress > EspeakProgram.StressSecondary;
        if (condition >= 0 && condition < 4) return stress < level[condition];
        _ = control;
        return false;
    }

    private int CountVowelPosition(IReadOnlyList<EspeakPhonemeListEntry> list, int pl)
    {
        int count = 0;
        for (int i = pl; ; i--)
        {
            if (At(list, i).Type == EspeakPhoneme.TypeVowel) count++;
            if (At(list, i).SourceIx != 0) break;
        }
        return count;
    }

    // Port of NumInstnWords.
    private int NumInstnWords(int progPos)
    {
        int instn = _prog[progPos];
        int type = instn >> 12;
        ReadOnlySpan<byte> nWords = [0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 4, 0, 0, 0, 0, 0];
        int n = nWords[type];
        if (n > 0) return n;
        switch (type)
        {
            case 0:
                if (((instn & 0xf00) >> 8) == EspeakProgram.ParamIpaName) return ((instn & 0xff) + 1) / 2 + 1;
                return 1;
            case 6:
                int t2 = (instn & 0xf00) >> 9;
                return t2 is 5 or 6 ? 12 : 1;
            case 2:
            case 3:
                int c = instn & 0x0f00;
                return c is 0x600 or 0x0d00 ? 2 : 1;
            default:
                if (progPos + 2 >= _prog.Length) return 2;
                int instn2 = _prog[progPos + 2];
                if ((instn2 >> 12) == 0xf) return 4;
                return instn2 == EspeakProgram.InstnContinue ? 3 : 2;
        }
    }
}
