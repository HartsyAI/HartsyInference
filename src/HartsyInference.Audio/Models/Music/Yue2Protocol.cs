namespace HartsyInference.Audio.Models.Music;

/// <summary>The chain-of-thought mode YuE2 plans a song in. The choice changes the instruction sentence, the token
/// prefix, the CFG negative branch and the default guidance scale, so it is a protocol-level switch rather than a
/// sampling preference.</summary>
public enum Yue2Cot
{
    /// <summary>No symbolic planning — straight to codec tokens. The only mode that guides (cfg 1.01) by default,
    /// and the only one that keeps the release's BF16 logit path (<c>legacy_off</c>).</summary>
    Off,

    /// <summary>Plan a melody-only ABC transcription with no chord symbols. Recommended for covers.</summary>
    Melody,

    /// <summary>Plan a chord-annotated ABC transcription. The release default.</summary>
    Full,
}

/// <summary>Which of the two autoregressive passes a logit batch belongs to. They differ in allowed vocabulary,
/// end token and sampler defaults.</summary>
public enum Yue2Phase
{
    /// <summary>Symbolic score planning; emits ordinary text ids and ends at <see cref="Yue2Protocol.AbcEnd"/>.</summary>
    Abc,

    /// <summary>Semantic codec generation; emits only codec ids and ends at <see cref="Yue2Protocol.MusicEnd"/>.</summary>
    Semantic,
}

/// <summary>YuE2's wire protocol: the special token ids, instruction strings and prefix layout that the released
/// <c>yue2-native-v1</c> checkpoint was trained against. Reproduced from the official <c>yue2_infer</c> package's
/// <c>protocol.py</c>; every constant here is a checkpoint fact, not a tunable.</summary>
public static class Yue2Protocol
{
    /// <summary>End-of-document; opens every prefix.</summary>
    public const int Eod = 151_643;

    /// <summary>Opens the ABC score span.</summary>
    public const int AbcStart = 151_847;

    /// <summary>Closes the ABC score span; the ABC phase's end token.</summary>
    public const int AbcEnd = 151_848;

    /// <summary>Opens the codec span.</summary>
    public const int MusicStart = 151_851;

    /// <summary>Closes the codec span; the semantic phase's end token.</summary>
    public const int MusicEnd = 151_852;

    /// <summary>First codec token id. A codec value <c>v</c> is the token <c>CodecOffset + v</c>.</summary>
    public const int CodecOffset = 151_853;

    /// <summary>Codec vocabulary size; the allowed span is <c>[CodecOffset, CodecOffset + CodecSize)</c>.</summary>
    public const int CodecSize = 32_768;

    /// <summary>Full vocabulary of the AR LM.</summary>
    public const int VocabSize = 184_704;

    /// <summary>Maximum sequence the checkpoint was trained to attend over. Prefix plus generation budget must fit.</summary>
    public const int Context = 24_576;

    /// <summary>The smallest contiguous id window a phase's sampler can ever draw from, as <c>(first id, count)</c>.
    /// <see cref="Yue2LogitProcessor"/> masks everything outside the phase's span to -inf before sampling, so the
    /// semantic pass projects, copies back and scans 184,704 logits to choose among 32,769 — 5.6x the bandwidth it
    /// can use. Sampling works in window coordinates (<c>id - Base</c>) and the drawn index is offset back.</summary>
    /// <remarks>The semantic window is exact: <see cref="MusicEnd"/> sits immediately below <see cref="CodecOffset"/>,
    /// so the end token and the whole codec span are one contiguous run. The ABC window is the smallest contiguous
    /// run <i>covering</i> its span — <c>[0, Eod)</c> plus the far-away <see cref="AbcEnd"/> — so it still holds ids
    /// the mask rejects, and only shortens the read-back, not the projection.</remarks>
    public static (int Base, int Count) Window(Yue2Phase phase) => phase == Yue2Phase.Abc
        ? (0, AbcEnd + 1)
        : (MusicEnd, CodecOffset + CodecSize - MusicEnd);

    /// <summary>Semantic tokens (and latent frames) per second of audio.</summary>
    public const int FramesPerSecond = 25;

    /// <summary>Song length a request gets when it does not ask for one: the release's own 9,000-token budget.</summary>
    public const double DefaultDurationSeconds = 360.0;

    /// <summary>Ceiling a request may ask for. Above the release's own budget because nothing in the checkpoint
    /// forbids it — 900 seconds is 22,500 semantic tokens inside a 24,576-token context — but the real limit is
    /// lower and prompt-dependent, which is what <see cref="BudgetForPrefix"/> settles per request.</summary>
    public const double MaxDurationSeconds = 900.0;

    /// <summary>The instruction sentence prepended to every prompt. The model was trained on these exact strings —
    /// rewording them silently changes behaviour.</summary>
    public static string Instruction(Yue2Cot cot) => cot switch
    {
        Yue2Cot.Off => "Generate music with codec tokens from the given conditions.",
        Yue2Cot.Melody => "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
        Yue2Cot.Full => "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
        _ => throw new ArgumentOutOfRangeException(nameof(cot)),
    };

    /// <summary>The prompt body the AR LM reads: instruction, then style tags, then lyrics.</summary>
    public static string PromptText(Yue2Cot cot, string style, string lyrics)
        => $"{Instruction(cot)}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n";

    /// <summary>Guidance scale when the caller has no opinion: the release guides only in <see cref="Yue2Cot.Off"/>,
    /// and only barely.</summary>
    public static float DefaultCfgScale(Yue2Cot cot) => cot == Yue2Cot.Off ? 1.01f : 1.0f;

    /// <summary>The positive-branch prefix. With <paramref name="abcIds"/> null the result opens an ABC span for the
    /// planner to fill; with ABC ids in hand it closes the span and opens the codec span, which is what the semantic
    /// pass consumes. <see cref="Yue2Cot.Off"/> emits an empty ABC span either way.</summary>
    public static int[] TokenPrefix(Yue2Cot cot, IReadOnlyList<int> promptIds, IReadOnlyList<int>? abcIds)
    {
        ArgumentNullException.ThrowIfNull(promptIds);
        List<int> ids = new(promptIds.Count + 8) { Eod };
        ids.AddRange(promptIds);
        if (cot == Yue2Cot.Off)
        {
            ids.Add(AbcStart);
            ids.Add(AbcEnd);
            ids.Add(MusicStart);
            return [.. ids];
        }
        ids.Add(AbcStart);
        if (abcIds is null) return [.. ids];
        ValidateAbcIds(abcIds);
        ids.AddRange(abcIds);
        ids.Add(AbcEnd);
        ids.Add(MusicStart);
        return [.. ids];
    }

    /// <summary>The CFG negative branch: the instruction alone, with style and lyrics dropped but the <b>exact</b>
    /// positive-branch ABC ids retained. Re-encoding the decoded score here instead of reusing the ids it was
    /// sampled as would desynchronise the two branches.</summary>
    public static int[] NegativePrefix(Yue2Cot cot, IReadOnlyList<int> instructionIds, IReadOnlyList<int>? abcIds)
    {
        ArgumentNullException.ThrowIfNull(instructionIds);
        List<int> ids = new(instructionIds.Count + 8) { Eod };
        ids.AddRange(instructionIds);
        if (cot == Yue2Cot.Off)
        {
            ids.Add(MusicStart);
            return [.. ids];
        }
        if (abcIds is null)
            throw new ArgumentNullException(nameof(abcIds), "Symbolic CFG must retain the exact positive-branch ABC ids.");
        ValidateAbcIds(abcIds);
        ids.Add(AbcStart);
        ids.AddRange(abcIds);
        ids.Add(AbcEnd);
        ids.Add(MusicStart);
        return [.. ids];
    }

    /// <summary>Splits <paramref name="frames"/> latent frames into the chunks one AR prefill can condition. Each
    /// chunk costs its own prefix plus two tokens per frame (the AR codec token and the NAR latent position), so the
    /// budget is halved. With a typical prefix this is ~12k frames, comfortably past the 360-second maximum.</summary>
    public static (int Start, int End)[] ChunkRanges(int frames, int prefixTokens, int context = Context)
    {
        int size = Math.Min((context - prefixTokens - 3) / 2, Context);
        if (frames < 1 || size < 1)
            throw new ArgumentOutOfRangeException(nameof(frames), "Empty codec or an oversized prefix leaves no acoustic context.");
        List<(int, int)> ranges = [];
        for (int start = 0; start < frames; start += size) ranges.Add((start, Math.Min(start + size, frames)));
        return [.. ranges];
    }

    /// <summary>The semantic budget for a requested duration, in tokens.</summary>
    public static int TokensForSeconds(double seconds)
        => Math.Max(1, (int)Math.Round(Math.Clamp(seconds, 0, MaxDurationSeconds) * FramesPerSecond, MidpointRounding.AwayFromZero));

    /// <summary>How much of <paramref name="requestedTokens"/> still fits the context alongside the prefix. Under
    /// guidance the negative branch is prefilled into its own cache and has its own length, so the longer of the two
    /// binds. Zero means the prompt leaves no room for music at all.</summary>
    /// <remarks>The release refuses this case outright — its sampler raises "no implicit truncation" rather than
    /// shortening anything. We diverge because the knob we expose is a duration <i>ceiling</i> rather than a token
    /// count: a prompt that leaves less room than asked for shortens the song instead of failing it after the score
    /// has already been planned. A request that already fits is unaffected, token for token.</remarks>
    public static int BudgetForPrefix(int requestedTokens, int prefixTokens, int negativeTokens = 0, int context = Context)
        => Math.Max(0, Math.Min(requestedTokens, context - Math.Max(prefixTokens, negativeTokens)));

    private static void ValidateAbcIds(IReadOnlyList<int> abcIds)
    {
        for (int i = 0; i < abcIds.Count; i++)
        {
            if ((uint)abcIds[i] >= Eod)
                throw new ArgumentOutOfRangeException(nameof(abcIds), "ABC ids must stay inside the ordinary text vocabulary.");
        }
    }
}
