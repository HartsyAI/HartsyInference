using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Audio.Phonemizer.Espeak;

namespace HartsyInference.Audio.Frontends;

/// <summary>English grapheme-to-phoneme front-end for Kokoro: a port of misaki's <c>G2P</c> (hexgrad/misaki
/// <c>en.py</c>), the phonemizer Kokoro-82M was trained on. Text is split into words and misaki's sub-tokens, read
/// right to left so each word sees what follows it ("the apple" → ði, "to go" → tə), resolved through
/// <see cref="MisakiLexicon"/>, and finished with misaki v1's flap spelling (ɾ → T). misaki falls back to espeak for
/// words its dictionaries miss; here the fallback is CMUdict, then the espeak port when its data is installed, then
/// letter-to-sound rules. spaCy's tagger is not ported, so tag-dependent readings take their usual form.</summary>
public sealed partial class EnglishG2P
{
    private const string Junk = "',-._‘’/";
    private const string Puncts = ";:,.!?—…\"“”";
    private const string NonQuotePuncts = ";:,.!?—…";
    private const string LeadingPuncts = "\"“”([{«";
    private const string TrailingPuncts = ";:,.!?…\"“”)]}»";
    private static readonly HashSet<string> TitleAbbreviations = new(StringComparer.OrdinalIgnoreCase)
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs",
    };

    private static readonly HashSet<string> Elisions = new(StringComparer.Ordinal)
    {
        "'n'", "'n", "n'", "'em", "'tis", "'twas", "'cause", "'til", "'round", "'bout", "o'", "'s",
    };

    private readonly MisakiLexicon _lexicon;
    private readonly Dictionary<string, string[]>? _cmudict;
    private readonly EspeakPhonemizer? _espeak;

    /// <summary>Builds the front-end over <paramref name="lexicon"/>, with optional OOV fallbacks: a CMUdict stream
    /// (lines <c>"word  PH PH ..."</c>, first pronunciation wins) and an en-us espeak phonemizer.</summary>
    public EnglishG2P(MisakiLexicon lexicon, Stream? cmudict = null, EspeakPhonemizer? espeak = null)
    {
        ArgumentNullException.ThrowIfNull(lexicon);
        _lexicon = lexicon;
        _cmudict = cmudict is null ? null : LoadCmudict(cmudict);
        _espeak = espeak;
    }

    /// <summary>As the stream constructor, reading CMUdict from <paramref name="cmudictPath"/> when given.</summary>
    public EnglishG2P(MisakiLexicon lexicon, string? cmudictPath, EspeakPhonemizer? espeak = null)
        : this(lexicon, OpenOptional(cmudictPath), espeak)
    {
    }

    /// <summary>Converts free text to the phoneme string Kokoro consumes: words separated by spaces, punctuation
    /// attached as written.</summary>
    public string ToIpa(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        List<Word> words = Tokenize(text.Replace('‘', '\'').Replace('’', '\'').Trim());
        bool mixedCase = text.Any(char.IsLower);
        TokenContext ctx = new(null);
        bool futureTo = false;
        for (int i = words.Count - 1; i >= 0; i--)
        {
            Word w = words[i];
            if (w.Tokens.Count == 1)
            {
                Token tk = w.Tokens[0];
                tk.Ps ??= _lexicon.Phonemize(tk.Text, ctx, tk.Currency, isHead: true, futureTo, mixedCase, ClauseStart(words, i))
                    ?? Fallback(tk.Text);
                ctx = NextContext(ctx, tk.Ps);
            }
            else
            {
                ctx = ResolveGroup(w.Tokens, ctx, futureTo, mixedCase, ClauseStart(words, i));
            }
            futureTo = w.Tokens.Count == 1 && w.Tokens[0].Text is "to" or "To" or "TO";
        }
        StringBuilder sb = new(text.Length * 2);
        foreach (Word w in words)
        {
            sb.Append(w.Tokens.Count == 1 ? w.Tokens[0].Ps : MergeGroup(w.Tokens));
            if (w.Space) sb.Append(' ');
        }
        // misaki v1, which Kokoro-82M v1.0 was trained on, spells the flap and glottal stop as T and t.
        return sb.ToString().Trim().Replace('ɾ', 'T').Replace('ʔ', 't');
    }

    /// <summary>Whether word <paramref name="i"/> opens a clause: nothing but opening quotes and brackets before it
    /// since the last pause punctuation.</summary>
    private static bool ClauseStart(List<Word> words, int i)
    {
        for (int j = i - 1; j >= 0; j--)
        {
            string? ps = words[j].Tokens.Count == 1 ? words[j].Tokens[0].Ps : null;
            if (ps is null || words[j].Tokens[0].Text.Any(char.IsLetterOrDigit)) return false;
            if (ps.Length > 0 && NonQuotePuncts.Contains(ps[^1])) return true;
        }
        return true;
    }

    /// <summary>misaki's span search over a word's sub-tokens: the longest known run ending at the right edge wins,
    /// then the rest is searched the same way; an unreadable non-junk piece sends the whole word to the fallback.</summary>
    private TokenContext ResolveGroup(List<Token> w, TokenContext ctx, bool futureTo, bool mixedCase, bool clauseStart)
    {
        int left = 0, right = w.Count;
        bool fallback = false;
        while (left < right)
        {
            string? ps = null;
            if (!w.Skip(left).Take(right - left).Any(static t => t.Ps is not null))
            {
                string text = string.Concat(w.Skip(left).Take(right - left).Select(static t => t.Text));
                char? currency = w.Skip(left).Take(right - left).Select(static t => t.Currency).Max();
                ps = _lexicon.Phonemize(text, ctx, currency, w[left].IsHead, futureTo, mixedCase, clauseStart && left == 0);
            }
            if (ps is not null)
            {
                w[left].Ps = ps;
                for (int j = left + 1; j < right; j++) w[j].Ps = "";
                ctx = NextContext(ctx, ps);
                right = left;
                left = 0;
            }
            else if (left + 1 < right)
            {
                left++;
            }
            else
            {
                right--;
                Token tk = w[right];
                if (tk.Ps is null)
                {
                    if (tk.Text.All(static c => Junk.Contains(c)))
                    {
                        tk.Ps = "";
                    }
                    else
                    {
                        fallback = true;
                        break;
                    }
                }
                left = 0;
            }
        }
        if (fallback)
        {
            w[0].Ps = Fallback(string.Concat(w.Select(static t => t.Text)));
            for (int j = 1; j < w.Count; j++) w[j].Ps = "";
        }
        else
        {
            ResolveStress(w);
        }
        return ctx;
    }

    /// <summary>misaki <c>resolve_tokens</c>: a word written as letters and digits/symbols keeps its pieces apart;
    /// a compound of letter pieces keeps primary stress on its heavier half.</summary>
    private static void ResolveStress(List<Token> tokens)
    {
        string text = string.Concat(tokens.Select(static t => t.Text));
        bool prespace = text.Contains(' ') || text.Contains('/') || text.Where(static c => !Junk.Contains(c))
            .Select(static c => char.IsLetter(c) ? 0 : char.IsDigit(c) ? 1 : 2).Distinct().Count() > 1;
        for (int i = 0; i < tokens.Count; i++)
        {
            Token tk = tokens[i];
            if (tk.Ps is null)
            {
                if (i == tokens.Count - 1 && tk.Text.Length == 1 && NonQuotePuncts.Contains(tk.Text[0])) tk.Ps = tk.Text;
                else if (tk.Text.All(static c => Junk.Contains(c))) tk.Ps = "";
            }
            else if (i > 0)
            {
                tk.Prespace = prespace;
            }
        }
        if (prespace) return;
        List<(bool Primary, int Weight, int Index)> indices = [];
        for (int i = 0; i < tokens.Count; i++)
        {
            string? ps = tokens[i].Ps;
            if (!string.IsNullOrEmpty(ps))
                indices.Add((ps.Contains(MisakiLexicon.PrimaryStress), MisakiLexicon.StressWeight(ps), i));
        }
        if (indices.Count == 2 && tokens[indices[0].Index].Text.Length == 1)
        {
            Token second = tokens[indices[1].Index];
            second.Ps = MisakiLexicon.ApplyStress(second.Ps, -0.5);
            return;
        }
        if (indices.Count < 2 || indices.Count(static x => x.Primary) <= (indices.Count + 1) / 2) return;
        foreach ((bool _, int _, int index) in indices.OrderBy(static x => x.Primary).ThenBy(static x => x.Weight)
            .ThenBy(static x => x.Index).Take(indices.Count / 2))
        {
            tokens[index].Ps = MisakiLexicon.ApplyStress(tokens[index].Ps, -0.5);
        }
    }

    private static string MergeGroup(List<Token> tokens)
    {
        StringBuilder sb = new();
        foreach (Token tk in tokens)
        {
            if (tk.Prespace && sb.Length > 0 && !char.IsWhiteSpace(sb[^1]) && !string.IsNullOrEmpty(tk.Ps)) sb.Append(' ');
            sb.Append(tk.Ps);
        }
        return sb.ToString();
    }

    /// <summary>misaki <c>token_context</c>: the first vowel, consonant or pause in <paramref name="ps"/> decides what
    /// the word before it sees; an empty reading leaves the context unchanged.</summary>
    private static TokenContext NextContext(TokenContext ctx, string? ps)
    {
        if (string.IsNullOrEmpty(ps)) return ctx;
        foreach (char c in ps)
        {
            if (NonQuotePuncts.Contains(c)) return new TokenContext(null);
            if (MisakiLexicon.Vowels.Contains(c)) return new TokenContext(true);
            if (MisakiLexicon.Consonants.Contains(c)) return new TokenContext(false);
        }
        return ctx;
    }

    /// <summary>Reading for a word misaki's dictionaries miss, sent there whole as misaki sends it to espeak and read
    /// the way espeak reads it: hyphen- and underscore-joined pieces run together, camel-case pieces apart, digits
    /// as numbers, pause punctuation kept. Each letter piece takes CMUdict first (a real pronunciation), then espeak
    /// (misaki's own fallback; the port is not yet exact, so it ranks behind the dictionary), then letter rules.</summary>
    private string Fallback(string text)
    {
        StringBuilder sb = new();
        bool joinNext = false;
        foreach (Match m in SubtokenRegex().Matches(text))
        {
            string piece = m.Value;
            string? ps;
            if (piece.Any(char.IsLetter)) ps = FallbackPiece(piece);
            else if (piece.Any(char.IsAsciiDigit)) ps = _lexicon.Phonemize(piece, new TokenContext(null), null, true, false, false, false);
            else if (piece.Length == 1 && NonQuotePuncts.Contains(piece[0]))
            {
                sb.Append(piece);
                joinNext = true;
                continue;
            }
            else
            {
                joinNext |= piece.All(static c => c is '-' or '_');
                continue;
            }
            if (string.IsNullOrEmpty(ps)) continue;
            if (sb.Length > 0 && !joinNext) sb.Append(' ');
            sb.Append(ps);
            joinNext = false;
        }
        return sb.ToString();
    }

    private string FallbackPiece(string word)
    {
        string lower = word.ToLowerInvariant();
        if (_cmudict is not null && _cmudict.TryGetValue(lower, out string[]? phones))
            return ArpabetToIpa.ConvertWord(phones);
        if (_espeak is not null)
        {
            string ipa = EspeakToMisaki.Convert(_espeak.PhonemizeToIpa(word, "en-us"));
            if (ipa.Length > 0) return ipa;
        }
        // A piece with no vowel letter ("kg") is an abbreviation; espeak spells those out.
        if (lower.AsSpan().IndexOfAny("aeiouy") < 0) return _lexicon.Spell(word) ?? "";
        return string.Join(' ', LetterRunRegex().Matches(lower).Select(static m => LetterToSound(m.Value)));
    }

    private static List<Word> Tokenize(string text)
    {
        List<Word> words = [];
        MatchCollection chunks = ChunkRegex().Matches(text);
        for (int c = 0; c < chunks.Count; c++)
        {
            int firstWord = words.Count;
            AddChunk(chunks[c].Value, words);
            if (words.Count > firstWord && c < chunks.Count - 1) words[^1].Space = true;
        }
        return words;
    }

    /// <summary>Splits one whitespace-delimited chunk the way spaCy's tokenizer and misaki's <c>subtokenize</c> do:
    /// leading and trailing punctuation become their own tokens, the rest one word of sub-tokens.</summary>
    private static void AddChunk(string chunk, List<Word> words)
    {
        if (chunk.All(static c => c is '-' or '–' or '—'))
        {
            words.Add(Word.Fixed(chunk, "—"));
            return;
        }
        int start = 0, end = chunk.Length;
        char? currency = null;
        while (start < end && (LeadingPuncts.Contains(chunk[start]) || IsOpeningQuote(chunk, start, end)))
        {
            words.Add(Word.Fixed(chunk[start].ToString(), PunctPhoneme(chunk[start], opening: true)));
            start++;
        }
        if (start + 1 < end && chunk[start] is '$' or '£' or '€' && char.IsAsciiDigit(chunk[start + 1]))
        {
            currency = chunk[start];
            start++;
        }
        List<Word> trailing = [];
        while (end > start && (TrailingPuncts.Contains(chunk[end - 1]) || IsClosingQuote(chunk, start, end))
            && !KeepsPeriod(chunk[start..end]))
        {
            // A run of periods is one token ("..."), as spaCy keeps it.
            int runStart = end - 1;
            while (chunk[end - 1] == '.' && runStart > start && chunk[runStart - 1] == '.') runStart--;
            string punct = chunk[runStart..end];
            trailing.Insert(0, Word.Fixed(punct, string.Concat(punct.Select(static p => PunctPhoneme(p, opening: false)))));
            end = runStart;
        }
        // Punctuation spaCy splits off inside a chunk ("is--\"Birds", "joined):--") becomes its own token too.
        int piece = start;
        foreach (Match infix in InfixRegex().Matches(chunk[start..end]))
        {
            int at = start + infix.Index;
            AddSubtokens(chunk[piece..at], words, ref currency);
            char c = chunk[at];
            words.Add(Word.Fixed(c.ToString(), PunctPhoneme(c, opening: at + 1 < end && char.IsLetterOrDigit(chunk[at + 1]))));
            piece = at + 1;
        }
        AddSubtokens(chunk[piece..end], words, ref currency);
        words.AddRange(trailing);
    }

    private static void AddSubtokens(string text, List<Word> words, ref char? currency)
    {
        List<Token> tokens = [];
        foreach (Match m in SubtokenRegex().Matches(text))
            tokens.Add(new Token(m.Value) { IsHead = tokens.Count == 0 });
        if (tokens.Count == 0) return;
        tokens[^1].Currency = currency;
        currency = null;
        words.Add(new Word(tokens));
    }

    /// <summary>A leading apostrophe that opens a quotation (spaCy's <c>``</c>), not an elision like "'em".</summary>
    private static bool IsOpeningQuote(string chunk, int start, int end) =>
        chunk[start] == '\'' && start + 1 < end && char.IsLetter(chunk[start + 1])
        && !Elisions.Contains(chunk[start..end].TrimEnd(TrailingPuncts.ToCharArray()).ToLowerInvariant());

    /// <summary>A trailing apostrophe that closes a quotation: after punctuation, or after a word that is neither a
    /// plural possessive ("dogs'") nor a dropped g ("nothin'").</summary>
    private static bool IsClosingQuote(string chunk, int start, int end)
    {
        if (chunk[end - 1] != '\'' || end - start < 2) return false;
        char before = chunk[end - 2];
        if (!char.IsLetter(before)) return before != '\'';
        string word = chunk[start..end].ToLowerInvariant();
        return before is not ('s' or 'S') && !word.EndsWith("in'", StringComparison.Ordinal) && !Elisions.Contains(word);
    }

    /// <summary>Whether a trailing period belongs to the word: a title ("Dr.") or a dotted initialism ("U.S.").</summary>
    private static bool KeepsPeriod(string word)
    {
        if (!word.EndsWith('.')) return false;
        string bare = word.TrimEnd('.');
        return TitleAbbreviations.Contains(bare) || (bare.Contains('.') && bare.Replace(".", "").All(char.IsLetter)
            && bare.Split('.').All(static p => p.Length is > 0 and < 3));
    }

    private static string PunctPhoneme(char c, bool opening) => c switch
    {
        '"' or '\'' => opening ? "“" : "”",
        '«' => "“",
        '»' => "”",
        '(' or '[' or '{' => "(",
        ')' or ']' or '}' => ")",
        _ => Puncts.Contains(c) ? c.ToString() : "",
    };

    private static Dictionary<string, string[]> LoadCmudict(Stream stream)
    {
        Dictionary<string, string[]> dict = new(StringComparer.Ordinal);
        using StreamReader reader = new(stream);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (line.Length == 0 || line.StartsWith(";;;", StringComparison.Ordinal)) continue;
            int comment = line.IndexOf('#');
            string[] parts = (comment >= 0 ? line[..comment] : line).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;
            string word = parts[0];
            int paren = word.IndexOf('(');
            if (paren >= 0) word = word[..paren];
            dict.TryAdd(word.ToLowerInvariant(), parts[1..]);
        }
        return dict;
    }

    private static Stream? OpenOptional(string? path)
    {
        if (path is null) return null;
        if (!File.Exists(path)) throw new FileNotFoundException($"CMUdict not found: {path}", path);
        return File.OpenRead(path);
    }

    private static readonly (string Gr, string Ipa)[] Digraphs =
    [
        ("tch", "ʧ"), ("sch", "sk"),
        ("ch", "ʧ"), ("sh", "ʃ"), ("th", "θ"), ("ph", "f"), ("wh", "w"), ("ck", "k"), ("ng", "ŋ"), ("qu", "kw"),
        ("ee", "i"), ("ea", "i"), ("oo", "u"), ("oa", "O"), ("ai", "A"), ("ay", "A"), ("ou", "W"),
        ("ow", "W"), ("oy", "Y"), ("oi", "Y"), ("ar", "ɑɹ"), ("or", "ɔɹ"), ("er", "əɹ"), ("ir", "ɜɹ"), ("ur", "ɜɹ"),
    ];

    private static readonly Dictionary<char, string> Letters = new()
    {
        ['a'] = "æ", ['b'] = "b", ['c'] = "k", ['d'] = "d", ['e'] = "ɛ", ['f'] = "f", ['g'] = "ɡ", ['h'] = "h",
        ['i'] = "ɪ", ['j'] = "ʤ", ['k'] = "k", ['l'] = "l", ['m'] = "m", ['n'] = "n", ['o'] = "ɑ", ['p'] = "p",
        ['q'] = "k", ['r'] = "ɹ", ['s'] = "s", ['t'] = "t", ['u'] = "ʌ", ['v'] = "v", ['w'] = "w", ['x'] = "ks",
        ['y'] = "j", ['z'] = "z",
    };

    /// <summary>Last-resort letter-to-sound rules for a lowercase letter run, in misaki's symbol set (one token per
    /// diphthong and affricate) with primary stress on the first vowel. Approximate.</summary>
    public static string LetterToSound(string word)
    {
        StringBuilder sb = new();
        int i = 0;
        while (i < word.Length)
        {
            bool matched = false;
            foreach ((string gr, string ipa) in Digraphs)
            {
                if (i + gr.Length <= word.Length && word.AsSpan(i, gr.Length).SequenceEqual(gr))
                {
                    sb.Append(ipa);
                    i += gr.Length;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
            // A doubled consonant letter is one sound ("happy").
            if (i > 0 && word[i] == word[i - 1] && !"aeiou".Contains(word[i]))
            {
                i++;
                continue;
            }
            // y is a consonant only before a vowel ("yes"); elsewhere it is the vowel of "happy" and "gym".
            if (word[i] == 'y' && (i + 1 >= word.Length || !"aeiou".Contains(word[i + 1])))
                sb.Append(i + 1 >= word.Length ? "i" : "ɪ");
            else if (Letters.TryGetValue(word[i], out string? letter))
                sb.Append(letter);
            i++;
        }
        return MisakiLexicon.ApplyStress(sb.ToString(), 2)!;
    }

    [GeneratedRegex(@"\S+")]
    private static partial Regex ChunkRegex();

    [GeneratedRegex(@"[()\[\]{};!?""“”…]|(?<!\d):|:(?!\d)")]
    private static partial Regex InfixRegex();

    [GeneratedRegex(@"[a-z]+")]
    private static partial Regex LetterRunRegex();

    // misaki's SUBTOKEN_REGEX: camelCase and acronym boundaries, numbers with separators, hyphen runs, and words with
    // inner apostrophes; characters no alternative matches are dropped, as Python's findall drops them.
    [GeneratedRegex(@"^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$")]
    private static partial Regex SubtokenRegex();

    /// <summary>One sub-token and its reading.</summary>
    private sealed class Token(string text)
    {
        public string Text { get; } = text;
        public string? Ps { get; set; }
        public bool IsHead { get; init; } = true;
        public char? Currency { get; set; }
        public bool Prespace { get; set; }
    }

    /// <summary>A whitespace-or-punctuation-delimited word: one token, or sub-tokens written without spaces.</summary>
    private sealed class Word(List<Token> tokens)
    {
        public List<Token> Tokens { get; } = tokens;
        public bool Space { get; set; }

        public static Word Fixed(string text, string ps) => new([new Token(text) { Ps = ps }]);
    }
}
