using System.IO;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>YuE "mm" tokenizer — the LLaMA SentencePiece model extended with YuE's structural / audio
/// special tokens, loaded from the model's <c>tokenizer.model</c> (m-a-p/xcodec_mini_infer,
/// <c>mm_tokenizer_v0.2_hf/</c>). Builds the Stage-1 lyrics→token prompt for <c>YueStage1Lm</c>.
///
/// <para>IDs are resolved from the loaded model, not hardcoded — so they're correct for whatever
/// checkpoint the user supplies. The Stage-1 prompt's special-token tail follows YuE's
/// <c>infer.py</c> (CoT) / docs/Research/YUE_ARCHITECTURE.md §7; verify token parity (bit-exact first
/// ~100 tokens) against the reference per the M5 checklist before treating output as faithful.</para></summary>
public sealed class YueTokenizer : IDisposable
{
    // The YuE structural / audio markers are SentencePiece CONTROL symbols spelled with ANGLE brackets
    // (<SOA>, <EOA>, <stage_1>, …) — verified against the real mm_tokenizer_v0.2 tokenizer.model. They are
    // NOT encodable via EncodeToIds (control symbols byte-fall-back, e.g. "<SOA>" -> [529,6156,29909,29958]),
    // and Microsoft.ML.Tokenizers does not surface them through SpecialTokens. Their ids are a fixed, stable
    // part of the mm_tokenizer_v0.2 vocab; we pin them and confirm each via a Decode(id) round-trip at load.
    /// <summary>Start-of-audio: hands off from the text prompt to cb0 generation.</summary>
    public const string SoaPiece = "<SOA>";
    /// <summary>End-of-audio: stops cb0 generation.</summary>
    public const string EoaPiece = "<EOA>";
    public const string Stage1Piece = "<stage_1>";
    /// <summary>Stage-2 handoff marker (Stage-2 prompt: [SOA][stage_1] + cb0 + [stage_2]).</summary>
    public const string Stage2Piece = "<stage_2>";
    /// <summary>xcodec codec-type separator — CodecManipulator("xcodec").sep_ids. Ends the Stage-1 prompt
    /// (… &lt;SOA&gt; &lt;xcodec&gt;) to prime cb0 generation.</summary>
    public const string XcodecPiece = "<xcodec>";

    // Fixed mm_tokenizer_v0.2 ids for the YuE control symbols (verified by Decode round-trip below).
    private const int SoaId = 32_001;
    private const int EoaId = 32_002;
    private const int Stage1Id = 32_013;
    private const int XcodecId = 32_016;
    private const int Stage2Id = 32_017;

    private readonly SentencePieceTokenizer _sp;
    private int[]? _startOfSegmentIds;
    private int[]? _endOfSegmentIds;

    public int Soa { get; }
    public int Eoa { get; }
    public int Stage1 { get; }
    public int Stage2 { get; }
    public int Xcodec { get; }

    public YueTokenizer(string tokenizerModelPath)
    {
        if (!File.Exists(tokenizerModelPath))
        {
            throw new FileNotFoundException($"YuE tokenizer.model not found: {tokenizerModelPath}", tokenizerModelPath);
        }
        using FileStream fs = File.OpenRead(tokenizerModelPath);
        _sp = SentencePieceTokenizer.Create(fs, addBeginningOfSentence: false, addEndOfSentence: false)
            ?? throw new InvalidOperationException($"Failed to load YuE SentencePiece tokenizer from '{tokenizerModelPath}'.");
        Soa = ResolveControl(SoaId, SoaPiece);
        Eoa = ResolveControl(EoaId, EoaPiece);
        Stage1 = ResolveControl(Stage1Id, Stage1Piece);
        Stage2 = ResolveControl(Stage2Id, Stage2Piece);
        Xcodec = ResolveControl(XcodecId, XcodecPiece);
    }

    /// <summary>Confirms a pinned control-symbol id maps back to its expected angle-bracket piece via
    /// <c>Decode</c> (the reliable piece-&gt;id direction for SP control symbols). A mismatch means the file
    /// isn't the YuE mm tokenizer — fail with guidance.</summary>
    private int ResolveControl(int id, string expectedPiece)
    {
        string decoded = _sp.Decode([id]);
        if (decoded != expectedPiece)
        {
            throw new InvalidOperationException(
                $"YuE tokenizer.model id {id} decodes to '{decoded}', expected '{expectedPiece}'. "
                + "Supply the YuE mm tokenizer (mm_tokenizer_v0.2_hf/tokenizer.model from m-a-p/xcodec_mini_infer).");
        }
        return id;
    }

    /// <summary>Splits lyrics into YuE's structured segments (one per <c>[verse]/[chorus]/…</c> tag),
    /// mirroring infer.py's <c>split_lyrics</c>: each becomes <c>"[label]\n{text}\n\n"</c>.</summary>
    public static List<string> SplitLyrics(string? lyrics)
    {
        List<string> segs = [];
        foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(
                     lyrics ?? "", @"\[(\w+)\](.*?)(?=\[|\Z)", System.Text.RegularExpressions.RegexOptions.Singleline))
            segs.Add($"[{m.Groups[1].Value}]\n{m.Groups[2].Value.Trim()}\n\n");
        return segs;
    }

    /// <summary>The Stage-1 "head" prompt (instruction + genre + full structured lyrics) — infer.py <c>prompt_texts[0]</c>.
    /// Testable; the structured lyrics join each <c>[label]</c> segment with a blank line, matching <c>split_lyrics</c>.</summary>
    public static string BuildStage1PromptText(string? genre, string? lyrics)
    {
        List<string> segs = SplitLyrics(lyrics);
        string full = segs.Count > 0 ? string.Join("\n", segs) : (lyrics ?? "").Trim();
        return $"Generate music from the given lyrics segment by segment.\n[Genre] {(genre ?? "").Trim()}\n{full}";
    }

    /// <summary>Encodes YuE's Stage-1 **segment-0** prompt EXACTLY as infer.py builds it:
    /// <c>tokenize(head) + tokenize("[start_of_segment]") + tokenize(section[0]) + [SOA] + [&lt;xcodec&gt; sep]</c>.
    /// The head restates the genre + all structured lyrics as context; the section restates segment-0's lyrics
    /// (this <c>[start_of_segment]</c> + section restatement is what primes coherent cb0 generation — omitting it
    /// yields gibberish). Later segments are appended by the pipeline's segment loop (<c>[end_of_segment]</c> +
    /// <c>[start_of_segment]</c> + section + [SOA] + sep). <c>&lt;stage_1&gt;</c> is Stage-2 only.</summary>
    public int[] EncodeStage1Prompt(string? genre, string? lyrics)
    {
        List<string> segs = SplitLyrics(lyrics);
        string section = segs.Count > 0 ? segs[0] : (lyrics ?? "").Trim();
        List<int> ids = [.. _sp.EncodeToIds(BuildStage1PromptText(genre, lyrics))];
        ids.AddRange(_sp.EncodeToIds("[start_of_segment]"));
        ids.AddRange(_sp.EncodeToIds(section));
        ids.Add(Soa);
        ids.Add(Xcodec);
        return [.. ids];
    }

    /// <summary>Builds a subsequent-segment continuation prompt (infer.py, i&gt;0):
    /// <c>tokenize("[end_of_segment][start_of_segment]") + tokenize(section) + [SOA] + [&lt;xcodec&gt; sep]</c>.
    /// The pipeline prepends the running generated context and appends this per segment.</summary>
    public int[] EncodeStage1SegmentContinuation(string? sectionLyrics)
    {
        List<int> ids = [.. _sp.EncodeToIds("[end_of_segment][start_of_segment]")];
        ids.AddRange(_sp.EncodeToIds(sectionLyrics ?? ""));
        ids.Add(Soa);
        ids.Add(Xcodec);
        return [.. ids];
    }

    /// <summary>Cached ids for the "[start_of_segment]" structural marker (infer.py's <c>start_of_segment</c>).</summary>
    public IReadOnlyList<int> StartOfSegmentIds => _startOfSegmentIds ??= [.. _sp.EncodeToIds("[start_of_segment]")];

    /// <summary>Cached ids for the "[end_of_segment]" structural marker (infer.py's <c>end_of_segment</c>).</summary>
    public IReadOnlyList<int> EndOfSegmentIds => _endOfSegmentIds ??= [.. _sp.EncodeToIds("[end_of_segment]")];

    /// <summary>Stage-1 "head" prompt ids (instruction + genre + full structured lyrics) — infer.py <c>prompt_texts[0]</c>.</summary>
    public int[] EncodeStage1Head(string? genre, string? lyrics) => [.. _sp.EncodeToIds(BuildStage1PromptText(genre, lyrics))];

    /// <summary>The structured lyric segments (infer.py <c>split_lyrics</c>) — one per <c>[label]</c> section.</summary>
    public IReadOnlyList<string> Stage1Segments(string? lyrics) => SplitLyrics(lyrics);

    /// <summary>Per-segment prompt ids driving iterative Stage-1 generation (infer.py loop):
    /// <c>(isFirst ? [] : end_of_segment) + start_of_segment + tokenize(section) + [SOA] + [&lt;xcodec&gt; sep]</c>.
    /// The head ids are prepended once by the pipeline for the first segment.</summary>
    public int[] EncodeSegmentPrompt(string? sectionText, bool isFirst)
    {
        List<int> ids = new(32);
        if (!isFirst) ids.AddRange(EndOfSegmentIds);
        ids.AddRange(StartOfSegmentIds);
        ids.AddRange(_sp.EncodeToIds(sectionText ?? ""));
        ids.Add(Soa);
        ids.Add(Xcodec);
        return [.. ids];
    }

    /// <summary>Encodes arbitrary text to SentencePiece ids (no special-token wrapping).</summary>
    public IReadOnlyList<int> EncodeRaw(string text) => _sp.EncodeToIds(text ?? "");

    public void Dispose() { } // SentencePieceTokenizer holds no unmanaged handles.
}
