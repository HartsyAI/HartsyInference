using System.IO;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>YuE "mm" tokenizer — the LLaMA SentencePiece model extended with YuE's structural / audio special tokens, loaded from the model's <c>tokenizer.model</c> (m-a-p/xcodec_mini_infer, <c>mm_tokenizer_v0.2_hf/</c>). Builds the Stage-1 lyrics→token prompt for <c>YueStage1Lm</c>.
///
/// <para>IDs are resolved from the loaded model, not hardcoded — so they're correct for whatever checkpoint the user supplies. The Stage-1 prompt's special-token tail follows YuE's <c>infer.py</c> (CoT) / docs/Research/YUE_ARCHITECTURE.md §7; verify token parity (bit-exact first ~100 tokens) against the reference per the M5 checklist before treating output as faithful.</para></summary>
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
    /// <summary>xcodec codec-type separator — CodecManipulator("xcodec").sep_ids. Ends the Stage-1 prompt (… &lt;SOA&gt; &lt;xcodec&gt;) to prime cb0 generation.</summary>
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

    /// <summary>Confirms a pinned control-symbol id maps back to its expected angle-bracket piece via <c>Decode</c> (the reliable piece-&gt;id direction for SP control symbols). A mismatch means the file isn't the YuE mm tokenizer — fail with guidance.</summary>
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

    /// <summary>Splits lyrics into YuE's structured segments (one per <c>[verse]/[chorus]/…</c> tag), mirroring infer.py's <c>split_lyrics</c>: each becomes <c>"[label]\n{text}\n\n"</c>.</summary>
    public static List<string> SplitLyrics(string? lyrics)
    {
        List<string> segs = [];
        foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(
                     lyrics ?? "", @"\[(\w+)\](.*?)(?=\[|\Z)", System.Text.RegularExpressions.RegexOptions.Singleline))
            segs.Add($"[{m.Groups[1].Value}]\n{m.Groups[2].Value.Trim()}\n\n");
        return segs;
    }

    /// <summary>The Stage-1 "head" prompt (instruction + genre + full structured lyrics) — infer.py <c>prompt_texts[0]</c>. Testable; the structured lyrics join each <c>[label]</c> segment with a blank line, matching <c>split_lyrics</c>.</summary>
    public static string BuildStage1PromptText(string? genre, string? lyrics)
    {
        List<string> segs = SplitLyrics(lyrics);
        string full = segs.Count > 0 ? string.Join("\n", segs) : (lyrics ?? "").Trim();
        return $"Generate music from the given lyrics segment by segment.\n[Genre] {(genre ?? "").Trim()}\n{full}";
    }

    /// <summary>Encodes YuE's Stage-1 **segment-0** prompt EXACTLY as infer.py builds it: <c>tokenize(head) + tokenize("[start_of_segment]") + tokenize(section[0]) + [SOA] + [&lt;xcodec&gt; sep]</c>. The head restates the genre + all structured lyrics as context; the section restates segment-0's lyrics (this <c>[start_of_segment]</c> + section restatement is what primes coherent cb0 generation — omitting it yields gibberish). Later segments are appended by the pipeline's segment loop (<c>[end_of_segment]</c> + <c>[start_of_segment]</c> + section + [SOA] + sep). <c>&lt;stage_1&gt;</c> is Stage-2 only.</summary>
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

    /// <summary>Builds a subsequent-segment continuation prompt (infer.py, i&gt;0): <c>tokenize("[end_of_segment][start_of_segment]") + tokenize(section) + [SOA] + [&lt;xcodec&gt; sep]</c>. The pipeline prepends the running generated context and appends this per segment.</summary>
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

    /// <summary>Per-segment prompt ids driving iterative Stage-1 generation (infer.py loop): <c>(isFirst ? [] : end_of_segment) + start_of_segment + tokenize(section) + [SOA] + [&lt;xcodec&gt; sep]</c>. The head ids are prepended once by the pipeline for the first segment.</summary>
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

    // ── Reference-audio in-context learning (infer.py --use_audio_prompt / --use_dual_tracks_prompt) ──

    /// <summary>CodecManipulator("xcodec") global token offset: LM id = 45334 + k*1024 + code. The ICL path runs the codec at target_bw=0.5 ⇒ codebook 0 only ⇒ k = 0.</summary>
    public const int XcodecGlobalOffset = 45_334;

    /// <summary>x-codec frame rate (tokens per second per track).</summary>
    public const int XcodecFps = 50;

    /// <summary>Builds infer.py's <c>audio_prompt_codec</c>: offsets raw codebook-0 indices into the LM's audio-token range, interleaves the two tracks when a dual-track reference is supplied, then slices the requested second-window. Pure arithmetic — no tokenizer state.
    ///
    /// <para>Single-track slices at <c>fps</c> tokens/second; dual-track interleaves <c>v0,i0,v1,i1,…</c> and slices at <c>2·fps</c>, so the window means the same wall-clock span either way. An odd dual-track start index flips the vocal/instrumental parity of the prompt — upstream does not correct this, so neither does this.</para></summary>
    /// <param name="vocalCb0">Codebook-0 indices of the single/vocal reference track.</param>
    /// <param name="instrumentalCb0">Codebook-0 indices of the instrumental track; empty selects the single-track path.</param>
    public static int[] BuildAudioPromptCodec(ReadOnlySpan<int> vocalCb0, ReadOnlySpan<int> instrumentalCb0,
        double startSeconds, double endSeconds, int globalOffset = XcodecGlobalOffset, int fps = XcodecFps)
    {
        bool dual = instrumentalCb0.Length > 0;
        if (dual && vocalCb0.Length != instrumentalCb0.Length)
        {
            throw new ArgumentException(
                $"Dual-track reference audio must encode to equal lengths (vocal {vocalCb0.Length} frames, "
                + $"instrumental {instrumentalCb0.Length}). Trim the two clips to the same duration.",
                nameof(instrumentalCb0));
        }

        int perSecond = dual ? fps * 2 : fps;
        int total = dual ? vocalCb0.Length * 2 : vocalCb0.Length;
        // Python's int() truncates toward zero, as does the C# cast; Python slicing tolerates an overrunning stop.
        int lo = Math.Clamp((int)(startSeconds * perSecond), 0, total);
        int hi = Math.Clamp((int)(endSeconds * perSecond), lo, total);

        int[] window = new int[hi - lo];
        for (int i = lo; i < hi; i++)
        {
            window[i - lo] = globalOffset + (dual ? ((i & 1) == 0 ? vocalCb0[i >> 1] : instrumentalCb0[i >> 1])
                : vocalCb0[i]);
        }
        return window;
    }

    /// <summary>Wraps offset reference codes in infer.py's <c>sentence_ids</c>: <c>tokenize("[start_of_reference]") + [SOA] + [&lt;xcodec&gt;] + codes + [EOA] + tokenize("[end_of_reference]")</c>. The markers are plain SentencePiece text, not control symbols. Append this to the head prompt (NOT before it) — infer.py builds <c>head_id = tokenize(prompt_texts[0]) + sentence_ids</c>, so only segment 0 carries it and later segments inherit it through the running context.</summary>
    public int[] EncodeReferenceBlock(IReadOnlyList<int> audioPromptCodec)
    {
        List<int> ids = new(audioPromptCodec.Count + 16);
        ids.AddRange(EncodeRaw("[start_of_reference]"));
        ids.Add(Soa);
        ids.Add(Xcodec);
        ids.AddRange(audioPromptCodec);
        ids.Add(Eoa);
        ids.AddRange(EncodeRaw("[end_of_reference]"));
        return [.. ids];
    }

    public void Dispose() { } // SentencePieceTokenizer holds no unmanaged handles.
}
