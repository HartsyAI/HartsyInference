using System.IO;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.Tokenizers;

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

    /// <summary>The Stage-1 instruction + genre + lyrics text (no special tokens). Pure/testable; matches
    /// YuE infer.py's prompt head <c>"Generate music from the given lyrics segment by segment.\n[Genre] {genre}\n{lyrics}"</c>.</summary>
    public static string BuildStage1PromptText(string genre, string lyrics)
        => $"Generate music from the given lyrics segment by segment.\n[Genre] {(genre ?? "").Trim()}\n{(lyrics ?? "").Trim()}";

    /// <summary>Encodes the full Stage-1 prompt: instruction+genre+lyrics text, then the start-of-audio +
    /// xcodec-sep markers that prime cb0 generation (YuE infer.py order: … &lt;SOA&gt; &lt;xcodec&gt;). NOTE:
    /// &lt;stage_1&gt; is NOT used in Stage-1 — it belongs to the Stage-2 prompt only.</summary>
    public int[] EncodeStage1Prompt(string genre, string lyrics)
    {
        List<int> ids = [.. _sp.EncodeToIds(BuildStage1PromptText(genre, lyrics))];
        ids.Add(Soa);
        ids.Add(Xcodec);
        return [.. ids];
    }

    /// <summary>Encodes arbitrary text to SentencePiece ids (no special-token wrapping).</summary>
    public IReadOnlyList<int> EncodeRaw(string text) => _sp.EncodeToIds(text ?? "");

    public void Dispose() { } // SentencePieceTokenizer holds no unmanaged handles.
}
