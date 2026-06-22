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
    /// <summary>Start-of-audio: hands off from the text prompt to cb0 generation.</summary>
    public const string SoaPiece = "[SOA]";
    /// <summary>End-of-audio: stops cb0 generation.</summary>
    public const string EoaPiece = "[EOA]";
    public const string Stage1Piece = "[stage_1]";
    public const string StartOfSegmentPiece = "[start_of_segment]";
    public const string EndOfSegmentPiece = "[end_of_segment]";

    private readonly SentencePieceTokenizer _sp;

    public int Soa { get; }
    public int Eoa { get; }
    public int Stage1 { get; }
    public int StartOfSegment { get; }
    public int EndOfSegment { get; }

    public YueTokenizer(string tokenizerModelPath)
    {
        if (!File.Exists(tokenizerModelPath))
        {
            throw new FileNotFoundException($"YuE tokenizer.model not found: {tokenizerModelPath}", tokenizerModelPath);
        }
        using FileStream fs = File.OpenRead(tokenizerModelPath);
        _sp = SentencePieceTokenizer.Create(fs, addBeginningOfSentence: false, addEndOfSentence: false)
            ?? throw new InvalidOperationException($"Failed to load YuE SentencePiece tokenizer from '{tokenizerModelPath}'.");
        Soa = ResolveSpecial(SoaPiece);
        Eoa = ResolveSpecial(EoaPiece);
        Stage1 = ResolveSpecial(Stage1Piece);
        StartOfSegment = ResolveSpecial(StartOfSegmentPiece);
        EndOfSegment = ResolveSpecial(EndOfSegmentPiece);
    }

    /// <summary>Resolves a special token's id from the loaded model. A non-single result means the file
    /// isn't the YuE mm tokenizer (or carries the markers under different strings) — fail with guidance.</summary>
    private int ResolveSpecial(string piece)
    {
        IReadOnlyList<int> ids = _sp.EncodeToIds(piece);
        if (ids.Count != 1)
        {
            throw new InvalidOperationException(
                $"YuE tokenizer.model does not carry the special token '{piece}' as a single piece (got {ids.Count} ids). "
                + "Supply the YuE mm tokenizer (mm_tokenizer_v0.2_hf/tokenizer.model from m-a-p/xcodec_mini_infer).");
        }
        return ids[0];
    }

    /// <summary>The Stage-1 instruction + genre + lyrics text (no special tokens). Pure/testable; matches
    /// YuE infer.py's prompt head <c>"Generate music from the given lyrics segment by segment.\n[Genre] {genre}\n{lyrics}"</c>.</summary>
    public static string BuildStage1PromptText(string genre, string lyrics)
        => $"Generate music from the given lyrics segment by segment.\n[Genre] {(genre ?? "").Trim()}\n{(lyrics ?? "").Trim()}";

    /// <summary>Encodes the full Stage-1 prompt: instruction+genre+lyrics text, then the segment + stage-1 +
    /// start-of-audio markers that hand off to cb0 generation.</summary>
    public int[] EncodeStage1Prompt(string genre, string lyrics)
    {
        List<int> ids = [.. _sp.EncodeToIds(BuildStage1PromptText(genre, lyrics))];
        ids.Add(StartOfSegment);
        ids.Add(Stage1);
        ids.Add(Soa);
        return [.. ids];
    }

    /// <summary>Encodes arbitrary text to SentencePiece ids (no special-token wrapping).</summary>
    public IReadOnlyList<int> EncodeRaw(string text) => _sp.EncodeToIds(text ?? "");

    public void Dispose() { } // SentencePieceTokenizer holds no unmanaged handles.
}
