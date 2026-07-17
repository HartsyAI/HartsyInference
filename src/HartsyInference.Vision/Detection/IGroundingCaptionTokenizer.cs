namespace HartsyInference.Vision.Detection;

/// <summary>Turns an open-vocabulary caption ("a cat . a remote control .") into token ids plus per-phrase token
/// spans. Abstracted so the detector does not hard-depend on a specific tokenizer (the default is BERT WordPiece,
/// matching the text tower), and so tests can substitute a trivial tokenizer with no vocab file.</summary>
public interface IGroundingCaptionTokenizer
{
    /// <summary>Tokenizes <paramref name="caption"/> and records where each phrase lands in the token sequence.</summary>
    GroundingCaptionTokens Tokenize(string caption);
}
