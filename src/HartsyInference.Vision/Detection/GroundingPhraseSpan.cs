namespace HartsyInference.Vision.Detection;

/// <summary>A caption phrase and the half-open token index range <c>[Start, End)</c> it occupies in the tokenized
/// caption. Grounding DINO scores each phrase by aggregating the contrastive logits over this span.</summary>
public readonly record struct GroundingPhraseSpan(string Phrase, int Start, int End);
