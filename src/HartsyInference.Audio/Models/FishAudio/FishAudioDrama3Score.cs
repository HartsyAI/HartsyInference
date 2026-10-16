namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>What <see cref="FishAudioDrama3Lm.Score"/> records over a prompt and a run of forced frames.</summary>
/// <param name="PromptHidden">Final-norm backbone hidden states of the prompt rows, <c>[rows, hidden]</c>.</param>
/// <param name="Candidates">The ids the main token is drawn over, the order of every candidate logit row.</param>
/// <param name="CandidateLogits">Per forced frame, the logits its main token was drawn from.</param>
/// <param name="HeadLogits">Per forced frame, every codebook head's logits with the frame's codes forced, <c>[N−1, V]</c>.</param>
/// <param name="Hidden">Per forced frame, the backbone hidden state after feeding it.</param>
public sealed record FishAudioDrama3Score(float[] PromptHidden, int[] Candidates, float[][] CandidateLogits, float[][] HeadLogits,
    float[][] Hidden);
