namespace HartsyInference.Audio.Frontends;

/// <summary>misaki <c>TokenContext</c>: whether the next spoken sound after the token being read is a vowel, null
/// when punctuation or nothing follows.</summary>
internal sealed record TokenContext(bool? FutureVowel);
