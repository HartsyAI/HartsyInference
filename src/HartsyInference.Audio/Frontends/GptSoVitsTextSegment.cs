namespace HartsyInference.Audio.Frontends;

/// <summary>One GPT-SoVITS synthesis segment: the text as cut, its normalized form and its phones.</summary>
public readonly record struct GptSoVitsTextSegment(string Text, string NormalizedText, IReadOnlyList<string> Phones);
