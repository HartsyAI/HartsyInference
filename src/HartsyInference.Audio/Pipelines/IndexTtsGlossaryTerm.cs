namespace HartsyInference.Audio.Pipelines;

/// <summary>One upstream <c>glossary.yaml</c> entry: the reading used on the English and on the Chinese
/// normalization path. A null reading leaves the term as written (upstream's <c>term_value.get(lang, term)</c>); a
/// plain-string YAML value sets both readings.</summary>
public readonly record struct IndexTtsGlossaryTerm(string? English, string? Chinese);
