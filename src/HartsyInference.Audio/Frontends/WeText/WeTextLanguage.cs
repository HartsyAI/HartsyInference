namespace HartsyInference.Audio.Frontends.WeText;

/// <summary>The grammar a <see cref="WeTextNormalizer"/> runs.</summary>
public enum WeTextLanguage
{
    /// <summary>Chinese when the text holds a CJK Unified Ideograph (U+4E00..U+9FFF), English otherwise.</summary>
    Auto,

    /// <summary>The English grammar (<c>en/tn</c>).</summary>
    English,

    /// <summary>The Chinese grammar (<c>zh/tn</c>).</summary>
    Chinese,
}
