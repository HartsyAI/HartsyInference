using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;
using HartsyInference.Audio.Frontends;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>ZipVoice's <c>EnglishTextNormalizer</c> (the espnet_tts_frontend cleaners): abbreviations, currency, fractions,
/// decimals, percentages, ordinals and numbers become words, the text the model was trained on.</summary>
internal static partial class ZipVoiceEnglishNormalizer
{
    private static readonly (Regex Pattern, string Word)[] Abbreviations = BuildAbbreviations(
    [
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"), ("jr", "junior"),
        ("maj", "major"), ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"),
        ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"), ("esq", "esquire"), ("ltd", "limited"),
        ("col", "colonel"), ("ft", "fort"), ("etc", "et cetera"), ("btw", "by the way"),
    ]);

    internal static string Normalize(string text)
    {
        foreach ((Regex pattern, string word) in Abbreviations) text = pattern.Replace(text, word);
        text = CommaNumber().Replace(text, m => m.Groups[1].Value.Replace(",", "", StringComparison.Ordinal));
        text = Pounds().Replace(text, "$1 pounds");
        text = Dollars().Replace(text, ExpandDollars);
        text = Fraction().Replace(text, ExpandFraction);
        text = DecimalNumber().Replace(text, m => m.Groups[1].Value.Replace(".", " point ", StringComparison.Ordinal));
        text = Percent().Replace(text, m => m.Groups[1].Value.Replace("%", " percent ", StringComparison.Ordinal));
        text = Ordinal().Replace(text, m => " " + InflectNumberWords.NumberToWords(m.Value) + " ");
        return Number().Replace(text, ExpandNumber);
    }

    private static (Regex, string)[] BuildAbbreviations((string Abbreviation, string Word)[] pairs)
    {
        (Regex, string)[] built = new (Regex, string)[pairs.Length];
        for (int i = 0; i < pairs.Length; i++)
        {
            Regex pattern = new($"\\b{pairs[i].Abbreviation}\\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            built[i] = (pattern, pairs[i].Word);
        }
        return built;
    }

    // Upstream raises on an amount int() cannot read and loses the whole text; the amount is kept as written here.
    private static string ExpandDollars(Match m)
    {
        string amount = m.Groups[1].Value;
        string[] parts = amount.Split('.');
        if (parts.Length > 2) return " " + amount + " dollars ";
        BigInteger dollars = BigInteger.Zero, cents = BigInteger.Zero;
        if (parts[0].Length > 0 && !BigInteger.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out dollars))
            return m.Value;
        if (parts.Length > 1 && parts[1].Length > 0
            && !BigInteger.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out cents))
            return m.Value;
        string dollarText = Plural(dollars, "dollar", "dollars"), centText = Plural(cents, "cent", "cents");
        if (!dollars.IsZero && !cents.IsZero) return $" {dollars} {dollarText}, {cents} {centText} ";
        if (!dollars.IsZero) return $" {dollars} {dollarText} ";
        return cents.IsZero ? " zero dollars " : $" {cents} {centText} ";
    }

    private static string Plural(BigInteger count, string one, string many) => count.IsOne ? one : many;

    private static string ExpandFraction(Match m)
    {
        BigInteger numerator = BigInteger.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
        BigInteger denominator = BigInteger.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
        if (numerator.IsOne && denominator == 2) return " one half ";
        if (numerator.IsOne && denominator == 4) return " one quarter ";
        string numeratorWords = InflectNumberWords.NumberToWords(numerator.ToString(CultureInfo.InvariantCulture));
        if (denominator == 2) return $" {numeratorWords} halves ";
        if (denominator == 4) return $" {numeratorWords} quarters ";
        string denominatorWords = InflectNumberWords.NumberToWords(denominator.ToString(CultureInfo.InvariantCulture));
        return $" {numeratorWords} {InflectNumberWords.Ordinal(denominatorWords)} ";
    }

    private static string ExpandNumber(Match m)
    {
        BigInteger value = BigInteger.Parse(m.Value, CultureInfo.InvariantCulture);
        if (value > 1000 && value < 3000)
        {
            int year = (int)value;
            if (year == 2000) return " two thousand ";
            if (year > 2000 && year < 2010) return $" two thousand {Words(year % 100)} ";
            if (year % 100 == 0) return $" {Words(year / 100)} hundred ";
            string grouped = InflectNumberWords.NumberToWords(year.ToString(CultureInfo.InvariantCulture), 2, "", "oh");
            return " " + grouped.Replace(", ", " ", StringComparison.Ordinal) + " ";
        }
        return " " + InflectNumberWords.NumberToWords(value.ToString(CultureInfo.InvariantCulture), andword: "") + " ";
    }

    private static string Words(int number) => InflectNumberWords.NumberToWords(number.ToString(CultureInfo.InvariantCulture));

    [GeneratedRegex(@"([0-9][0-9,]+[0-9])")]
    private static partial Regex CommaNumber();

    [GeneratedRegex(@"£([0-9,]*[0-9]+)")]
    private static partial Regex Pounds();

    [GeneratedRegex(@"\$([0-9.,]*[0-9]+)")]
    private static partial Regex Dollars();

    [GeneratedRegex(@"([0-9]+)/([0-9]+)")]
    private static partial Regex Fraction();

    [GeneratedRegex(@"([0-9]+\.[0-9]+)")]
    private static partial Regex DecimalNumber();

    [GeneratedRegex(@"([0-9.,]*[0-9]+%)")]
    private static partial Regex Percent();

    [GeneratedRegex(@"[0-9]+(st|nd|rd|th)")]
    private static partial Regex Ordinal();

    [GeneratedRegex(@"[0-9]+")]
    private static partial Regex Number();
}
