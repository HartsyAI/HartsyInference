using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>GPT-SoVITS's English text normalization: upstream <c>english.text_normalize</c>, which runs the
/// <c>en_normalization/expend.py</c> rules (ordinals, arithmetic, times, units, currency, decimals, fractions,
/// numbers via inflect), strips accents and every character outside <c> A-Za-z'.,?!-</c>, splits capitals and
/// collapses punctuation runs. The rules are reproduced as written, including readings a general normalizer would
/// not choose ("4:30 PM" → "four thirty a.m. P M", "1990s" → "nineteen ninety seconds"): the model was trained on
/// this text.</summary>
public static partial class GptSoVitsEnglishNormalizer
{
    // Upstream builds this map's pattern with re.escape over the keys, so the two bracketed keys only match their own
    // literal text; only the single-character full-width marks are effectively mapped.
    private static readonly (string From, string To)[] RepMap =
        [("[;:：，；]", ","), ("[\"’]", "'"), ("。", "."), ("！", "!"), ("？", "?")];

    private static readonly Dictionary<string, (string One, string Many)> Measurements = new(StringComparer.Ordinal)
    {
        ["m"] = ("meter", "meters"), ["km"] = ("kilometer", "kilometers"),
        ["km/h"] = ("kilometer per hour", "kilometers per hour"), ["ft"] = ("feet", "feet"), ["L"] = ("liter", "liters"),
        ["tbsp"] = ("tablespoon", "tablespoons"), ["tsp"] = ("teaspoon", "teaspoons"), ["h"] = ("hour", "hours"),
        ["min"] = ("minute", "minutes"), ["s"] = ("second", "seconds"), ["°C"] = ("degree celsius", "degrees celsius"),
        ["°F"] = ("degree fahrenheit", "degrees fahrenheit"),
    };

    private static readonly Dictionary<string, string> AsmdWords = new(StringComparer.Ordinal)
    {
        ["+"] = " plus ", ["-"] = " minus ", ["×"] = " times ", ["÷"] = " divided by ", ["="] = " Equals ",
    };

    /// <summary>Upstream <c>english.text_normalize</c>.</summary>
    public static string TextNormalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        foreach ((string from, string to) in RepMap) text = text.Replace(from, to, StringComparison.Ordinal);
        text = Normalize(text);
        return ConsecutivePunctuationRegex().Replace(text, "$1");
    }

    /// <summary>Upstream <c>en_normalization.expend.normalize</c>.</summary>
    public static string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = OrdinalNumberRegex().Replace(text, m => InflectNumberWords.Ordinal(m.Groups[1].Value) + ", ");
        while (AsmdRegex().IsMatch(text))
            text = AsmdRegex().Replace(text, m => m.Groups[1].Value + AsmdWords[m.Groups[8].Value] + m.Groups[9].Value);
        text = NegativeIntegerRegex().Replace(text, m => "negative " + m.Groups[2].Value);

        text = CommaNumberRegex().Replace(text, m => m.Groups[1].Value.Replace(",", "", StringComparison.Ordinal));
        text = TimeRegex().Replace(text, ExpandTime);
        text = MeasurementRegex().Replace(text, ExpandMeasurement);
        text = PoundsStartRegex().Replace(text, m => ExpandCurrency(m, "pound", "pounds", "penny", "pence"));
        text = PoundsEndRegex().Replace(text, m => ExpandCurrency(m, "pound", "pounds", "penny", "pence"));
        text = DollarsStartRegex().Replace(text, m => ExpandCurrency(m, "dollar", "dollars", "cent", "cents"));
        text = DollarsEndRegex().Replace(text, m => ExpandCurrency(m, "dollar", "dollars", "cent", "cents"));
        text = DecimalNumberRegex().Replace(text, ExpandDecimal);
        text = FractionRegex().Replace(text, ExpandFraction);
        text = OrdinalRegex().Replace(text, m => InflectNumberWords.NumberToWords(m.Value));
        text = NumberRegex().Replace(text, ExpandNumber);

        text = StripAccents(text);
        text = text.Replace("%", " percent", StringComparison.Ordinal);
        text = DisallowedCharsRegex().Replace(text, "");
        text = ThatIsRegex().Replace(text, "that is");
        text = ForExampleRegex().Replace(text, "for example");
        return CapitalSplitRegex().Replace(text, " $1");
    }

    private static string StripAccents(string text)
    {
        string decomposed = text.Normalize(NormalizationForm.FormD);
        StringBuilder sb = new(decomposed.Length);
        foreach (char ch in decomposed)
            if (CharUnicodeInfo.GetUnicodeCategory(ch) != UnicodeCategory.NonSpacingMark) sb.Append(ch);
        return sb.ToString();
    }

    private static string ExpandTime(Match m)
    {
        int hours = int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
        int minutes = int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
        string period = hours < 12 ? "a.m." : "p.m.";
        if (hours > 12) hours -= 12;
        string hourWord = Words(hours);
        if (minutes == 0) return hourWord + " o'clock " + period;
        return hourWord + " " + Words(minutes) + " " + period;
    }

    private static string ExpandMeasurement(Match m)
    {
        string sign = m.Groups[3].Value;
        string whole = m.Groups[1].Value;
        string digits = whole.Replace(sign, "", StringComparison.Ordinal).Replace(".", "", StringComparison.Ordinal);
        bool one = !m.Groups[2].Success && BigInteger.Parse(digits, CultureInfo.InvariantCulture) == BigInteger.One;
        (string singular, string plural) = Measurements[sign];
        return whole.Replace(sign, " " + (one ? singular : plural), StringComparison.Ordinal);
    }

    // Upstream raises on a malformed amount (a leading comma or paren reaches int()); that text is kept as written.
    private static string ExpandCurrency(Match m, string major, string majors, string minor, string minors)
    {
        string match = m.Groups[1].Value;
        string[] parts = match.Split('.');
        if (parts.Length > 2) return match + " " + majors;
        BigInteger whole = BigInteger.Zero, fraction = BigInteger.Zero;
        if (parts[0].Length > 0 && !BigInteger.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out whole))
            return m.Value;
        if (parts.Length > 1 && parts[1].Length > 0
            && !BigInteger.TryParse(parts[1].PadRight(2, '0'), NumberStyles.None, CultureInfo.InvariantCulture, out fraction))
            return m.Value;
        string wholeText = whole.ToString(CultureInfo.InvariantCulture) + " " + (whole == BigInteger.One ? major : majors);
        string fractionText = fraction.ToString(CultureInfo.InvariantCulture) + " " + (fraction == BigInteger.One ? minor : minors);
        if (!whole.IsZero && !fraction.IsZero) return wholeText + " and " + fractionText;
        if (!whole.IsZero) return wholeText;
        if (!fraction.IsZero) return fractionText;
        return "zero " + majors;
    }

    private static string ExpandDecimal(Match m)
    {
        string[] parts = m.Groups[1].Value.Split('.');
        StringBuilder sb = new(parts[0]);
        sb.Append(" point ");
        for (int i = 0; i < parts[1].Length; i++)
        {
            if (i > 0) sb.Append(' ');
            sb.Append(parts[1][i]);
        }
        return sb.ToString();
    }

    private static string ExpandFraction(Match m)
    {
        string[] parts = m.Value.Split('/');
        BigInteger numerator = BigInteger.Parse(parts[0], CultureInfo.InvariantCulture);
        BigInteger denominator = BigInteger.Parse(parts[1], CultureInfo.InvariantCulture);
        string numeratorPart = InflectNumberWords.NumberToWords(numerator.ToString(CultureInfo.InvariantCulture));
        string denominatorPart;
        if (denominator == 2)
            denominatorPart = numerator == BigInteger.One ? "half" : "halves";
        else if (denominator == BigInteger.One)
            return numeratorPart;
        else
        {
            denominatorPart = InflectNumberWords.Ordinal(
                InflectNumberWords.NumberToWords(denominator.ToString(CultureInfo.InvariantCulture)));
            if (numerator > BigInteger.One) denominatorPart += "s";
        }
        return numeratorPart + " " + denominatorPart;
    }

    private static string ExpandNumber(Match m)
    {
        BigInteger num = BigInteger.Parse(m.Value, CultureInfo.InvariantCulture);
        string digits = num.ToString(CultureInfo.InvariantCulture);
        if (num > 1000 && num < 3000)
        {
            int n = (int)num;
            if (n == 2000) return "two thousand";
            if (n > 2000 && n < 2010) return "two thousand " + Words(n % 100);
            if (n % 100 == 0) return Words(n / 100) + " hundred";
            return InflectNumberWords.NumberToWords(digits, group: 2, andword: "", zero: "oh")
                .Replace(", ", " ", StringComparison.Ordinal);
        }
        return InflectNumberWords.NumberToWords(digits, andword: "");
    }

    private static string Words(int n) => InflectNumberWords.NumberToWords(n.ToString(CultureInfo.InvariantCulture));

    [GeneratedRegex(@"([!?…,.\-\s])([!?…,.\-])+")]
    private static partial Regex ConsecutivePunctuationRegex();

    [GeneratedRegex(@"\b([0-9]+)\. ")]
    private static partial Regex OrdinalNumberRegex();

    [GeneratedRegex(@"((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))\s+([+\-×÷=])\s+((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))")]
    private static partial Regex AsmdRegex();

    [GeneratedRegex(@"(?:^|\s+)(-)(\d+)")]
    private static partial Regex NegativeIntegerRegex();

    [GeneratedRegex(@"([0-9][0-9,]+[0-9])")]
    private static partial Regex CommaNumberRegex();

    [GeneratedRegex(@"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b")]
    private static partial Regex TimeRegex();

    [GeneratedRegex(@"\b([0-9]+(\.[0-9]+)?(m|km|km/h|ft|L|tbsp|tsp|h|min|s|°C|°F))\b")]
    private static partial Regex MeasurementRegex();

    [GeneratedRegex(@"£([0-9.,]*[0-9]+)")]
    private static partial Regex PoundsStartRegex();

    [GeneratedRegex(@"([0-9.,]*[0-9]+)£")]
    private static partial Regex PoundsEndRegex();

    [GeneratedRegex(@"\$([0-9.,]*[0-9]+)")]
    private static partial Regex DollarsStartRegex();

    [GeneratedRegex(@"([(0-9.,]*[0-9]+)\$")]
    private static partial Regex DollarsEndRegex();

    [GeneratedRegex(@"([0-9]+\.\s*[0-9]+)")]
    private static partial Regex DecimalNumberRegex();

    [GeneratedRegex(@"([0-9]+/[0-9]+)")]
    private static partial Regex FractionRegex();

    [GeneratedRegex(@"[0-9]+(st|nd|rd|th)")]
    private static partial Regex OrdinalRegex();

    [GeneratedRegex(@"[0-9]+")]
    private static partial Regex NumberRegex();

    [GeneratedRegex(@"[^ A-Za-z'.,?!\-]")]
    private static partial Regex DisallowedCharsRegex();

    [GeneratedRegex(@"i\.e\.", RegexOptions.IgnoreCase)]
    private static partial Regex ThatIsRegex();

    [GeneratedRegex(@"e\.g\.", RegexOptions.IgnoreCase)]
    private static partial Regex ForExampleRegex();

    [GeneratedRegex(@"(?<!^)(?<!\s)([A-Z])")]
    private static partial Regex CapitalSplitRegex();
}
