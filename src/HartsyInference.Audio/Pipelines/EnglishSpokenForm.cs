using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Spoken-form expansion of English numerals, symbols and common abbreviations, reproducing the cases of
/// WeTextProcessing's English normalizer (the one the IndexTTS reference runs before tokenizing) that matter for
/// ordinary prompts, in the same style: no hyphens (<c>twenty five</c>), an <c>and</c> only inside the final group
/// (<c>one hundred and one</c>, <c>two hundred thirty four thousand five hundred and sixty seven</c>), four-digit
/// years read in pairs (<c>twenty twenty five</c>), <c>2.0 → two point oh</c>, <c>% → percent</c>,
/// <c>$5 → five dollars</c>, ordinals, clock times and the usual titles. Everything outside these patterns passes
/// through unchanged — the reference model was trained on normalized text, so digits left in place are read poorly
/// (<c>"2.0"</c> comes out as "wine" without this).</summary>
internal static partial class EnglishSpokenForm
{
    private static readonly string[] Ones =
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
    private static readonly string[] Tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    private static readonly (long Value, string Name)[] Scales = [(1_000_000_000_000L, "trillion"), (1_000_000_000L, "billion"), (1_000_000L, "million"), (1_000L, "thousand")];

    private static readonly Dictionary<string, string> Titles = new(StringComparer.Ordinal)
    {
        ["Dr."] = "doctor", ["Mr."] = "Mister", ["Mrs."] = "Misses", ["Ms."] = "Miss", ["Jr."] = "junior", ["Mt."] = "Mount",
    };

    private static readonly Dictionary<string, string> Units = new(StringComparer.Ordinal)
    {
        ["km/h"] = "kilometers per hour", ["mph"] = "miles per hour", ["km"] = "kilometers", ["mi"] = "miles", ["kg"] = "kilograms",
        ["cm"] = "centimeters", ["mm"] = "millimeters", ["ft"] = "feet", ["lb"] = "pounds", ["lbs"] = "pounds",
    };

    [GeneratedRegex(@"\b(?:Dr|Mr|Mrs|Ms|Jr|Mt)\.")]
    private static partial Regex TitlePattern();

    [GeneratedRegex(@"\b(?:U\.S\.A\.?|U\.S\.?|U\.K\.?|E\.U\.?)", RegexOptions.None)]
    private static partial Regex DottedAcronym();

    [GeneratedRegex(@"\b([ap])\.m\.?", RegexOptions.IgnoreCase)]
    private static partial Regex DottedMeridiem();

    [GeneratedRegex(@"(?<=\d)\s?([ap])m\b", RegexOptions.IgnoreCase)]
    private static partial Regex Meridiem();

    [GeneratedRegex(@"\$(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")]
    private static partial Regex Money();

    [GeneratedRegex(@"(?<![\w.])(-)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?%")]
    private static partial Regex Percent();

    [GeneratedRegex(@"\b(\d{1,2}):(\d{2})\b")]
    private static partial Regex ClockTime();

    [GeneratedRegex(@"\b(\d+):(\d+)\b")]
    private static partial Regex Ratio();

    [GeneratedRegex(@"\b(\d+)(st|nd|rd|th)\b")]
    private static partial Regex Ordinal();

    [GeneratedRegex(@"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)\.(\d+)(?![\w.]*\d)")]
    private static partial Regex Decimal();

    [GeneratedRegex(@"(?<![\w-])-(\d+)\b")]
    private static partial Regex Negative();

    [GeneratedRegex(@"\b(\d+)\s?(km/h|mph|km|mi|kg|cm|mm|ft|lbs|lb)\b")]
    private static partial Regex Measure();

    [GeneratedRegex(@"\d{1,3}(?:,\d{3})+|\d+")]
    private static partial Regex Integer();

    public static string Expand(string text)
    {
        if (text.AsSpan().IndexOfAnyInRange('0', '9') < 0 && !text.Contains('.', StringComparison.Ordinal) && !text.Contains('%', StringComparison.Ordinal))
            return text;

        string s = TitlePattern().Replace(text, static m => Titles[m.Value]);
        s = DottedAcronym().Replace(s, static m => m.Value.Replace(".", "", StringComparison.Ordinal));
        s = DottedMeridiem().Replace(s, static m => m.Groups[1].Value.ToUpperInvariant() + "M");
        s = Meridiem().Replace(s, static m => " " + m.Groups[1].Value.ToUpperInvariant() + "M");
        s = Money().Replace(s, static m => m.Groups[2].Success
            ? $"{Decimal(m.Groups[1].Value, m.Groups[2].Value)} dollars"
            : $"{Cardinal(ParseInt(m.Groups[1].Value))} dollars");
        s = Percent().Replace(s, static m => $"{(m.Groups[1].Success ? "negative " : "")}{(m.Groups[3].Success ? Decimal(m.Groups[2].Value, m.Groups[3].Value) : Cardinal(ParseInt(m.Groups[2].Value)))} percent");
        s = ClockTime().Replace(s, static m => ClockWords(int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture), int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture)));
        s = Ratio().Replace(s, static m => $"{Cardinal(ParseInt(m.Groups[1].Value))} to {Cardinal(ParseInt(m.Groups[2].Value))}");
        s = Ordinal().Replace(s, static m => OrdinalWords(ParseInt(m.Groups[1].Value)));
        s = Measure().Replace(s, static m => $"{Cardinal(ParseInt(m.Groups[1].Value))} {Units[m.Groups[2].Value]}");
        s = Decimal().Replace(s, static m => Decimal(m.Groups[1].Value, m.Groups[2].Value));
        s = Negative().Replace(s, static m => "negative " + Spoken(m.Groups[1].Value));
        string source = s;
        return Integer().Replace(source, m =>
        {
            bool letterBefore = m.Index > 0 && char.IsLetter(source[m.Index - 1]);
            bool letterAfter = m.Index + m.Length < source.Length && char.IsLetter(source[m.Index + m.Length]);
            return (letterBefore ? " " : "") + Spoken(m.Value) + (letterAfter ? " " : "");
        });
    }

    private static string Spoken(string digits)
    {
        string plain = digits.Replace(",", "", StringComparison.Ordinal);
        if (plain.Length > 1 && plain[0] == '0') return string.Join(' ', plain.Select(static c => c == '0' ? "oh" : Ones[c - '0']));
        long n = ParseInt(plain);
        if (!digits.Contains(',', StringComparison.Ordinal) && plain.Length == 4 && IsYearPair(n)) return YearWords((int)n);
        return Cardinal(n);
    }

    private static long ParseInt(string digits) => long.TryParse(digits.Replace(",", "", StringComparison.Ordinal), NumberStyles.None, CultureInfo.InvariantCulture, out long n) ? n : 0;

    /// <summary>Four-digit numbers the reference reads like a year/pair: 1010–1999 (bar x000, 1001–1009) and 2010–2199.</summary>
    private static bool IsYearPair(long n) => (n >= 1010 && n <= 1999 && n % 1000 != 0) || (n >= 2010 && n <= 2199);

    private static string YearWords(int n)
    {
        int head = n / 100, tail = n % 100;
        if (tail == 0) return $"{Cardinal(head)} hundred";
        if (tail < 10) return $"{Cardinal(head)} oh {Ones[tail]}";
        return $"{Cardinal(head)} {Cardinal(tail)}";
    }

    /// <summary><c>whole.fraction</c> → "whole point d d d" (a zero digit is "oh").</summary>
    private static string Decimal(string whole, string fractionWithDot)
    {
        string frac = fractionWithDot.TrimStart('.');
        StringBuilder sb = new();
        sb.Append(Cardinal(ParseInt(whole))).Append(" point");
        foreach (char c in frac) sb.Append(' ').Append(c == '0' ? "oh" : Ones[c - '0']);
        return sb.ToString();
    }

    private static string ClockWords(int hour, int minute)
    {
        if (hour > 24 || minute > 59) return $"{hour}:{minute:D2}";
        string h = Cardinal(hour);
        if (minute == 0) return h;
        return minute < 10 ? $"{h} o {Ones[minute]}" : $"{h} {Cardinal(minute)}";
    }

    internal static string Cardinal(long n)
    {
        if (n < 0) return "negative " + Cardinal(-n);
        if (n < 20) return Ones[n];
        StringBuilder sb = new();
        AppendGroups(sb, n);
        return sb.ToString();
    }

    private static void AppendGroups(StringBuilder sb, long n, bool final = true)
    {
        foreach ((long value, string name) in Scales)
        {
            if (n < value) continue;
            AppendGroups(sb, n / value, final: false);
            sb.Append(' ').Append(name);
            n %= value;
            if (n == 0) return;
            sb.Append(' ');
            AppendGroups(sb, n, final);
            return;
        }
        AppendBelowThousand(sb, n, final);
    }

    /// <summary>0 &lt; n &lt; 1000; the "and" appears only when this is the last group and has both hundreds and a remainder.</summary>
    private static void AppendBelowThousand(StringBuilder sb, long n, bool final)
    {
        if (n >= 100)
        {
            sb.Append(Ones[n / 100]).Append(" hundred");
            n %= 100;
            if (n == 0) return;
            sb.Append(final ? " and " : " ");
        }
        if (n < 20) sb.Append(Ones[n]);
        else
        {
            sb.Append(Tens[n / 10]);
            if (n % 10 != 0) sb.Append(' ').Append(Ones[n % 10]);
        }
    }

    private static string OrdinalWords(long n)
    {
        string card = Cardinal(n);
        int cut = card.LastIndexOf(' ') + 1;
        string last = card[cut..];
        string ord = last switch
        {
            "one" => "first", "two" => "second", "three" => "third", "five" => "fifth", "eight" => "eighth", "nine" => "ninth", "twelve" => "twelfth",
            _ when last.EndsWith('y') => last[..^1] + "ieth",
            _ => last + "th",
        };
        return card[..cut] + ord;
    }
}
