using System.Globalization;

namespace HartsyInference.Audio.Frontends;

/// <summary>The subset of Python <c>num2words</c> (English) that misaki's number reading calls: cardinals, ordinals,
/// years and decimals. Output keeps num2words' spelling ("one hundred and five", "twenty-four") because the caller
/// splits on non-letters and drops "and", exactly as misaki does with num2words' strings.</summary>
internal static class EnglishNumberWords
{
    private static readonly string[] Ones =
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
    private static readonly string[] Tens =
        ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    private static readonly (long Value, string Name)[] Scales =
    [
        (1_000_000_000_000_000_000L, "quintillion"), (1_000_000_000_000_000L, "quadrillion"),
        (1_000_000_000_000L, "trillion"), (1_000_000_000L, "billion"), (1_000_000L, "million"), (1_000L, "thousand"),
    ];
    private static readonly Dictionary<string, string> OrdinalWords = new(StringComparer.Ordinal)
    {
        ["one"] = "first", ["two"] = "second", ["three"] = "third", ["five"] = "fifth", ["eight"] = "eighth",
        ["nine"] = "ninth", ["twelve"] = "twelfth",
    };

    /// <summary><c>num2words(n)</c>.</summary>
    public static string Cardinal(long n)
    {
        if (n < 0) return "minus " + Cardinal(-n);
        if (n < 20) return Ones[n];
        if (n < 100) return Tens[n / 10] + (n % 10 != 0 ? "-" + Ones[n % 10] : "");
        if (n < 1000) return Ones[n / 100] + " hundred" + (n % 100 != 0 ? " and " + Cardinal(n % 100) : "");
        foreach ((long value, string name) in Scales)
        {
            if (n < value) continue;
            long rest = n % value;
            string head = Cardinal(n / value) + " " + name;
            if (rest == 0) return head;
            return head + (rest < 100 ? " and " : ", ") + Cardinal(rest);
        }
        throw new ArgumentOutOfRangeException(nameof(n), n, "unreachable");
    }

    /// <summary><c>num2words(n, to='ordinal')</c>.</summary>
    public static string Ordinal(long n)
    {
        string words = Cardinal(n);
        int cut = Math.Max(words.LastIndexOf(' '), words.LastIndexOf('-')) + 1;
        string last = words[cut..];
        string ordinal = OrdinalWords.TryGetValue(last, out string? irregular) ? irregular
            : last.EndsWith('y') ? last[..^1] + "ieth" : last + "th";
        return words[..cut] + ordinal;
    }

    /// <summary><c>num2words(n, to='year')</c>: "nineteen ninety", "nineteen oh-five", "two thousand and five".</summary>
    public static string Year(long n)
    {
        long high = n / 100, low = n % 100;
        if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) return Cardinal(n);
        string lowText = low == 0 ? "hundred" : low < 10 ? "oh-" + Cardinal(low) : Cardinal(low);
        return Cardinal(high) + " " + lowText;
    }

    /// <summary><c>num2words(float(text))</c>: the integer part, then "point" and each fraction digit of the shortest
    /// round-trip repr (a whole value reads as a cardinal).</summary>
    public static string Decimal(string text)
    {
        double value = double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
        string repr = value.ToString("R", CultureInfo.InvariantCulture);
        if (repr.Contains('E', StringComparison.Ordinal) || Math.Abs(value) >= long.MaxValue)
            return string.Join(' ', text.Where(char.IsAsciiDigit).Select(c => Ones[c - '0']));
        int dot = repr.IndexOf('.');
        if (dot < 0) return Cardinal(long.Parse(repr, CultureInfo.InvariantCulture));
        string whole = repr[..dot];
        string fraction = repr[(dot + 1)..];
        List<string> words = [Cardinal(long.Parse(whole, CultureInfo.InvariantCulture)), "point"];
        foreach (char c in fraction) words.Add(Ones[c - '0']);
        return string.Join(' ', words);
    }
}
