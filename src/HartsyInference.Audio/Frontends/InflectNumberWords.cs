using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>Python <c>inflect</c> 7.x <c>engine.number_to_words</c> and <c>engine.ordinal</c>, reproduced step for step
/// (chunking, the regex enword passes, comma and "and" placement, grouped readings). Text frontends trained on inflect
/// output, such as GPT-SoVITS's English normalizer, need its exact spelling, which differs from num2words.</summary>
internal static partial class InflectNumberWords
{
    private const string Decimal = "point";
    private const string Comma = ",";

    private static readonly string[] Unit = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
    private static readonly string[] Teen =
        ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
    private static readonly string[] Ten =
        ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    private static readonly string[] Mill =
    [
        " ", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion", " sextillion",
        " septillion", " octillion", " nonillion", " decillion",
    ];
    private static readonly Dictionary<string, string> OrdinalWords = new(StringComparer.Ordinal)
    {
        ["ty"] = "tieth", ["one"] = "first", ["two"] = "second", ["three"] = "third", ["five"] = "fifth",
        ["eight"] = "eighth", ["nine"] = "ninth", ["twelve"] = "twelfth",
    };
    private static readonly HashSet<string> NthSuffixes = new(StringComparer.Ordinal) { "th", "st", "nd", "rd" };

    /// <summary><c>number_to_words(num, group=group, andword=andword, zero=zero)</c> with inflect's other defaults
    /// (one="one", decimal="point", comma=","). Throws <see cref="ArgumentOutOfRangeException"/> past the decillions,
    /// where inflect raises <c>NumOutOfRangeError</c>.</summary>
    public static string NumberToWords(string num, int group = 0, string andword = "and", string zero = "zero")
    {
        ArgumentNullException.ThrowIfNull(num);
        ArgumentOutOfRangeException.ThrowIfLessThan(group, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(group, 3);
        ArgumentNullException.ThrowIfNull(andword);
        ArgumentNullException.ThrowIfNull(zero);
        string trimmed = num.TrimStart();
        string sign = trimmed.Length == 0 ? "" : trimmed[0] switch { '+' => "plus", '-' => "minus", _ => "" };
        if (NthSuffixes.Contains(num)) num = zero;
        bool myord = num.Length >= 2 && NthSuffixes.Contains(num[^2..]);
        if (myord) num = num[..^2];

        List<string> chunks = [.. group != 0 ? num.Split('.') : num.Split('.', 2)];
        bool finalPoint = chunks[^1].Length == 0;
        if (finalPoint) chunks.RemoveAt(chunks.Count - 1);
        int loopStart = chunks.Count > 0 && chunks[0].Length == 0 ? 1 : 0;
        // inflect's `first` is True, then None after the first chunk; False when the leading chunk is blank.
        bool first = loopStart == 0;
        bool firstChunkHandled = false;
        for (int i = loopStart; i < chunks.Count; i++)
        {
            string chunk = NonDigitRegex().Replace(chunks[i], "");
            if (chunk.Length == 0) chunk = "0";
            chunk = group == 0 && !first ? Enword(chunk, 1, andword, zero) : Enword(chunk, group, andword, zero);
            if (chunk.EndsWith(", ", StringComparison.Ordinal)) chunk = chunk[..^2];
            chunk = WhitespaceCommaRegex().Replace(chunk, ",");
            if (group == 0 && first) chunk = CommaWordRegex().Replace(chunk, " " + andword + " $1");
            chunk = WhitespaceRegex().Replace(chunk, " ").Trim();
            if (first) { first = false; firstChunkHandled = true; }
            chunks[i] = chunk;
        }

        List<string> numChunks = [];
        if (firstChunkHandled) numChunks.AddRange(chunks[0].Split(Comma + " "));
        if (myord && numChunks.Count > 0) numChunks[^1] = SubOrdinal(numChunks[^1]);
        for (int i = 1; i < chunks.Count; i++)
        {
            numChunks.Add(Decimal);
            numChunks.AddRange(chunks[i].Split(Comma + " "));
        }
        if (finalPoint) numChunks.Add(Decimal);

        string value = group != 0 ? string.Join(", ", numChunks) : Render(numChunks);
        return sign.Length > 0 ? sign + " " + value : value;
    }

    /// <summary><c>ordinal(num)</c>: a digit string gains its suffix ("1" → "1st", "112" → "112th"); a word gains
    /// its ordinal form ("four" → "fourth", "twenty" → "twentieth").</summary>
    public static string Ordinal(string num)
    {
        ArgumentNullException.ThrowIfNull(num);
        if (num.Length == 0 || !char.IsDigit(num[0])) return SubOrdinal(num);
        int lastDigit;
        if (num.Contains('.', StringComparison.Ordinal))
            lastDigit = char.IsDigit(num[^1]) ? num[^1] - '0' : LastTwoDigits(num[..^1]);
        else
            lastDigit = LastTwoDigits(num);
        string suffix = lastDigit is 11 or 12 or 13 ? "th" : (lastDigit % 10) switch
        {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        };
        return num + suffix;
    }

    private static int LastTwoDigits(string digits)
    {
        string tail = digits.Length > 2 ? digits[^2..] : digits;
        return int.Parse(tail, System.Globalization.CultureInfo.InvariantCulture) % 100;
    }

    private static string SubOrdinal(string value)
    {
        string replaced = OrdinalSuffixRegex().Replace(value, m => OrdinalWords[m.Groups[1].Value]);
        return replaced == value ? value + "th" : replaced;
    }

    private static string Render(List<string> chunks)
    {
        if (chunks.Count == 0) throw new ArgumentException("inflect cannot render an empty number.");
        StringBuilder sb = new(chunks[0]);
        bool first = !chunks[0].EndsWith(Decimal, StringComparison.Ordinal);
        for (int i = 1; i < chunks.Count; i++)
        {
            if (chunks[i] == Decimal) first = false;
            else if (first) sb.Append(Comma);
            sb.Append(' ').Append(chunks[i]);
        }
        return sb.ToString();
    }

    private static string Enword(string num, int group, string andword, string zero)
    {
        switch (group)
        {
            case 1:
                return DigitRegex().Replace(num, m => Group1(m.Groups[1].Value[0] - '0', zero));
            case 2:
                num = TwoDigitsRegex().Replace(num, m => Group2(m.Groups[1].Value[0] - '0', m.Groups[2].Value[0] - '0', zero));
                return DigitRegex().Replace(num, m => Group1b(m.Groups[1].Value[0] - '0', zero), 1);
            case 3:
                num = ThreeDigitsRegex().Replace(num, m => Group3(m, zero));
                num = TwoDigitsRegex().Replace(num, m => Group2(m.Groups[1].Value[0] - '0', m.Groups[2].Value[0] - '0', zero), 1);
                return DigitRegex().Replace(num, m => Group1(m.Groups[1].Value[0] - '0', zero), 1);
        }
        string significant = num.TrimStart('0');
        if (significant.Length == 0) return zero;
        if (significant == "1") return "one";
        num = num.TrimStart().TrimStart('0');
        int millCount = 0;
        while (ThreeDigitsWordRegex().IsMatch(num))
        {
            num = ThreeDigitsWordRegex().Replace(num, m =>
            {
                string r = Hundreds(m.Groups[1].Value[0] - '0', m.Groups[2].Value[0] - '0', m.Groups[3].Value[0] - '0',
                    millCount, andword);
                millCount++;
                return r;
            }, 1);
        }
        num = TwoDigitsWordRegex().Replace(num, m =>
            Tens(m.Groups[1].Value[0] - '0', m.Groups[2].Value[0] - '0', millCount) + ", ", 1);
        return OneDigitWordRegex().Replace(num, m => Unit[m.Groups[1].Value[0] - '0'] + MillName(millCount) + ", ", 1);
    }

    private static string MillName(int index)
    {
        if (index >= Mill.Length) throw new ArgumentOutOfRangeException(nameof(index), "Number is beyond inflect's decillions.");
        return Mill[index];
    }

    private static string Tens(int tens, int units, int millIndex)
    {
        if (tens != 1)
            return Ten[tens] + (tens != 0 && units != 0 ? "-" : "") + Unit[units] + MillName(millIndex);
        return Teen[units] + MillName(millIndex);
    }

    private static string Hundreds(int hundreds, int tens, int units, int millIndex, string andword)
    {
        if (hundreds != 0)
        {
            string and = tens != 0 || units != 0 ? " " + andword + " " : "";
            return Unit[hundreds] + " hundred" + and + Tens(tens, units, 0) + MillName(millIndex) + ", ";
        }
        if (tens != 0 || units != 0) return Tens(tens, units, 0) + MillName(millIndex) + ", ";
        return "";
    }

    private static string Group1(int units, string zero) => units switch
    {
        1 => " one, ",
        0 => " " + zero + ", ",
        _ => Unit[units] + ", ",
    };

    private static string Group1b(int units, string zero) => units != 0 ? Unit[units] + ", " : " " + zero + ", ";

    private static string Group2(int tens, int units, string zero)
    {
        if (tens != 0) return Tens(tens, units, 0) + ", ";
        return units != 0 ? " " + zero + " " + Unit[units] + ", " : " " + zero + " " + zero + ", ";
    }

    private static string Group3(Match m, string zero)
    {
        int hundreds = m.Groups[1].Value[0] - '0', tens = m.Groups[2].Value[0] - '0', units = m.Groups[3].Value[0] - '0';
        string hunword = hundreds switch { 1 => " one", 0 => " " + zero, _ => Unit[hundreds] };
        string tenword = tens != 0 ? Tens(tens, units, 0)
            : units != 0 ? " " + zero + " " + Unit[units] : " " + zero + " " + zero;
        return hunword + " " + tenword + ", ";
    }

    [GeneratedRegex(@"\D")]
    private static partial Regex NonDigitRegex();

    [GeneratedRegex(@"\s+,")]
    private static partial Regex WhitespaceCommaRegex();

    [GeneratedRegex(@", (\S+)\s+\z")]
    private static partial Regex CommaWordRegex();

    [GeneratedRegex(@"\s+")]
    private static partial Regex WhitespaceRegex();

    [GeneratedRegex(@"(ty|one|two|three|five|eight|nine|twelve)\z")]
    private static partial Regex OrdinalSuffixRegex();

    [GeneratedRegex(@"(\d)")]
    private static partial Regex DigitRegex();

    [GeneratedRegex(@"(\d)(\d)")]
    private static partial Regex TwoDigitsRegex();

    [GeneratedRegex(@"(\d)(\d)(\d)")]
    private static partial Regex ThreeDigitsRegex();

    [GeneratedRegex(@"(\d)(\d)(\d)(?=\D*\z)")]
    private static partial Regex ThreeDigitsWordRegex();

    [GeneratedRegex(@"(\d)(\d)(?=\D*\z)")]
    private static partial Regex TwoDigitsWordRegex();

    [GeneratedRegex(@"(\d)(?=\D*\z)")]
    private static partial Regex OneDigitWordRegex();
}
