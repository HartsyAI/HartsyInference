using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>Port of the Python <c>inflect</c> package's <c>engine().number_to_words</c> for digit-only strings with its
/// default arguments (group 0, comma ",", andword "and"), the form TTS frontends use to spell out digit runs
/// (CosyVoice's <c>spell_out_number</c>). Output matches inflect 7.3.1 exactly, including its comma placement
/// ("one thousand, nine hundred and ninety-nine") and the "and" before a final group ("two thousand and twenty-six").</summary>
public static partial class InflectNumberWords
{
    private static readonly string[] Unit = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
    private static readonly string[] Teen = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
    private static readonly string[] Ten = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    private static readonly string[] Mill =
        [" ", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion", " sextillion", " septillion", " octillion", " nonillion", " decillion"];

    [GeneratedRegex(@"(\d)(\d)(\d)(?=\D*\z)")]
    private static partial Regex ThreeDigitsWord();

    [GeneratedRegex(@"(\d)(\d)(?=\D*\z)")]
    private static partial Regex TwoDigitsWord();

    [GeneratedRegex(@"(\d)(?=\D*\z)")]
    private static partial Regex OneDigitWord();

    [GeneratedRegex(@"\s+,")]
    private static partial Regex WhitespacesComma();

    [GeneratedRegex(@", (\S+)\s+\z")]
    private static partial Regex CommaWord();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespaces();

    /// <summary>Spells a run of ASCII digits as words.</summary>
    /// <exception cref="ArgumentException">The input is empty or contains a non-digit.</exception>
    public static string ToWords(string digits)
    {
        ArgumentException.ThrowIfNullOrEmpty(digits);
        foreach (char c in digits)
            if (c is < '0' or > '9') throw new ArgumentException($"Expected digits only, got '{digits}'.", nameof(digits));

        string chunk = Enword(digits);
        if (chunk.EndsWith(", ", StringComparison.Ordinal)) chunk = chunk[..^2];
        chunk = WhitespacesComma().Replace(chunk, ",");
        chunk = CommaWord().Replace(chunk, " and $1");
        chunk = Whitespaces().Replace(chunk, " ");
        return chunk.Trim();
    }

    /// <summary>Replaces every run of ASCII digits in <paramref name="text"/> with its words, leaving everything else as is
    /// (CosyVoice <c>cosyvoice/utils/frontend_utils.py</c> <c>spell_out_number</c>).</summary>
    public static string SpellOutDigitRuns(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        StringBuilder sb = new(text.Length);
        int start = -1;
        for (int i = 0; i <= text.Length; i++)
        {
            bool digit = i < text.Length && char.IsDigit(text[i]);
            if (digit)
            {
                if (start < 0) start = i;
                continue;
            }
            if (start >= 0)
            {
                sb.Append(ToWords(ToAsciiDigits(text.AsSpan(start, i - start))));
                start = -1;
            }
            if (i < text.Length) sb.Append(text[i]);
        }
        return sb.ToString();
    }

    // Python's str.isdigit accepts every Unicode decimal digit; inflect then parses the run with int(), which maps them
    // to their ASCII values. Mirror that so a full-width or other-script digit run spells out the same way.
    private static string ToAsciiDigits(ReadOnlySpan<char> run)
    {
        StringBuilder sb = new(run.Length);
        foreach (char c in run) sb.Append((char)('0' + (int)char.GetNumericValue(c)));
        return sb.ToString();
    }

    private static string Enword(string num)
    {
        string trimmed = num.TrimStart('0');
        if (trimmed.Length == 0) return "zero";
        if (trimmed == "1") return "one";

        num = trimmed;
        int millCount = 0;
        Match three = ThreeDigitsWord().Match(num);
        while (three.Success)
        {
            string words = HundFn(Digit(three, 1), Digit(three, 2), Digit(three, 3), millCount);
            num = num[..three.Index] + words + num[(three.Index + three.Length)..];
            millCount++;
            three = ThreeDigitsWord().Match(num);
        }
        Match two = TwoDigitsWord().Match(num);
        if (two.Success)
            num = num[..two.Index] + TenFn(Digit(two, 1), Digit(two, 2), millCount) + ", " + num[(two.Index + two.Length)..];
        Match one = OneDigitWord().Match(num);
        if (one.Success)
            num = num[..one.Index] + Unit[Digit(one, 1)] + Mill[millCount] + ", " + num[(one.Index + one.Length)..];
        return num;
    }

    private static int Digit(Match m, int group) => m.Groups[group].Value[0] - '0';

    private static string TenFn(int tens, int units, int mindex = 0)
    {
        if (tens != 1)
        {
            string hyphen = tens != 0 && units != 0 ? "-" : "";
            return Ten[tens] + hyphen + Unit[units] + Mill[mindex];
        }
        return Teen[units] + Mill[mindex];
    }

    private static string HundFn(int hundreds, int tens, int units, int mindex)
    {
        if (hundreds != 0)
        {
            string andWord = tens != 0 || units != 0 ? " and " : "";
            return $"{Unit[hundreds]} hundred{andWord}{TenFn(tens, units)}{Mill[mindex]}, ";
        }
        if (tens != 0 || units != 0) return $"{TenFn(tens, units)}{Mill[mindex]}, ";
        return "";
    }
}
