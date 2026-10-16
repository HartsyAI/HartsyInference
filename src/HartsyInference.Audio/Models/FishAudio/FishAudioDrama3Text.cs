using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Caller-text rules shared by the Drama 3 prompt builder and chunker, ported from fish-speech's
/// <c>fish_omni/generate.py</c>: the white-space set, speaker tags and the tokens only the prompt builder may write.</summary>
public static partial class FishAudioDrama3Text
{
    /// <summary>Highest speaker id the tokenizer has a token for (<c>&lt;|speaker:0|&gt;</c> to <c>&lt;|speaker:15|&gt;</c>).</summary>
    public const int MaxSpeakerId = 15;

    /// <summary>Tokens the prompt builder writes; other special tokens in caller text are content.</summary>
    public static readonly string[] StructuralTokens =
        ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"];

    // The Unicode White_Space property: Python's argument-less str.strip() also strips U+001C..U+001F.
    private static readonly char[] WhiteSpace =
    [
        (char)0x09, (char)0x0A, (char)0x0B, (char)0x0C, (char)0x0D, (char)0x20, (char)0x85, (char)0xA0, (char)0x1680,
        (char)0x2000, (char)0x2001, (char)0x2002, (char)0x2003, (char)0x2004, (char)0x2005, (char)0x2006, (char)0x2007,
        (char)0x2008, (char)0x2009, (char)0x200A, (char)0x2028, (char)0x2029, (char)0x202F, (char)0x205F, (char)0x3000,
    ];

    /// <summary>Matches one speaker tag, capturing its digits.</summary>
    [GeneratedRegex(@"<\|speaker:([0-9]+)\|>", RegexOptions.CultureInvariant)]
    public static partial Regex SpeakerTag();

    /// <summary>Strips Unicode white space from both ends.</summary>
    public static string Trim(string text) => text.Trim(WhiteSpace);

    /// <summary>Strips Unicode white space from the start.</summary>
    public static string TrimStart(string text) => text.TrimStart(WhiteSpace);

    /// <summary>True when <paramref name="text"/> holds nothing but Unicode white space.</summary>
    public static bool IsBlank(string text) => text.AsSpan().Trim(WhiteSpace).IsEmpty;

    /// <summary>The leading speaker tag of <paramref name="text"/> after white space, or null.</summary>
    public static Match? LeadingSpeakerTag(string text)
    {
        Match tag = SpeakerTag().Match(TrimStart(text));
        return tag.Success && tag.Index == 0 ? tag : null;
    }

    /// <summary>Rejects empty text, structural tokens and speaker tags outside canonical <c>0..15</c>.</summary>
    /// <param name="location">Names the text in the error, e.g. <c>reference 2</c>.</param>
    public static void Validate(string text, string location)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (IsBlank(text))
        {
            throw new ArgumentException($"Drama 3 {location}: text must not be empty.");
        }
        foreach (string token in StructuralTokens)
        {
            if (text.Contains(token, StringComparison.Ordinal))
            {
                throw new ArgumentException($"Drama 3 {location}: {token} is written by the prompt builder.");
            }
        }
        int marker = text.IndexOf("<|speaker", StringComparison.Ordinal);
        while (marker >= 0)
        {
            Match tag = SpeakerTag().Match(text, marker);
            string digits = tag.Success && tag.Index == marker ? tag.Groups[1].Value : "";
            bool canonical = digits.Length > 0 && (digits.Length == 1 || digits[0] != '0')
                && digits.Length <= 2 && int.Parse(digits, System.Globalization.CultureInfo.InvariantCulture) <= MaxSpeakerId;
            if (!canonical)
            {
                throw new ArgumentException(
                    $"Drama 3 {location}: speaker tags must be <|speaker:0|> to <|speaker:{MaxSpeakerId}|>.");
            }
            marker = text.IndexOf("<|speaker", marker + 1, StringComparison.Ordinal);
        }
    }

    /// <summary>Prefixes <c>&lt;|speaker:0|&gt; </c> unless a speaker tag already leads the text after white space.</summary>
    public static string WithSpeakerTag(string text) => LeadingSpeakerTag(text) is null ? $"<|speaker:0|> {text}" : text;

    /// <summary>A reference transcript as its <c>Text:</c> line carries it: tagged, with the tag moved in front of
    /// any leading white space so it directly follows <c>Text: </c>.</summary>
    public static string ReferenceTranscript(string transcript)
    {
        string tagged = WithSpeakerTag(transcript);
        string body = TrimStart(tagged);
        Match tag = SpeakerTag().Match(body);
        string leading = tagged[..(tagged.Length - body.Length)];
        return tag.Value + leading + body[tag.Length..];
    }
}
