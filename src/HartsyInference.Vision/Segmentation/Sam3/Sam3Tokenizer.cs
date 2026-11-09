using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 prompt tokenizer, a port of upstream <c>SimpleTokenizer</c> (facebookresearch/sam3 <c>model/tokenizer_ve.py</c>).
/// Output is <c>[SOT, content..., EOT, 0, ...]</c> of length <see cref="ContextLength"/>; when the content does not fit, it is cut
/// to fit and the last slot is forced to EOT. Padding is id 0, which is also the byte-level <c>!</c> token (G2), so the padding mask
/// is derived as <c>id == 0</c> by <see cref="Sam3TextEncoder"/>.
/// Pre-tokens use upstream's regex. ASCII pre-tokens are encoded by the shared <see cref="ClipTokenizer"/> (its BPE matches upstream
/// there). Non-ASCII pre-tokens need the UTF-8 byte-level mapping, which <see cref="ClipTokenizer"/> does not apply, so they run a
/// local merge loop over the same <c>clip_vocab.json</c> and <c>clip_merges.txt</c>. Known gaps against upstream: ftfy repairs beyond
/// NFC and the common curly quotes (mojibake, width folding, ligatures, control characters) are not reproduced, and HTML entities
/// use .NET's decoder.</summary>
public sealed class Sam3Tokenizer : IDisposable
{
    /// <summary>Start-of-text token id (upstream <c>sot_token_id</c>).</summary>
    public const int StartOfTextId = ClipTokenizer.StartOfTextId;

    /// <summary>End-of-text token id (upstream <c>eot_token_id</c>).</summary>
    public const int EndOfTextId = ClipTokenizer.EndOfTextId;

    /// <summary>Padding token id. Upstream pads with 0, which is the byte-level <c>!</c> token (G2).</summary>
    public const int PaddingId = 0;

    /// <summary>Tokens per prompt, including SOT, EOT and padding (upstream <c>VETextEncoder</c> context length).</summary>
    public const int ContextLength = 32;

    /// <summary>Number of BPE merges upstream reads from the file (<c>merges[1 : 49152 - 256 - 2 + 1]</c>, after the header line).</summary>
    private const int MergeCount = 49152 - 256 - 2;

    /// <summary>Upstream pre-token pattern (CLIP's), with IgnoreCase as upstream compiles it.</summary>
    private static readonly Regex _preTokenPattern = new(
        @"<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    /// <summary>The single and double curly quotes that ftfy folds to ASCII quotes.</summary>
    private static readonly SearchValues<char> _curlyQuotes = SearchValues.Create("‘’‚‛“”„‟");

    private readonly ClipTokenizer _clip;
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string First, string Second), int> _mergeRanks;
    private int _disposed;

    /// <summary>Loads the CLIP vocabulary and merges embedded in this assembly.</summary>
    public Sam3Tokenizer()
    {
        _clip = new ClipTokenizer();
        _vocab = LoadVocab();
        _mergeRanks = LoadMergeRanks();
    }

    /// <summary>Encodes one prompt into <see cref="ContextLength"/> token ids, framed by SOT and EOT and padded with 0. The same prompt
    /// always gives the same ids. An empty prompt gives <c>[SOT, EOT, 0, ...]</c>, as upstream does.</summary>
    /// <param name="prompt">The text prompt. Case and whitespace runs do not change the ids.</param>
    public int[] Encode(string prompt)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ThrowIfDisposed();

        string cleaned = CleanText(prompt);
        List<int> content = new(cleaned.Length);
        foreach (Match match in _preTokenPattern.Matches(cleaned))
        {
            AppendPreToken(match.Value, content);
        }

        int[] ids = new int[ContextLength];
        int kept = Math.Min(content.Count, ContextLength - 2);
        ids[0] = StartOfTextId;
        for (int i = 0; i < kept; i++)
        {
            ids[i + 1] = content[i];
        }

        // Upstream keeps the first ContextLength tokens of [SOT, content, EOT] and forces the last one to EOT when it truncates.
        ids[kept + 1] = EndOfTextId;
        return ids;
    }

    /// <summary>Releases the shared BPE resources.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _clip.Dispose();
        }
    }

    /// <summary>Upstream <c>_clean_lower</c> with the ftfy subset this port reproduces: NFC, common curly quotes folded to straight quotes,
    /// HTML entities decoded twice, then lowercase. Whitespace collapsing is omitted because the pre-token pattern ignores whitespace.</summary>
    private static string CleanText(string text)
    {
        string normalized = text.Normalize(NormalizationForm.FormC);
        normalized = FoldCurlyQuotes(normalized);
        normalized = WebUtility.HtmlDecode(WebUtility.HtmlDecode(normalized));
        return normalized.ToLowerInvariant();
    }

    private static string FoldCurlyQuotes(string text)
    {
        if (text.AsSpan().IndexOfAny(_curlyQuotes) < 0)
        {
            return text;
        }

        StringBuilder builder = new(text.Length);
        foreach (char c in text)
        {
            switch (c)
            {
                case '‘':
                case '’':
                case '‚':
                case '‛':
                    builder.Append('\'');
                    break;
                case '“':
                case '”':
                case '„':
                case '‟':
                    builder.Append('"');
                    break;
                default:
                    builder.Append(c);
                    break;
            }
        }
        return builder.ToString();
    }

    private void AppendPreToken(string token, List<int> content)
    {
        if (token == "<start_of_text>")
        {
            content.Add(StartOfTextId);
            return;
        }

        if (token == "<end_of_text>")
        {
            content.Add(EndOfTextId);
            return;
        }

        if (Ascii.IsValid(token))
        {
            foreach (int id in _clip.EncodeRaw(token))
            {
                content.Add(id);
            }
            return;
        }

        AppendByteLevelToken(token, content);
    }

    /// <summary>Upstream <c>bpe()</c> on the UTF-8 byte-level form of one pre-token: the mapped chars, the last one suffixed with
    /// <c>&lt;/w&gt;</c>, merged by rank until no ranked pair remains.</summary>
    private void AppendByteLevelToken(string token, List<int> content)
    {
        string mapped = ByteLevelCodec.Encode(token);
        List<string> word = new(mapped.Length);
        for (int i = 0; i < mapped.Length; i++)
        {
            word.Add(mapped[i].ToString());
        }
        word[^1] += "</w>";

        while (word.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIndex = -1;
            for (int i = 0; i + 1 < word.Count; i++)
            {
                if (_mergeRanks.TryGetValue((word[i], word[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
            {
                break;
            }

            string first = word[bestIndex];
            string second = word[bestIndex + 1];
            List<string> merged = new(word.Count);
            int k = 0;
            while (k < word.Count)
            {
                if (k + 1 < word.Count && word[k] == first && word[k + 1] == second)
                {
                    merged.Add(first + second);
                    k += 2;
                }
                else
                {
                    merged.Add(word[k]);
                    k++;
                }
            }
            word = merged;
        }

        foreach (string piece in word)
        {
            if (!_vocab.TryGetValue(piece, out int id))
            {
                throw new InvalidDataException($"SAM 3 tokenizer: BPE piece '{piece}' is missing from the CLIP vocabulary.");
            }
            content.Add(id);
        }
    }

    private static Dictionary<string, int> LoadVocab()
    {
        using Stream stream = EmbeddedTokenizerResources.OpenClipVocab();
        Dictionary<string, int>? vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(stream);
        return vocab ?? throw new InvalidDataException("SAM 3 tokenizer: the embedded CLIP vocabulary is empty.");
    }

    private static Dictionary<(string First, string Second), int> LoadMergeRanks()
    {
        Dictionary<(string First, string Second), int> ranks = new(MergeCount);
        using Stream stream = EmbeddedTokenizerResources.OpenClipMerges();
        using StreamReader reader = new(stream, Encoding.UTF8);
        _ = reader.ReadLine(); // header line "#version: 0.2", skipped by upstream
        string? line;
        while (ranks.Count < MergeCount && (line = reader.ReadLine()) is not null)
        {
            if (line.Length == 0)
            {
                continue;
            }

            string[] parts = line.Split(' ');
            if (parts.Length != 2)
            {
                throw new InvalidDataException($"SAM 3 tokenizer: malformed BPE merge line '{line}'.");
            }
            ranks[(parts[0], parts[1])] = ranks.Count;
        }

        if (ranks.Count != MergeCount)
        {
            throw new InvalidDataException($"SAM 3 tokenizer: expected {MergeCount} BPE merges, found {ranks.Count}.");
        }
        return ranks;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            throw new ObjectDisposedException(nameof(Sam3Tokenizer));
        }
    }
}
