using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Builds S2 prompts the way fish-speech's <c>Conversation</c>/<c>ContentSequence</c> do. Each text part is
/// tokenized on its own (so BPE boundaries fall where upstream's do); a code part contributes
/// <c>semanticBegin + code0</c> as its token and the full code column as the codebook rows.</summary>
public sealed class FishAudioS2Prompt
{
    public const string ImStart = "<|im_start|>";
    public const string ImEnd = "<|im_end|>";
    public const string VoiceModality = "<|voice|>";

    private static readonly Regex SpeakerTag = new(@"(<\|speaker:\d+\|>)", RegexOptions.Compiled);
    private static readonly Regex HasSpeakerTag = new(@"<\|speaker:\d+\|>", RegexOptions.Compiled);

    private readonly Func<string, int[]> _encode;
    private readonly int _semanticBegin;
    private readonly List<int> _tokens = [];
    private readonly List<int[]?> _codes = [];

    /// <param name="encode">Tokenizes one text part, parsing special tokens inline and adding none of its own.</param>
    public FishAudioS2Prompt(Func<string, int[]> encode, int semanticBeginId)
    {
        _encode = encode;
        _semanticBegin = semanticBeginId;
    }

    public int Length => _tokens.Count;
    public int[] Tokens => [.. _tokens];
    public IReadOnlyList<int[]?> Codes => _codes;

    /// <summary>Replaces this prompt's contents with a copy of <paramref name="other"/>'s.</summary>
    public void CopyFrom(FishAudioS2Prompt other)
    {
        _tokens.Clear(); _codes.Clear();
        _tokens.AddRange(other._tokens); _codes.AddRange(other._codes);
    }

    public FishAudioS2Prompt Text(string text)
    {
        foreach (int id in _encode(text.Normalize(NormalizationForm.FormC)))
        {
            _tokens.Add(id);
            _codes.Add(null);
        }
        return this;
    }

    /// <summary>Appends <paramref name="codes"/> (<c>[numCodebooks, T]</c>, row 0 = semantic) as T frames.</summary>
    public FishAudioS2Prompt AppendCodes(int[,] codes)
    {
        int rows = codes.GetLength(0), t = codes.GetLength(1);
        for (int j = 0; j < t; j++)
        {
            int[] column = new int[rows];
            for (int i = 0; i < rows; i++) column[i] = codes[i, j];
            _tokens.Add(_semanticBegin + column[0]);
            _codes.Add(column);
        }
        return this;
    }

    /// <summary>The opening of a turn: <c>&lt;|im_start|&gt;role\n</c> plus the modality token, if any.</summary>
    public FishAudioS2Prompt StartTurn(string role, bool voice = false) => Text($"{ImStart}{role}\n{(voice ? VoiceModality : "")}");

    public FishAudioS2Prompt EndTurn() => Text($"{ImEnd}\n");

    /// <summary>System turn: plain instruction, or — with a reference — the reference text and its codes.</summary>
    public FishAudioS2Prompt System(string? referenceText = null, int[,]? referenceCodes = null)
    {
        StartTurn("system");
        if (referenceCodes is null)
            Text("convert the provided text to speech");
        else
        {
            string tagged = HasSpeakerTag.IsMatch(referenceText ?? "") ? referenceText! : $"<|speaker:0|>{referenceText}";
            Text("convert the provided text to speech reference to the following:\n\nText:\n");
            Text(tagged);
            Text("\n\nSpeech:\n");
            AppendCodes(referenceCodes);
        }
        return EndTurn();
    }

    public FishAudioS2Prompt User(string text) => StartTurn("user").Text(text).EndTurn();

    /// <summary>A finished assistant turn (a previous batch's generated codes).</summary>
    public FishAudioS2Prompt Assistant(int[,] codes) => StartTurn("assistant", voice: true).AppendCodes(codes).EndTurn();

    /// <summary>The open assistant turn the model continues from.</summary>
    public FishAudioS2Prompt OpenAssistant() => StartTurn("assistant", voice: true);

    /// <summary>fish-speech <c>split_text_by_speaker</c> + <c>group_turns_into_batches</c>: text without speaker tags
    /// is a single batch; tagged text is cut into turns grouped by speaker count and UTF-8 size.</summary>
    public static List<string> SplitBatches(string text, int maxSpeakers = 5, int maxBytes = 512)
    {
        string[] parts = SpeakerTag.Split(text);
        List<string> turns = [];
        for (int i = 0; i < parts.Length;)
        {
            string part = parts[i].Trim();
            if (SpeakerTag.IsMatch(part) && part.StartsWith("<|speaker:", StringComparison.Ordinal))
            {
                if (i + 1 < parts.Length) { turns.Add((part + parts[i + 1]).Trim()); i += 2; }
                else { turns.Add(part); i++; }
            }
            else i++;
        }
        if (turns.Count == 0) return [text];

        List<string> batches = [];
        List<string> current = [];
        int bytes = 0;
        foreach (string turn in turns)
        {
            int turnBytes = Encoding.UTF8.GetByteCount(turn);
            if (current.Count >= maxSpeakers || (bytes + turnBytes > maxBytes && current.Count > 0))
            {
                batches.Add(string.Join("\n", current));
                current = [turn];
                bytes = turnBytes;
            }
            else { current.Add(turn); bytes += turnBytes; }
        }
        if (current.Count > 0) batches.Add(string.Join("\n", current));
        return batches;
    }
}
