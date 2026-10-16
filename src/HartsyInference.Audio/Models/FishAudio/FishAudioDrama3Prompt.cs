using System.Text;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Builds Drama 3 prompts as fish-speech's <c>build_drama3_prompt</c> does: a system turn carrying every voice reference
/// (<c>[Reference N]</c> blocks of transcript and codes), earlier conversation turns (user text, then the assistant's codes), and the
/// open assistant voice turn the model continues. A row is <c>[token, code 0, …, code N−1]</c>: text rows carry zero codes, audio rows
/// <c>[semantic_start + code 0, codes…]</c>. Every literal is tokenized on its own, as upstream does, since tokenizing the joined
/// string changes ids at the seams.</summary>
/// <param name="encode">Tokenizes one literal, parsing special tokens inline and adding none of its own.</param>
/// <param name="config">The model the prompt is for.</param>
public sealed class FishAudioDrama3Prompt(Func<string, int[]> encode, FishAudioDrama3Config config)
{
    public const string TtsInstruction = "convert the provided text to speech";
    public const string ReferenceInstruction = "convert the provided text to speech reference to the following:";
    public const string MessageEnd = "<|im_end|>\n";
    public const string GenerationStart = "<|im_start|>assistant\n<|voice|>";

    private readonly Func<string, int[]> _encode = encode ?? throw new ArgumentNullException(nameof(encode));
    private readonly FishAudioDrama3Config _config = config ?? throw new ArgumentNullException(nameof(config));

    /// <summary>Values per row: the token, then one code per codebook.</summary>
    public int Width => 1 + _config.NumCodebooks;

    /// <summary>The prompt rows <c>[rows, Width]</c> for speaking <paramref name="text"/>.</summary>
    /// <param name="maxRows">Most rows the prompt may hold; <see cref="FishAudioDrama3Config.MaxPromptRows"/> unless the model has less room.</param>
    /// <exception cref="FishAudioDrama3PromptTooLongException">The prompt exceeds <paramref name="maxRows"/>.</exception>
    public int[,] Build(string text, IReadOnlyList<FishAudioDrama3Turn> references, IReadOnlyList<FishAudioDrama3Turn> history,
        int maxRows = FishAudioDrama3Config.MaxPromptRows)
    {
        ArgumentNullException.ThrowIfNull(references);
        ArgumentNullException.ThrowIfNull(history);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxRows);
        FishAudioDrama3Text.Validate(text, "text");
        List<int[]> rows = [];
        Text(rows, "<|im_start|>system\n");
        Text(rows, references.Count > 0 ? ReferenceInstruction : TtsInstruction);
        for (int i = 0; i < references.Count; i++)
        {
            string location = $"reference {i + 1}";
            FishAudioDrama3Text.Validate(references[i].Text, location);
            CheckCodes(references[i].Codes, location);
            Text(rows, $"\n\n[Reference {i + 1}]\nText: {FishAudioDrama3Text.ReferenceTranscript(references[i].Text)}\nSpeech: ");
            Codes(rows, references[i].Codes);
        }
        Text(rows, MessageEnd);
        for (int i = 0; i < history.Count; i++)
        {
            string location = $"history {i + 1}";
            FishAudioDrama3Text.Validate(history[i].Text, location);
            CheckCodes(history[i].Codes, location);
            Text(rows, "<|im_start|>user\n");
            Text(rows, FishAudioDrama3Text.WithSpeakerTag(history[i].Text));
            Text(rows, MessageEnd);
            Text(rows, GenerationStart);
            Codes(rows, history[i].Codes);
            Text(rows, MessageEnd);
        }
        Text(rows, "<|im_start|>user\n");
        Text(rows, FishAudioDrama3Text.WithSpeakerTag(text));
        Text(rows, MessageEnd);
        Text(rows, GenerationStart);
        if (rows.Count > maxRows)
        {
            throw new FishAudioDrama3PromptTooLongException(rows.Count, maxRows);
        }
        int[,] result = new int[rows.Count, Width];
        for (int r = 0; r < rows.Count; r++)
        {
            for (int c = 0; c < Width; c++) result[r, c] = rows[r][c];
        }
        return result;
    }

    /// <summary>The prompt for one chunk of a long text: the caller's <paramref name="fixedHistory"/> always, then as much of
    /// <paramref name="chunkHistory"/> as fits in <paramref name="maxRows"/>, trying its candidates longest first (fish-speech
    /// <c>build_drama3_chunk_prompt</c>).</summary>
    public int[,] BuildWithHistory(string text, IReadOnlyList<FishAudioDrama3Turn> references, IReadOnlyList<FishAudioDrama3Turn> fixedHistory,
        FishAudioDrama3History chunkHistory, int maxRows = FishAudioDrama3Config.MaxPromptRows)
    {
        ArgumentNullException.ThrowIfNull(fixedHistory);
        ArgumentNullException.ThrowIfNull(chunkHistory);
        FishAudioDrama3PromptTooLongException? last = null;
        foreach (IReadOnlyList<FishAudioDrama3Turn> turns in chunkHistory.Candidates())
        {
            try
            {
                return Build(text, references, [.. fixedHistory, .. turns], maxRows);
            }
            catch (FishAudioDrama3PromptTooLongException tooLong)
            {
                last = tooLong;
            }
        }
        throw last ?? new FishAudioDrama3PromptTooLongException(0, maxRows);
    }

    private void Text(List<int[]> rows, string literal)
    {
        foreach (int id in _encode(literal.Normalize(NormalizationForm.FormC)))
        {
            int[] row = new int[Width];
            row[0] = id;
            rows.Add(row);
        }
    }

    private void Codes(List<int[]> rows, int[,] codes)
    {
        int frames = codes.GetLength(1);
        for (int f = 0; f < frames; f++)
        {
            int[] row = new int[Width];
            row[0] = _config.SemanticStartId + codes[0, f];
            for (int c = 0; c < _config.NumCodebooks; c++) row[1 + c] = codes[c, f];
            rows.Add(row);
        }
    }

    private void CheckCodes(int[,] codes, string location)
    {
        ArgumentNullException.ThrowIfNull(codes);
        if (codes.GetLength(0) != _config.NumCodebooks || codes.GetLength(1) == 0)
        {
            throw new ArgumentException(
                $"Drama 3 {location}: codes must be [{_config.NumCodebooks}, frames ≥ 1], not [{codes.GetLength(0)}, {codes.GetLength(1)}].");
        }
        foreach (int code in codes)
        {
            if ((uint)code >= (uint)_config.CodebookSize)
            {
                throw new ArgumentException($"Drama 3 {location}: codes must lie in [0, {_config.CodebookSize}).");
            }
        }
    }
}
