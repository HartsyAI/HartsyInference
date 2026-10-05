using System.Text;
using HartsyInference.Audio.Models.QwenOmni;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>Builds the Qwen2.5-Omni ChatML conditioning prompt AuK feeds its thinker: instruction text followed by an optional audio block.</summary>
/// <remarks>Text segments go through the caller's byte-level encoder and the control tokens are spliced as ids, so the instruction is never scanned for special-token literals.</remarks>
public static class AukPrompt
{
    /// <summary>Marker appended to a text-only (no reference audio) instruction.</summary>
    public const string NoPromptAudioMarker = "|<no_prompt_audio>|";

    /// <summary>Default system turn of the Qwen2.5-Omni chat template.</summary>
    public const string SystemPrompt = "You are a helpful assistant.";

    /// <summary>Token id of <c>&lt;|im_start|&gt;</c>.</summary>
    public const int ImStartId = 151_644;

    /// <summary>Token id of <c>&lt;|im_end|&gt;</c>.</summary>
    public const int ImEndId = 151_645;

    private readonly record struct Segment(string? Text, int Id, string Literal, int Count);

    /// <summary>The prompt as text with <c>&lt;|AUDIO|&gt;</c> repeated <paramref name="audioTokens"/> times; zero omits the audio block and appends <see cref="NoPromptAudioMarker"/> to the instruction.</summary>
    public static string BuildText(string instruction, int audioTokens, QwenOmniConfig? cfg = null)
    {
        StringBuilder sb = new(instruction.Length + 128 + audioTokens * 9);
        foreach (Segment s in Segments(instruction, audioTokens, cfg ?? QwenOmniConfig.Default))
        {
            if (s.Text is not null) sb.Append(s.Text);
            else for (int i = 0; i < s.Count; i++) sb.Append(s.Literal);
        }
        return sb.ToString();
    }

    /// <summary>The prompt as token ids; <paramref name="encode"/> must be byte-level exact (<c>GgufTokenizer.EncodeOrdinary</c> or <c>Qwen2Tokenizer.EncodeRawByteLevel</c>).</summary>
    public static int[] BuildIds(string instruction, int audioTokens, Func<string, IReadOnlyList<int>> encode, QwenOmniConfig? cfg = null)
    {
        ArgumentNullException.ThrowIfNull(encode);
        List<int> ids = new(64 + audioTokens);
        foreach (Segment s in Segments(instruction, audioTokens, cfg ?? QwenOmniConfig.Default))
        {
            if (s.Text is not null) ids.AddRange(encode(s.Text));
            else for (int i = 0; i < s.Count; i++) ids.Add(s.Id);
        }
        return [.. ids];
    }

    private static List<Segment> Segments(string instruction, int audioTokens, QwenOmniConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(instruction);
        if (audioTokens < 0) throw new ArgumentOutOfRangeException(nameof(audioTokens));
        string text = instruction;
        if (audioTokens == 0 && !text.EndsWith(NoPromptAudioMarker, StringComparison.Ordinal)) text += NoPromptAudioMarker;

        List<Segment> segs =
        [
            new(null, ImStartId, "<|im_start|>", 1),
            new($"system\n{SystemPrompt}", 0, "", 0),
            new(null, ImEndId, "<|im_end|>", 1),
            new("\n", 0, "", 0),
            new(null, ImStartId, "<|im_start|>", 1),
            new("user\n" + text, 0, "", 0),
        ];
        if (audioTokens > 0)
        {
            segs.Add(new(null, cfg.AudioBosTokenId, "<|audio_bos|>", 1));
            segs.Add(new(null, cfg.AudioTokenId, "<|AUDIO|>", audioTokens));
            segs.Add(new(null, cfg.AudioEosTokenId, "<|audio_eos|>", 1));
        }
        segs.Add(new(null, ImEndId, "<|im_end|>", 1));
        segs.Add(new("\n", 0, "", 0));
        segs.Add(new(null, ImStartId, "<|im_start|>", 1));
        segs.Add(new("assistant\n", 0, "", 0));
        return segs;
    }
}
