namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Fish-Speech prompt builders for the layouts the engine runs.</summary>
public static class FishSpeechPrompt
{
    /// <summary>The fish-speech-1.4 prompt (upstream <c>encode_tokens</c>, v1.4.3): the cleaned text as a user turn, then the
    /// open assistant turn. The <c>&lt;|semantic|&gt;</c> row is generated, not prefilled.</summary>
    public static string Plain14(string text) =>
        $"{FishSpeechTokenizer.ImStart}user\n{FishSpeechTextCleaner.Clean(text)}{FishSpeechTokenizer.ImEnd}"
        + $"{FishSpeechTokenizer.ImStart}assistant\n";
}
