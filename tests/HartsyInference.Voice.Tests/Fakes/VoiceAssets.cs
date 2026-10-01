using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Streaming;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>Real-weight test inputs resolved the way the engine resolves them (the audio cache for Whisper and Kokoro, the
/// test models root for the wake folder) plus content-word recall scoring.</summary>
internal static class VoiceAssets
{
    /// <summary>The JFK line, minus "and so, my": the words recall is scored on.</summary>
    public static readonly string[] JfkWords = ["fellow", "americans", "ask", "not", "what", "your", "country", "can", "do", "for", "you"];

    public static string WakeRoot => Path.Combine(TestPaths.ModelsDir, "audio", "wake");

    public static string SileroWeights => Path.Combine(WakeRoot, "vad", "silero_vad_16k.safetensors");

    public static string RnnoiseWeights => Path.Combine(WakeRoot, "denoise", "rnnoise.safetensors");

    public static string Jfk => Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");

    /// <summary>Files a Whisper checkpoint needs in the audio cache.</summary>
    public static string[] WhisperFiles(string repo)
    {
        string dir = AudioModelCache.GetRepoDirectory(repo, "stt");
        return [Path.Combine(dir, "model.safetensors"), Path.Combine(dir, "added_tokens.json")];
    }

    /// <summary>Files Kokoro with the af_heart voice needs.</summary>
    public static string[] KokoroFiles()
    {
        string dir = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M", "tts");
        return [Path.Combine(dir, "config.json"), Path.Combine(dir, "voices", "af_heart.bin"), Path.Combine(RepoPaths.ModelsRoot(), "audio", "cmudict.dict")];
    }

    /// <summary>The JFK clip at 16 kHz mono, ±1.</summary>
    public static float[] Jfk16k() => AudioClipCodec.DecodeMono(new AudioClip { Data = File.ReadAllBytes(Jfk), Format = "wav" }, 16_000);

    /// <summary><paramref name="audio"/> through the gateway's narrowband path: 16 kHz → 8 kHz → 16 kHz with the streaming
    /// resamplers the gateway uses, in 20 ms frames.</summary>
    public static float[] Narrowband(float[] audio)
    {
        StreamingResampler down = new(16_000, 8_000, 320);
        StreamingResampler up = new(8_000, 16_000, 160);
        int frames = audio.Length / 320;
        float[] result = new float[frames * 320];
        float[] input = new float[320];
        float[] narrow = new float[160];
        float[] wide = new float[320];
        for (int f = 0; f < frames; f++)
        {
            Array.Copy(audio, f * 320, input, 0, 320);
            down.Process(input, narrow);
            up.Process(narrow, wide);
            wide.CopyTo(result, f * 320);
        }
        return result;
    }

    /// <summary>Share of <paramref name="expected"/> words present, as whole words, in <paramref name="heard"/>.</summary>
    public static double Recall(string heard, IReadOnlyCollection<string> expected)
    {
        HashSet<string> words = new(Words(heard), StringComparer.Ordinal);
        return expected.Count(words.Contains) / (double)expected.Count;
    }

    /// <summary>Lower-cased words of <paramref name="text"/>, letters and digits only.</summary>
    public static IEnumerable<string> Words(string text) =>
        new string([.. text.Select(c => char.IsLetterOrDigit(c) ? char.ToLowerInvariant(c) : ' ')]).Split(' ', StringSplitOptions.RemoveEmptyEntries);
}
