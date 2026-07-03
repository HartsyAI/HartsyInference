using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.VibeVoice;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Prompt-builder tests for VibeVoice — these need no model weights, only the
/// embedded Qwen2.5 BPE vocab. They lock in the fix for the "fluent but wrong words"
/// hallucination: the target script text MUST appear, tokenized exactly like Qwen2.5,
/// as a contiguous non-speech span inside the LM input the pipeline prefills. Before the
/// byte-level-BPE fix, <see cref="VibeVoiceTokenizer.Encode"/> dropped leading spaces
/// (" Speaker" → "Speaker"), so the LM never saw the real words and free-ran.</summary>
public sealed class VibeVoiceProcessorTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly string _tmpDir;

    public VibeVoiceProcessorTests(ITestOutputHelper output)
    {
        _out = output;
        _tmpDir = Path.Combine(Path.GetTempPath(), "vv_proc_tests_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tmpDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tmpDir, recursive: true); } catch { /* best effort */ }
    }

    /// <summary>Word-bearing text must tokenize identically to HuggingFace
    /// <c>Qwen/Qwen2.5-1.5B</c>. This is the guard against the "fluent but wrong words"
    /// regression: before the byte-level fix every leading space was dropped
    /// (" Speaker"→"Speaker", "Ġworld"→"world"), producing wrong IDs for every word.
    ///
    /// <para><b>Known caveat:</b> ML.Tokenizers' BpeTokenizer splits at the byte-level
    /// newline boundary, so a trailing "<c>:\n</c>" or "<c>.\n</c>" emits two tokens
    /// (<c>:</c>+<c>Ċ</c>) where Qwen2.5's fast/slow tokenizer applies the <c>: Ċ</c>
    /// merge (id 510). That is a line-ending-only difference — never a word — and is
    /// the same limitation the parity-verified <c>Qwen3Tokenizer.EncodeRaw</c> carries.
    /// Assertions below use no-trailing-newline strings so they pin exact word parity.</para></summary>
    [Fact]
    public void Tokenizer_MatchesQwen25_ByteLevelIds()
    {
        using VibeVoiceTokenizer tok = new();
        // Exact Qwen2.5 IDs — the words that must condition the LM.
        Assert.Equal(
            new[] { 29073, 220, 15, 25, 21927, 1879, 11, 419, 374, 264, 1273, 315, 279, 28431, 13 },
            tok.Encode(" Speaker 0: Hello world, this is a test of the benchmark."));
        Assert.Equal(new[] { 2918, 1946 }, tok.Encode(" Text input"));   // ĠText Ġinput
        Assert.Equal(new[] { 27930, 1946 }, tok.Encode(" Voice input")); // ĠVoice Ġinput
        Assert.Equal(new[] { 38741, 2550 }, tok.Encode(" Speech output")); // ĠSpeech Ġoutput
        // Pre-fix, a leading space was dropped, so " world" collided with the word-start
        // token "world" (1879 ≠ the standalone id). Confirm the space is honored.
        Assert.Equal(new[] { 1879 }, tok.Encode(" world"));  // Ġworld
        Assert.NotEqual(new[] { 1879 }, tok.Encode("world")); // world (no leading space)
        _out.WriteLine("Tokenizer byte-level word IDs match Qwen2.5 reference.");
    }

    [Fact]
    public void Prepare_PlacesTargetTextAsContiguousNonSpeechSpan()
    {
        using VibeVoiceTokenizer tok = new();
        VibeVoiceProcessor proc = new(tok);

        string wavPath = WriteSyntheticWav(Path.Combine(_tmpDir, "ref.wav"), seconds: 3.0);
        string target = "Hello world, this is a test of the benchmark.";

        VibeVoiceProcessor.PreparedPrompt prep = proc.Prepare(new[] { target }, new[] { wavPath });

        // The exact tokens the model must condition on: " Speaker 0:{target}\n" — no space
        // after the colon, matching upstream's f" Speaker {id}:{text}\n" text-input format.
        int[] expected = tok.Encode($" Speaker 0:{target}\n");
        int at = IndexOfSub(prep.TokenIds, expected);
        _out.WriteLine($"Prompt length {prep.TokenIds.Length}; target span at index {at} ({expected.Length} tokens).");
        Assert.True(at >= 0, "target script tokens were not found contiguously in the LM input sequence.");

        // Those positions are text — never speech-input (voice-latent) positions.
        for (int i = 0; i < expected.Length; i++)
            Assert.False(prep.SpeechInputMask[at + i], $"text token at {at + i} was wrongly flagged as speech input.");

        // The target text must sit AFTER the voice section (which carries the speech mask)
        // and BEFORE the trailing speech_start cursor.
        int firstSpeech = Array.IndexOf(prep.SpeechInputMask, true);
        Assert.True(firstSpeech >= 0, "no voice-latent positions present.");
        Assert.True(at > firstSpeech, "target text must come after the voice-input section.");
        Assert.Equal(VibeVoiceTokenizer.SpeechStartTokenId, prep.TokenIds[^1]);
        Assert.False(prep.SpeechInputMask[^1]);

        // Round-trip: decoding the span reads back the human-readable script.
        string decoded = tok.Decode(expected);
        _out.WriteLine($"Decoded target span: {decoded}");
        Assert.Contains("Speaker 0", decoded);
        Assert.Contains("benchmark", decoded);
    }

    private static string WriteSyntheticWav(string path, double seconds)
    {
        int sr = 24_000;
        int n = (int)(sr * seconds);
        float[] pcm = new float[n];
        for (int i = 0; i < n; i++) pcm[i] = 0.1f * MathF.Sin(2 * MathF.PI * 180f * i / sr);
        WavFile.WriteMono16(path, pcm, sr);
        return path;
    }

    private static int IndexOfSub(int[] hay, int[] needle)
    {
        for (int i = 0; i + needle.Length <= hay.Length; i++)
        {
            bool ok = true;
            for (int j = 0; j < needle.Length; j++)
                if (hay[i + j] != needle[j]) { ok = false; break; }
            if (ok) return i;
        }
        return -1;
    }
}
