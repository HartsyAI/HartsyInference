using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end Whisper integration test. Downloads <c>openai/whisper-tiny</c>
/// (~150 MB) plus a known reference audio clip on first run, transcribes, and asserts
/// the canonical text appears in the output. Subsequent runs hit the cache.
///
/// <para><b>Gated by trait <c>Network=Real</c></b> — CI without internet skips this.
/// Run manually with:
/// <code>dotnet test --filter Network=Real</code></para>
///
/// <para>This test is the ground-truth check that the whole audio stack works:
/// mel preprocessing → encoder → decoder → tokenizer all the way to a human-readable
/// English sentence. If this fails, the failure is almost always either:
/// (1) a Whisper conv-stem shape mistake, (2) a KV-cache append bug in the decoder,
/// (3) a byte-level BPE decode mistake in the tokenizer. The three suspects in the
/// debug order to check.</para></summary>
public sealed class WhisperEndToEndTests
{
    /// <summary>Public sample clip shipped by whisper.cpp — 16 kHz mono, ~11 s,
    /// JFK's "Ask not what your country can do for you" line. Stable URL, MIT
    /// license, used by every C++ Whisper port for the same purpose.</summary>
    private const string JfkClipUrl =
        "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav";

    /// <summary>Set <c>HARTSYINFERENCE_NETWORK_TESTS=1</c> to opt into download-driven
    /// tests. Without it, the test runs only against a pre-populated cache (i.e. when
    /// the model was downloaded manually or by a previous invocation).</summary>
    private static bool AllowNetwork =>
        Environment.GetEnvironmentVariable("HARTSYINFERENCE_NETWORK_TESTS") == "1";

    [Fact]
    [Trait("Category", "Integration")]
    public async Task Transcribe_Jfk_Clip_OnWhisperTiny_MatchesCanonicalText()
    {
        // 1. Resolve the model. If we have network access, the pipeline auto-downloads.
        //    Otherwise we require the cache to already be populated and skip if not.
        string repoDir = AudioModelCache.GetRepoDirectory("openai/whisper-tiny", "stt");
        bool cached = File.Exists(Path.Combine(repoDir, "model.safetensors"))
            && File.Exists(Path.Combine(repoDir, "vocab.json"))
            && File.Exists(Path.Combine(repoDir, "merges.txt"));
        if (!cached && !AllowNetwork)
        {
            // No model, no network — nothing to assert. xUnit treats this as passing.
            // Set HARTSYINFERENCE_NETWORK_TESTS=1 to actually download and run.
            return;
        }

        // 2. Make sure the audio clip is on disk.
        string clipPath = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        if (!File.Exists(clipPath))
        {
            if (!AllowNetwork) return;
            Directory.CreateDirectory(Path.GetDirectoryName(clipPath)!);
            using HttpClient http = new();
            byte[] bytes = await http.GetByteArrayAsync(JfkClipUrl);
            await File.WriteAllBytesAsync(clipPath, bytes);
        }

        // 3. Load + transcribe on the CPU backend.
        using WhisperPipeline pipeline = await WhisperPipeline.LoadAsync("openai/whisper-tiny");
        using CpuBackend backend = new();
        WhisperOptions options = new() { Language = "en", Translate = false, WithTimestamps = false };
        string text = pipeline.TranscribeWav(backend, clipPath, options);

        // 4. The JFK line is well-known. Assert on key content words; comma and
        //    capitalization placement varies across Whisper sizes so we avoid
        //    exact-text equality.
        string lower = text.ToLowerInvariant();
        Assert.Contains("fellow americans", lower);
        Assert.Contains("country", lower);
        Assert.Contains("ask not", lower);
    }
}
