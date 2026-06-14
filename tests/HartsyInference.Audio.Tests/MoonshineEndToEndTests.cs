using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end Moonshine integration test. Same shape as Whisper E2E: gated on the
/// cache being populated, opt-in network download via <c>HARTSYINFERENCE_NETWORK_TESTS=1</c>.
///
/// <para>Validates the full audio path through a fundamentally different architecture than
/// Whisper: raw waveform (no mel), 3-layer Conv1D stem, RoPE-encoder, RoPE-decoder with
/// cross-attention + SwiGLU MLP, weight-tied LM head. If this passes, our infrastructure
/// (HF cache, safetensors loader, IBackend Conv2D dispatch, multi-head reshape, KV cache,
/// SentencePiece byte-fallback BPE decode) is validated end-to-end on a second model
/// family.</para></summary>
public sealed class MoonshineEndToEndTests
{
    private const string JfkClipUrl =
        "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav";

    private static bool AllowNetwork =>
        Environment.GetEnvironmentVariable("HARTSYINFERENCE_NETWORK_TESTS") == "1";

    [Fact]
    [Trait("Category", "Integration")]
    public async Task Transcribe_Jfk_OnMoonshineBase_MatchesCanonicalText()
    {
        string repoDir = AudioModelCache.GetRepoDirectory("UsefulSensors/moonshine-base");
        bool cached = File.Exists(Path.Combine(repoDir, "model.safetensors"))
            && File.Exists(Path.Combine(repoDir, "tokenizer.json"));
        if (!cached && !AllowNetwork) return;

        string clipPath = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        if (!File.Exists(clipPath))
        {
            if (!AllowNetwork) return;
            Directory.CreateDirectory(Path.GetDirectoryName(clipPath)!);
            using HttpClient http = new();
            byte[] bytes = await http.GetByteArrayAsync(JfkClipUrl);
            await File.WriteAllBytesAsync(clipPath, bytes);
        }

        using MoonshinePipeline pipeline = await MoonshinePipeline.LoadAsync("UsefulSensors/moonshine-base");
        using CpuBackend backend = new();
        string text = pipeline.TranscribeWav(backend, clipPath);

        // Moonshine produces no commas / capitalization variation on this clip — text is
        // remarkably stable across runs. Substring checks keep the test robust to any
        // future minor changes.
        string lower = text.ToLowerInvariant();
        Assert.Contains("fellow americans", lower);
        Assert.Contains("country", lower);
        Assert.Contains("ask not", lower);
    }
}
