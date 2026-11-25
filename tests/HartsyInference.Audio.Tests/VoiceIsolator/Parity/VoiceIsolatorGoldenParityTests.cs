using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.VoiceIsolator;
using HartsyInference.Core.Backends.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.VoiceIsolator.Parity;

/// <summary>Parity of the released checkpoints against the golden renders shipped in <c>inworld-ai/voice-isolator</c> @ c73b3fab
/// (<c>tests/fixtures/*</c>, made by the training-side reference on CPU in fp32). <c>VOICE_ISOLATOR_DIR</c> names a folder holding
/// <c>denoiser.safetensors</c>, <c>denoiser_realtime.safetensors</c> and <c>fixtures/&lt;case&gt;/{meta.json,input.wav,expected.npy}</c>.
/// Without it each test is reported as skipped; <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> makes a missing folder a failure.</summary>
[Trait("Category", "Integration")]
public sealed class VoiceIsolatorGoldenParityTests(ITestOutputHelper output)
{
    [RealWeightsFact("VOICE_ISOLATOR_DIR")]
    public void Offline3sMatchesTheReference() => Check("offline_3s");

    [RealWeightsFact("VOICE_ISOLATOR_DIR")]
    public void Offline11sMatchesTheReference() => Check("offline_11s");

    [RealWeightsFact("VOICE_ISOLATOR_DIR")]
    public void RealtimeStream3sMatchesTheReference() => Check("realtime_stream_3s");

    [RealWeightsFact("VOICE_ISOLATOR_DIR")]
    public void RealtimeStream11sMatchesTheReference() => Check("realtime_stream_11s");

    private void Check(string name)
    {
        string? root = Environment.GetEnvironmentVariable("VOICE_ISOLATOR_DIR");
        string caseDir = root is null ? string.Empty : Path.Combine(root, "fixtures", name);
        if (root is null || !File.Exists(Path.Combine(caseDir, "meta.json")))
        {
            Assert.Fail($"VOICE_ISOLATOR_DIR does not hold fixtures/{name}/meta.json; nothing was validated.");
            return;
        }
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(caseDir, "meta.json")));
        string model = Path.Combine(root, Path.GetFileName(meta.RootElement.GetProperty("model").GetString()!));
        Assert.Equal(meta.RootElement.GetProperty("model_sha256").GetString(), Sha256(model));
        Assert.Equal(meta.RootElement.GetProperty("input_sha256").GetString(), Sha256(Path.Combine(caseDir, "input.wav")));
        Assert.Equal(meta.RootElement.GetProperty("expected_sha256").GetString(), Sha256(Path.Combine(caseDir, "expected.npy")));
        double tolerance = meta.RootElement.GetProperty("tolerance").GetDouble();
        float[] input = WavFile.Read(Path.Combine(caseDir, "input.wav")).ToMono();
        float[] expected = NpyFloat32.Read(Path.Combine(caseDir, "expected.npy"));

        using VoiceIsolatorNetwork network = VoiceIsolatorNetwork.Load(model);
        using CpuBackend backend = new();
        long started = Environment.TickCount64;
        float[] actual = meta.RootElement.TryGetProperty("runner", out JsonElement runner)
            ? Stream(network, backend, input, runner, meta.RootElement.GetProperty("lookahead_frames").GetInt32())
            : VoiceIsolatorRenderer.Render(network, backend, input,
                meta.RootElement.GetProperty("chunk_seconds").GetDouble(), meta.RootElement.GetProperty("overlap_seconds").GetDouble());
        long elapsed = Environment.TickCount64 - started;

        Assert.Equal(expected.Length, actual.Length);
        double maxAbs = 0;
        double squares = 0;
        double reference = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
            squares += d * d;
            reference += (double)expected[i] * expected[i];
        }
        output.WriteLine($"{name}: {expected.Length} samples, max abs error {maxAbs:E3} (tolerance {tolerance:E1}), "
            + $"relRMS {Math.Sqrt(squares / reference):E3}, {elapsed} ms on CPU");
        Assert.True(maxAbs <= tolerance, $"{name}: max abs error {maxAbs:E3} exceeds {tolerance:E1}");
    }

    /// <summary>The reference's stateful runner: its block, cache and window settings, and the piece sizes it pushed, cycling.</summary>
    private static float[] Stream(VoiceIsolatorNetwork network, CpuBackend backend, float[] input, JsonElement runner, int lookahead)
    {
        int block = runner.GetProperty("block_frames").GetInt32();
        int cache = runner.GetProperty("cache_frames").GetInt32();
        int window = runner.GetProperty("window_frames").GetInt32();
        int crossfade = runner.GetProperty("crossfade_frames").GetInt32();
        int[] pushes = runner.GetProperty("push").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        using VoiceIsolatorStream stream = new(network, backend, block, cache, window, crossfade, lookahead);
        List<float> output = [];
        int position = 0;
        for (int index = 0; position < input.Length; index++)
        {
            int piece = Math.Min(pushes[index % pushes.Length], input.Length - position);
            output.AddRange(stream.Push(input.AsSpan(position, piece)));
            position += piece;
        }
        output.AddRange(stream.Flush());
        return [.. output];
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}
