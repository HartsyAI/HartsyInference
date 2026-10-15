using System.Runtime.InteropServices;
using System.Text.Json;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>VibeVoice-1.5B first-frame parity on real weights (CPU F32): the engine's generation stages against
/// upstream fp32 dumps from <c>tests/python-reference/vibevoice_first_frame_dump.py --native-prompt</c>, run with the
/// same prompt, voice prompt and per-frame diffusion noise. Needs <c>HARTSY_VIBEVOICE_PARITY_DIR</c> naming a dump
/// directory whose <c>manifest.json</c> records the weights directory, voice clip, text, seed and step count.</summary>
[Trait("Category", "Integration")]
public sealed class VibeVoiceFirstFrameParityTests(ITestOutputHelper output)
{
    // Upstream fp32 run with 8 threads vs 1 thread differs by 1.8e-4 relative on frame 5's decoded audio, and the
    // difference compounds through the autoregressive feedback loop. 2e-3 is about ten times that floor.
    private const double RelativeTolerance = 2e-3;

    [Fact]
    public void FirstFrameStages_MatchUpstreamFp32Cpu()
    {
        string? dir = Environment.GetEnvironmentVariable("HARTSY_VIBEVOICE_PARITY_DIR");
        string manifestPath = Path.Combine(dir ?? string.Empty, "manifest.json");
        if (string.IsNullOrEmpty(dir) || !File.Exists(manifestPath))
        {
            Assert.False(Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1",
                "HARTSY_VIBEVOICE_PARITY_DIR is not set to an upstream VibeVoice stage dump directory.");
            return;
        }

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = doc.RootElement;
        string weights = root.GetProperty("weights").GetString()!;
        string voiceWav = root.GetProperty("voice_wav").GetString()!;
        string text = root.GetProperty("text").GetString()!;
        int seed = root.GetProperty("seed").GetInt32();
        int steps = root.GetProperty("steps").GetInt32();
        double cfgScale = root.GetProperty("cfg_scale").GetDouble();

        Dictionary<string, float[]> engine = new(StringComparer.Ordinal);
        using (CpuBackend backend = new())
        {
            VibeVoicePipeline pipeline = VibeVoicePipeline.LoadFromDirectory(weights);
            pipeline.SynthesizeTraced(backend, [text], [voiceWav], steps, seed, cfgScale, 10,
                (name, values) => engine.TryAdd(name, values.ToArray()));
            pipeline.Dispose();
        }

        List<string> report = new();
        string? firstDivergent = null;
        VibeVoiceProcessor processor = new(new VibeVoiceTokenizer());
        PreparedPromptRef prompt = PreparedPromptRef.From(processor.Prepare([text], [voiceWav]));

        foreach (JsonElement stage in root.GetProperty("stages").EnumerateArray())
        {
            string name = stage.GetProperty("name").GetString()!;
            string dtype = stage.GetProperty("dtype").GetString()!;
            string file = Path.Combine(dir, stage.GetProperty("file").GetString()!);
            if (dtype == "i32")
            {
                int[] expected = ReadI32(file);
                bool same = name == "tokens" && expected.SequenceEqual(prompt.TokenIds);
                if (!same && firstDivergent is null) firstDivergent = name;
                report.Add($"{name,-20} n={expected.Length,-7} token ids {(same ? "ok" : "DIVERGES")}");
                if (!same)
                {
                    report.Add($"  upstream: {string.Join(' ', expected)}");
                    report.Add($"  engine:   {string.Join(' ', prompt.TokenIds)}");
                }
                continue;
            }
            if (dtype == "u8")
            {
                byte[] expected = File.ReadAllBytes(file);
                bool same = name == "speech_mask" && expected.Select(b => b != 0).SequenceEqual(prompt.SpeechInputMask);
                if (!same && firstDivergent is null) firstDivergent = name;
                report.Add($"{name,-20} n={expected.Length,-7} speech mask {(same ? "ok" : "DIVERGES")}");
                continue;
            }
            if (!engine.TryGetValue(name, out float[]? actual))
            {
                report.Add($"{name,-20} (engine does not record this stage)");
                continue;
            }
            float[] upstream = ReadF32(file);
            if (upstream.Length != actual.Length)
            {
                if (firstDivergent is null) firstDivergent = name;
                report.Add($"{name,-20} length upstream={upstream.Length} engine={actual.Length} DIVERGES");
                continue;
            }
            double maxAbs = 0;
            double maxRef = 0;
            for (int i = 0; i < upstream.Length; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs((double)actual[i] - upstream[i]));
                maxRef = Math.Max(maxRef, Math.Abs((double)upstream[i]));
            }
            double rel = maxAbs / Math.Max(maxRef, 1e-12);
            bool ok = rel <= RelativeTolerance;
            if (!ok && firstDivergent is null) firstDivergent = name;
            report.Add($"{name,-20} n={upstream.Length,-7} maxAbsDiff={maxAbs,11:E3} maxAbsRef={maxRef,11:E3} " +
                       $"rel={rel,10:E3} {(ok ? "ok" : "DIVERGES")}");
        }

        foreach (string line in report)
        {
            output.WriteLine(line);
        }
        int expectedFrames = (int)Math.Round(root.GetProperty("frames_generated").GetDouble());
        int engineFrames = engine.Keys.Count(key => key.StartsWith("frame_", StringComparison.Ordinal));
        Assert.Equal(expectedFrames, engineFrames);
        Assert.True(firstDivergent is null, $"first divergent stage: {firstDivergent}");
    }

    private static float[] ReadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    private static int[] ReadI32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        return MemoryMarshal.Cast<byte, int>(bytes).ToArray();
    }

    /// <summary>Copies the prompt the engine builds for this text, so token ids can be checked against upstream.</summary>
    private sealed record PreparedPromptRef(int[] TokenIds, bool[] SpeechInputMask)
    {
        public static PreparedPromptRef From(VibeVoiceProcessor.PreparedPrompt prep) =>
            new(prep.TokenIds, prep.SpeechInputMask);
    }
}
