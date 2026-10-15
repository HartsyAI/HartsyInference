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
/// directory whose <c>manifest.json</c> records the weights directory, voice clip, text, seed and step count. Every
/// stage in the manifest must be traced by the engine and compared; a missing or unsupported stage fails the test.</summary>
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

        VibeVoiceProcessor processor = new(new VibeVoiceTokenizer());
        VibeVoiceProcessor.PreparedPrompt prep = processor.Prepare([text], [voiceWav]);

        JsonElement stages = root.GetProperty("stages");
        int stageCount = stages.GetArrayLength();
        List<string> report = new();
        List<string> failures = new();
        int compared = 0;

        foreach (JsonElement stage in stages.EnumerateArray())
        {
            string name = stage.GetProperty("name").GetString()!;
            string dtype = stage.GetProperty("dtype").GetString()!;
            string file = Path.Combine(dir, stage.GetProperty("file").GetString()!);
            if (name == "tokens" && dtype == "i32")
            {
                int[] expected = ReadI32(file);
                bool same = expected.SequenceEqual(prep.TokenIds);
                report.Add($"{name,-20} n={expected.Length,-7} token ids {(same ? "ok" : "DIVERGES")}");
                if (!same)
                {
                    failures.Add($"{name}: engine prompt ids differ from upstream");
                    report.Add($"  upstream: {string.Join(' ', expected)}");
                    report.Add($"  engine:   {string.Join(' ', prep.TokenIds)}");
                }
                compared++;
            }
            else if (name == "speech_mask" && dtype == "u8")
            {
                byte[] expected = File.ReadAllBytes(file);
                bool same = expected.Select(b => b != 0).SequenceEqual(prep.SpeechInputMask);
                report.Add($"{name,-20} n={expected.Length,-7} speech mask {(same ? "ok" : "DIVERGES")}");
                if (!same)
                {
                    failures.Add($"{name}: engine speech mask differs from upstream");
                }
                compared++;
            }
            else if (dtype == "f32")
            {
                if (!engine.TryGetValue(name, out float[]? actual))
                {
                    failures.Add($"{name}: dumped by upstream but not traced by SynthesizeTraced");
                    report.Add($"{name,-20} NOT TRACED");
                    continue;
                }
                float[] upstream = ReadF32(file);
                if (upstream.Length != actual.Length)
                {
                    failures.Add($"{name}: length upstream={upstream.Length} engine={actual.Length}");
                    report.Add($"{name,-20} length upstream={upstream.Length} engine={actual.Length} DIVERGES");
                    compared++;
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
                if (!ok)
                {
                    failures.Add($"{name}: relative error {rel:E2} exceeds {RelativeTolerance:E0}");
                }
                report.Add($"{name,-20} n={upstream.Length,-7} maxAbsDiff={maxAbs,11:E3} maxAbsRef={maxRef,11:E3} " +
                           $"rel={rel,10:E3} {(ok ? "ok" : "DIVERGES")}");
                compared++;
            }
            else
            {
                failures.Add($"{name}: unsupported dtype {dtype} for a dumped stage");
                report.Add($"{name,-20} UNSUPPORTED DTYPE {dtype}");
            }
        }

        foreach (string line in report)
        {
            output.WriteLine(line);
        }
        output.WriteLine($"compared {compared} of {stageCount} manifest stages; engine traced {engine.Count} stages");

        int expectedFrames = (int)Math.Round(root.GetProperty("frames_generated").GetDouble());
        int engineFrames = engine.Keys.Count(key => key.StartsWith("frame_", StringComparison.Ordinal));
        if (engineFrames != expectedFrames)
        {
            failures.Add($"frames: engine emitted {engineFrames}, upstream emitted {expectedFrames}");
        }
        if (compared != stageCount)
        {
            failures.Add($"only {compared} of {stageCount} manifest stages were compared");
        }
        Assert.True(failures.Count == 0, "parity failures, earliest first:\n  " + string.Join("\n  ", failures));
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
}
