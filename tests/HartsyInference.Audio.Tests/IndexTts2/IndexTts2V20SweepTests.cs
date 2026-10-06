using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Writes one WAV per sentence of a JSON list through the IndexTTS-2.0 pipeline and prints the per-stage
/// timings — the harness behind the Whisper quality sweep and the CPU speed measurements. Gated on
/// <c>INDEXTTS2_SWEEP_JSON</c> (a JSON string array) and <c>INDEXTTS2_SWEEP_OUT</c> plus the usual real-weight
/// environment; optional <c>INDEXTTS2_SWEEP_SEED</c> (default 11), <c>INDEXTTS2_SWEEP_TAG</c> and
/// <c>INDEXTTS2_SWEEP_LIMIT</c>.</summary>
[Trait("Category", "Integration")]
public sealed class IndexTts2V20SweepTests(ITestOutputHelper output)
{
    [Fact]
    public async Task Sweep_WritesWavsAndTimings()
    {
        string? json = Environment.GetEnvironmentVariable("INDEXTTS2_SWEEP_JSON");
        string? outDir = Environment.GetEnvironmentVariable("INDEXTTS2_SWEEP_OUT");
        string? v20Dir = Environment.GetEnvironmentVariable("INDEXTTS2_V20_DIR");
        string? codecPath = Environment.GetEnvironmentVariable("INDEXTTS2_MASKGCT_CODEC_PATH");
        string? w2vBertPath = Environment.GetEnvironmentVariable("INDEXTTS2_W2VBERT_SAFETENSORS_PATH");
        string? campplusPath = Environment.GetEnvironmentVariable("INDEXTTS2_CAMPPLUS_PATH");
        string? bigVganPath = Environment.GetEnvironmentVariable("INDEXTTS2_BIGVGAN_PT_PATH");
        string? refWavPath = Environment.GetEnvironmentVariable("INDEXTTS2_REF_WAV");
        if (string.IsNullOrEmpty(json) || !File.Exists(json) || string.IsNullOrEmpty(outDir)) { output.WriteLine("sweep env missing — skip."); return; }
        if (string.IsNullOrEmpty(v20Dir) || !Directory.Exists(v20Dir)) { output.WriteLine("INDEXTTS2_V20_DIR missing — skip."); return; }
        foreach (string? p in new[] { codecPath, w2vBertPath, campplusPath, bigVganPath, refWavPath })
            if (string.IsNullOrEmpty(p) || !File.Exists(p)) { output.WriteLine("auxiliary checkpoint missing — skip."); return; }

        string[] sentences = JsonSerializer.Deserialize<string[]>(File.ReadAllText(json)) ?? [];
        int limit = int.TryParse(Environment.GetEnvironmentVariable("INDEXTTS2_SWEEP_LIMIT"), out int l) ? l : sentences.Length;
        ulong seedBase = ulong.TryParse(Environment.GetEnvironmentVariable("INDEXTTS2_SWEEP_SEED"), out ulong sb) ? sb : 11;
        string tag = Environment.GetEnvironmentVariable("INDEXTTS2_SWEEP_TAG") ?? "cs";
        Directory.CreateDirectory(outDir);

        using CpuBackend backend = new();
        using IndexTts2Pipeline pipeline = await IndexTts2Pipeline.LoadAsync(
            Path.Combine(v20Dir, "bpe.model"), Path.Combine(v20Dir, "gpt.pth"), Path.Combine(v20Dir, "s2mel.pth"), codecPath!,
            w2vBertPath!, Path.Combine(v20Dir, "wav2vec2bert_stats.pt"), campplusPath!, bigVganPath!,
            feat1Path: Path.Combine(v20Dir, "feat1.pt"), feat2Path: Path.Combine(v20Dir, "feat2.pt"), cfg: IndexTts2Config.V2_0);

        WavFile.DecodedAudio refAudio = WavFile.Read(refWavPath!);
        IndexTts2Timings refTimings = new();
        using IndexTts2Reference reference = pipeline.PrepareReference(backend, refAudio.Channels[0], refAudio.SampleRate, refTimings);
        output.WriteLine($"reference prepared in {refTimings.ReferenceMs:F0} ms ({reference.Seconds:F1}s clip)");
        File.AppendAllText(Path.Combine(outDir, $"timings_{tag}.txt"), $"reference {refTimings.ReferenceMs:F0} ms{Environment.NewLine}");

        for (int i = 0; i < Math.Min(limit, sentences.Length); i++)
        {
            IndexTts2Timings t = new();
            float[] pcm = pipeline.Synthesize(backend, sentences[i], reference, new IndexTts2Options { Seed = seedBase + (ulong)i, Timings = t });
            string path = Path.Combine(outDir, $"s{i}_{tag}.wav");
            WavFile.WriteMono16(path, pcm, 22_050);
            output.WriteLine($"[{i}] {t}");
            File.AppendAllText(Path.Combine(outDir, $"timings_{tag}.txt"), $"[{i}] {t}{Environment.NewLine}");
        }
    }
}
