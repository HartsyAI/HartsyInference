using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>Whisper greedy tokens against an HF transformers reference on <see cref="WhisperParityClips"/>, with and
/// without timestamps, on the CPU backend. The reference
/// (<c>tests/python-reference/whisper_logmel_parity/whisper_reference.py tokens</c>) runs
/// <c>WhisperForConditionalGeneration</c> on <c>WhisperFeatureExtractor</c>'s features with this pipeline's decoding
/// rule — prompt, the suppressed ids, argmax, stop on end-of-text — not <c>generate()</c>, whose suppress lists and
/// timestamp rules the engine does not apply. It records every step's top-1/top-2 margin and runner-up, and where the
/// engine parts from it this test feeds the reference's tokens before that step through the engine's own encoder and
/// decoder, one at a time as the pipeline does, and reports the engine's margin there too: a divergence is placed and
/// weighed on both sides.
///
/// <para>Opt-in: <c>HARTSY_WHISPER_TOKEN_PARITY=1</c> with <c>HARTSYINFERENCE_WHISPER_PARITY_DIR</c>. Models:
/// <c>HARTSY_WHISPER_TOKEN_PARITY_MODELS</c> (comma-separated repo ids; default tiny, base, small.en); clips:
/// <c>HARTSY_WHISPER_TOKEN_PARITY_CLIPS</c> (comma-separated names; default all), to split a long run. Each run writes
/// its ids and text under <c>engine-{label}/</c> (<c>HARTSY_WHISPER_TOKEN_PARITY_LABEL</c>, default <c>engine</c>);
/// <c>HARTSY_WHISPER_TOKEN_PARITY_COMPARE</c> names another run's label (a build of the previous engine) to set beside
/// it. Tables go to the test output and, with <c>HARTSY_WHISPER_TOKEN_PARITY_OUT</c>, are appended to that file.
/// <c>HARTSY_WHISPER_TOKEN_PARITY_EXPLAIN=0</c> skips the teacher-forced explanation. The prompt and the suppressed
/// ids must equal the reference's; token differences are reported, and fail the test only with
/// <c>HARTSY_WHISPER_TOKEN_PARITY_STRICT=1</c>. With Whisper's exact GELU all 144 decodes (six models) match on the CPU
/// backend (2026-10-01, AVX2), but the reference's closest calls are 0.004 logits apart, so another summation order
/// (vector width, backend) can flip one without a bug.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WhisperTokenParityTests(ITestOutputHelper output)
{
    private const string GateEnvVar = "HARTSY_WHISPER_TOKEN_PARITY";
    private const string ModelsEnvVar = "HARTSY_WHISPER_TOKEN_PARITY_MODELS";
    private const string LabelEnvVar = "HARTSY_WHISPER_TOKEN_PARITY_LABEL";
    private const string CompareEnvVar = "HARTSY_WHISPER_TOKEN_PARITY_COMPARE";
    private const string OutEnvVar = "HARTSY_WHISPER_TOKEN_PARITY_OUT";
    private const string StrictEnvVar = "HARTSY_WHISPER_TOKEN_PARITY_STRICT";
    private const string ExplainEnvVar = "HARTSY_WHISPER_TOKEN_PARITY_EXPLAIN";
    private const string ClipsEnvVar = "HARTSY_WHISPER_TOKEN_PARITY_CLIPS";
    private const string DefaultModels = "openai/whisper-tiny,openai/whisper-base,openai/whisper-small.en";
    private const int WindowSamples = 30 * WhisperParityClips.SampleRate;

    /// <summary>The 11 words <c>WhisperEnglishOnlyTests</c> pins (every word after the leading "and so, my").</summary>
    private static readonly string[] JfkWords =
        ["fellow", "americans", "ask", "not", "what", "your", "country", "can", "do", "for", "you"];

    [Fact]
    public async Task GreedyTokens_MatchHfReference()
    {
        string? dir = Environment.GetEnvironmentVariable(WhisperParityClips.DirEnvVar);
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1" || string.IsNullOrEmpty(dir))
        {
            output.WriteLine($"SKIPPED: set {GateEnvVar}=1 and {WhisperParityClips.DirEnvVar}.");
            return;
        }
        IReadOnlyList<(string Name, float[] Audio)> clips = WhisperParityClips.Build();
        if (!File.Exists(Path.Combine(dir, "clips", "clips.tsv")))
        {
            WhisperParityClips.Write(dir, clips);
        }
        string label = Environment.GetEnvironmentVariable(LabelEnvVar) ?? "engine";
        string? compare = Environment.GetEnvironmentVariable(CompareEnvVar);
        HashSet<string>? clipFilter = Environment.GetEnvironmentVariable(ClipsEnvVar) is { Length: > 0 } list
            ? new HashSet<string>(list.Split(',', StringSplitOptions.TrimEntries), StringComparer.Ordinal)
            : null;

        StringBuilder table = new();
        table.AppendLine($"### Whisper greedy tokens vs the HF reference — engine `{label}`, CPU backend"
            + (compare is null ? "" : $", previous engine `{compare}`"));
        table.AppendLine();
        table.AppendLine("| Model | Clip | Mode | HF tokens | vs HF | smallest HF margin | "
            + (compare is null ? "" : "previous vs HF | ") + "JFK words | transcript |");
        table.AppendLine("|---|---|---|---:|---|---:|" + (compare is null ? "" : "---|") + "---:|---|");
        List<string> divergences = [];
        using IBackend backend = new CpuBackend();
        foreach (string repo in Models())
        {
            string repoDir = AudioModelCache.GetRepoDirectory(repo, "stt");
            string[] files = WhisperPipeline.ModelFiles.Where(f => f.Required).Select(f => Path.Combine(repoDir, f.Name)).ToArray();
            if (!RealWeightGate.Require(output.WriteLine, files))
            {
                continue;
            }
            using WhisperPipeline whisper = await WhisperPipeline.LoadAsync(repo);
            using WhisperTokenizer tokenizer = new(repoDir);
            TeacherForcing? teacher = null;
            try
            {
                string safe = repo.Replace('/', '_');
                foreach ((string clip, float[] audio) in clips.Where(c => clipFilter is null || clipFilter.Contains(c.Name)))
                {
                    foreach (bool timestamps in (bool[])[false, true])
                    {
                        string mode = timestamps ? "timestamps" : "notimestamps";
                        WhisperOptions options = new() { Language = whisper.IsMultilingual ? "en" : null, WithTimestamps = timestamps };
                        Stopwatch sw = Stopwatch.StartNew();
                        List<int> ids = whisper.TranscribeTokenIds(backend, audio, WhisperParityClips.SampleRate, options);
                        sw.Stop();
                        string rendered = Render(whisper, tokenizer, ids);
                        string runDir = Path.Combine(dir, "engine-" + label, safe);
                        Directory.CreateDirectory(runDir);
                        File.WriteAllText(Path.Combine(runDir, $"{clip}.{mode}.tokens"), string.Join(",", ids));
                        File.WriteAllText(Path.Combine(runDir, $"{clip}.{mode}.txt"), rendered);
                        output.WriteLine($"{repo} {clip} {mode}: {ids.Count} tokens in {sw.Elapsed.TotalSeconds:F1}s | {rendered}");

                        string hfPath = Path.Combine(dir, "hf", safe, $"{clip}.{mode}.json");
                        string verdict = "no ref", hfCount = "—", hfMargin = "—", previous = "—";
                        if (File.Exists(hfPath))
                        {
                            HfReference hf = HfReference.Read(hfPath);
                            int[] prompt = tokenizer.BuildPromptIds(options.Language, false, timestamps);
                            Assert.True(prompt.SequenceEqual(hf.Prompt), $"{repo} {mode}: prompt [{string.Join(",", prompt)}] "
                                + $"vs the reference's [{string.Join(",", hf.Prompt)}]");
                            int[] suppressed = Suppressed(tokenizer);
                            Assert.True(suppressed.SequenceEqual(hf.Suppressed), $"{repo}: suppressed ids differ from the reference's");
                            int step = FirstDifference(hf.Tokens, ids);
                            verdict = Describe(hf, ids, step);
                            if (step >= 0)
                            {
                                if (Environment.GetEnvironmentVariable(ExplainEnvVar) != "0")
                                {
                                    teacher ??= TeacherForcing.Load(repo, repoDir);
                                    verdict += "; " + teacher.Explain(backend, audio, prompt, hf.Tokens, step, suppressed, tokenizer.EotId);
                                }
                                divergences.Add($"{repo} {clip} {mode}: {verdict}");
                            }
                            hfCount = hf.Tokens.Length.ToString(CultureInfo.InvariantCulture);
                            hfMargin = hf.Margins.Length == 0 ? "—" : hf.Margins.Min().ToString("F3", CultureInfo.InvariantCulture);
                            string previousPath = Path.Combine(dir, "engine-" + compare, safe, $"{clip}.{mode}.tokens");
                            if (compare is not null && File.Exists(previousPath))
                            {
                                int[] before = ParseIds(File.ReadAllText(previousPath));
                                previous = Describe(hf, before, FirstDifference(hf.Tokens, before));
                            }
                        }
                        string previousCell = compare is null ? "" : previous + " | ";
                        int hits = JfkWordHits(whisper.DecodeText(ids));
                        table.AppendLine(CultureInfo.InvariantCulture, $"| {repo} | {clip} | {mode} | {hfCount} | {verdict} | "
                            + $"{hfMargin} | {previousCell}{hits}/{JfkWords.Length} | {Cell(rendered)} |");
                    }
                }
            }
            finally
            {
                teacher?.Dispose();
            }
        }
        string text = table.ToString();
        output.WriteLine(text);
        string? outPath = Environment.GetEnvironmentVariable(OutEnvVar);
        if (!string.IsNullOrEmpty(outPath))
        {
            File.AppendAllText(outPath, text + Environment.NewLine);
        }
        if (Environment.GetEnvironmentVariable(StrictEnvVar) == "1")
        {
            Assert.True(divergences.Count == 0, string.Join("\n", divergences));
        }
    }

    /// <summary>The ids the pipeline's greedy argmax masks: SOT through the transcribe token, no-speech, no-timestamps.</summary>
    private static int[] Suppressed(WhisperTokenizer tokenizer)
    {
        List<int> ids = [];
        for (int id = tokenizer.SotId; id <= tokenizer.TranscribeId; id++)
        {
            ids.Add(id);
        }
        ids.Add(tokenizer.NoSpeechId);
        ids.Add(tokenizer.NoTimestampsId);
        return [.. ids.Distinct().Order()];
    }

    /// <summary>The first step at which the two decodes differ (a stop counts as a token), or -1 when they are equal.</summary>
    private static int FirstDifference(IReadOnlyList<int> reference, IReadOnlyList<int> ids)
    {
        int i = 0;
        while (i < reference.Count && i < ids.Count && reference[i] == ids[i])
        {
            i++;
        }
        return i == reference.Count && i == ids.Count ? -1 : i;
    }

    /// <summary>"identical", or the step, both choices and the reference's margin and runner-up there.</summary>
    private static string Describe(HfReference hf, IReadOnlyList<int> ids, int step)
    {
        if (step < 0)
        {
            return "identical";
        }
        string hfAt = step < hf.Tokens.Length ? hf.Tokens[step].ToString(CultureInfo.InvariantCulture) : "EOT";
        string engineAt = step < ids.Count ? ids[step].ToString(CultureInfo.InvariantCulture) : "EOT";
        string margin = step < hf.Margins.Length ? hf.Margins[step].ToString("F4", CultureInfo.InvariantCulture) : "—";
        string runnerUp = step < hf.RunnersUp.Length ? hf.RunnersUp[step].ToString(CultureInfo.InvariantCulture) : "—";
        return $"differ at step {step}: HF {hfAt} vs {engineAt} (HF margin {margin}, HF runner-up {runnerUp})";
    }

    /// <summary>The decoded text with each timestamp token shown as <c>&lt;s.ss&gt;</c>.</summary>
    private static string Render(WhisperPipeline whisper, WhisperTokenizer tokenizer, List<int> ids)
    {
        StringBuilder sb = new();
        List<int> span = [];
        foreach (int id in ids)
        {
            if (!tokenizer.IsTimestampId(id))
            {
                span.Add(id);
                continue;
            }
            sb.Append(whisper.DecodeText(span));
            span.Clear();
            sb.Append(CultureInfo.InvariantCulture, $"<{tokenizer.SecondsForTimestamp(id):F2}>");
        }
        sb.Append(whisper.DecodeText(span));
        return sb.ToString().Trim();
    }

    private static int[] ParseIds(string text)
        => text.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();

    private static IEnumerable<string> Models()
    {
        string list = Environment.GetEnvironmentVariable(ModelsEnvVar) ?? DefaultModels;
        return list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }

    private static int JfkWordHits(string text)
    {
        StringBuilder sb = new(text.Length);
        foreach (char c in text.ToLowerInvariant())
        {
            sb.Append(char.IsLetterOrDigit(c) || c == '\'' ? c : ' ');
        }
        HashSet<string> words = new(sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries), StringComparer.Ordinal);
        return JfkWords.Count(words.Contains);
    }

    private static string Cell(string text) => text.Replace("|", "\\|").Replace('\n', ' ');

    /// <summary>One reference decode as <c>whisper_reference.py</c> writes it.</summary>
    private sealed record HfReference(int[] Prompt, int[] Suppressed, int[] Tokens, float[] Margins, int[] RunnersUp)
    {
        public static HfReference Read(string path)
        {
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement root = doc.RootElement;
            int[] runnersUp = root.TryGetProperty("runners_up", out JsonElement r) ? r.EnumerateArray().Select(e => e.GetInt32()).ToArray() : [];
            return new HfReference(Ints(root, "prompt"), Ints(root, "suppressed"), Ints(root, "tokens"),
                root.GetProperty("margins").EnumerateArray().Select(e => e.GetSingle()).ToArray(), runnersUp);
        }

        private static int[] Ints(JsonElement root, string name) => root.GetProperty(name).EnumerateArray().Select(e => e.GetInt32()).ToArray();
    }

    /// <summary>The engine's encoder and decoder loaded beside the pipeline, to read its logits after a given prefix.</summary>
    private sealed class TeacherForcing(SafeTensorsLoader loader, WhisperConfig config, WhisperEncoder encoder,
        WhisperDecoder decoder) : IDisposable
    {
        public static TeacherForcing Load(string repo, string repoDir)
        {
            WhisperConfig config = WhisperPipeline.InferConfig(repo);
            using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(repoDir, "config.json"))))
            {
                int vocab = doc.RootElement.GetProperty("vocab_size").GetInt32();
                config = config with { VocabSize = vocab, IsMultilingual = vocab >= 51865 };
            }
            SafeTensorsLoader loader = new();
            loader.Load(Path.Combine(repoDir, "model.safetensors"));
            Dictionary<string, Tensor> weights = loader.GetAllTensors();
            WhisperEncoder encoder = new(config);
            WhisperDecoder decoder = new(config);
            encoder.LoadWeights(weights);
            decoder.LoadWeights(weights);
            return new TeacherForcing(loader, config, encoder, decoder);
        }

        /// <summary>The engine's choice, runner-up and margin at <paramref name="step"/> with the reference's tokens before
        /// it as the prefix, fed exactly as the pipeline feeds its own: the prompt in one pass, then one token per step.</summary>
        public string Explain(IBackend backend, float[] audio, int[] prompt, int[] reference, int step, int[] suppressed, int eot)
        {
            MelSpectrogramExtractor extractor = new(MelSpectrogramExtractor.WhisperConfig(config.NumMelBins));
            int frames = extractor.OutputFrames(WindowSamples);
            using Tensor mel = new(new TensorShape(1, config.NumMelBins, frames), DType.F32);
            extractor.ComputeZeroPadded(audio.AsSpan(0, Math.Min(audio.Length, WindowSamples)), WindowSamples, mel.AsSpan<float>());
            using Tensor encoded = encoder.Forward(backend, mel);
            using WhisperDecoder.DecodeState state = decoder.StartDecode(backend, encoded);
            Tensor logits = decoder.DecodeStep(backend, prompt, state);
            for (int i = 0; i < step; i++)
            {
                logits.Dispose();
                logits = decoder.DecodeStep(backend, [reference[i]], state);
            }
            float[] row = logits.AsSpan<float>()[..config.VocabSize].ToArray();
            logits.Dispose();
            foreach (int id in suppressed)
            {
                row[id] = float.NegativeInfinity;
            }
            int best = -1, second = -1;
            for (int v = 0; v < row.Length; v++)
            {
                if (best < 0 || row[v] > row[best])
                {
                    second = best;
                    best = v;
                }
                else if (second < 0 || row[v] > row[second])
                {
                    second = v;
                }
            }
            string hfToken = step < reference.Length ? reference[step].ToString(CultureInfo.InvariantCulture) : $"EOT {eot}";
            int hfId = step < reference.Length ? reference[step] : eot;
            return string.Create(CultureInfo.InvariantCulture,
                $"engine teacher-forced: {best} over {second} by {row[best] - row[second]:F4}; HF's {hfToken} trails by {row[best] - row[hfId]:F4}");
        }

        public void Dispose()
        {
            encoder.Dispose();
            decoder.Dispose();
            loader.Dispose();
        }
    }
}
