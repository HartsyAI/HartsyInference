using System.Diagnostics;
using System.Runtime.InteropServices;
using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The official DeepSeek-V4.1-Flash checkpoint (pinned revision dba1be0a) through the CPU host reference. The weights are found through
/// <see cref="ModelResolver"/> (the models root's <c>llm/deepseek-v4.1-flash</c>); a missing directory skips unless <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c>, which fails instead.
/// Run alone, under a memory cap so a bad estimate cannot take the machine down:
/// <c>systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 env HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models dotnet test tests/HartsyInference.LLM.Tests -c Release -f net10.0 --filter "FullyQualifiedName~DeepSeekV41RealWeightsTests"</c>.
/// The full-depth run is further gated behind <c>DSV41_FULL_RUN=1</c>.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class DeepSeekV41RealWeightsTests
{
    private readonly ITestOutputHelper _output;

    public DeepSeekV41RealWeightsTests(ITestOutputHelper output) => _output = output;

    private static string ModelDirectory() =>
        ModelResolver.Resolve(DeepSeekV41Catalog.Id, null, Modality.Text).LocalPath
        ?? Path.Combine(RepoPaths.ModelsRoot(), "llm", "deepseek-v4.1-flash");

    private bool HaveWeights(out string dir)
    {
        dir = ModelDirectory();
        return RealWeightGate.Require(_output.WriteLine, Path.Combine(dir, "config.json"), Path.Combine(dir, "model.safetensors.index.json"));
    }

    private static string Rss() => $"{Process.GetCurrentProcess().WorkingSet64 / (1L << 30)} GiB (peak {Process.GetCurrentProcess().PeakWorkingSet64 / (1L << 30)} GiB)";

    [Fact]
    public void WorkingMemoryEstimate_ForTheOfficialConfigIsReported()
    {
        if (!HaveWeights(out string dir)) return;
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(dir);
        DeepSeekV41Config cfg = checkpoint.Config;
        foreach (DeepSeekV41Residency residency in Enum.GetValues<DeepSeekV41Residency>())
        {
            DeepSeekV41LoadOptions options = new(HfTextDirectoryLoader.MaxSequenceTokens, Residency: residency);
            long dense = checkpoint.Weights.BytesByClass[DeepSeekV41WeightClass.Dense] + checkpoint.Weights.BytesByClass[DeepSeekV41WeightClass.Embed] + checkpoint.Weights.BytesByClass[DeepSeekV41WeightClass.Head];
            _output.WriteLine($"{residency}: anonymous {DeepSeekV41WorkingMemory.AnonymousBytes(cfg, dense, options) / (double)(1L << 30):F1} GiB "
                + $"(state {DeepSeekV41WorkingMemory.SequenceStateBytes(cfg, options.MaxTokens) / (double)(1L << 30):F2}, activations {DeepSeekV41WorkingMemory.ActivationBytes(cfg, options.MaxTokens) / (double)(1L << 30):F2}, small {DeepSeekV41WorkingMemory.SmallTensorBytes(cfg) / (double)(1L << 30):F2} GiB)");
        }
    }

    /// <summary>Layers 0 (dense), 1 (Engram) and 2 (compressor and indexer) with real weights: every decoder, the Engram gather and the sparse attention run, in a few GiB.</summary>
    [Fact]
    public void ThreeLayers_PrefillAndDecodeGiveFiniteStates()
    {
        if (!HaveWeights(out string dir)) return;
        using CpuBackend backend = new();
        Stopwatch sw = Stopwatch.StartNew();
        using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(backend, dir, new DeepSeekV41LoadOptions(MaxTokens: 64, MaxLayers: 3));
        _output.WriteLine($"load (3 layers) {sw.Elapsed.TotalSeconds:F1}s, RSS {Rss()}");
        using FileStream tokStream = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokStream, bosToken: DeepSeekV41TextModel.BosLiteral, eosToken: DeepSeekV41TextModel.EosLiteral);

        DeepSeekV41HostModel model = loaded.Model;
        List<int> ids = [tokenizer.BosId!.Value, .. tokenizer.EncodeOrdinary("The capital of France is")];
        DeepSeekV41SequenceState state = model.CreateState(64);
        float[] hidden = new float[ids.Count * model.Dim];
        sw.Restart();
        model.Forward(ids.ToArray(), state, hidden);
        _output.WriteLine($"prefill {ids.Count} tokens {sw.Elapsed.TotalSeconds:F1}s, RSS {Rss()}");
        Assert.All(hidden, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(hidden, v => v != 0f);

        float[] one = new float[model.Dim];
        sw.Restart();
        model.Forward([ids[^1]], state, one);
        _output.WriteLine($"decode 1 token {sw.Elapsed.TotalSeconds:F1}s, RSS {Rss()}");
        Assert.All(one, v => Assert.True(float.IsFinite(v)));
        float[] logits = model.Logits(one);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        _output.WriteLine($"logits {sw.Elapsed.TotalSeconds:F1}s total, RSS {Rss()}");
    }

    [Fact]
    public void FullDepth_PrefillGivesFiniteLogitsAndGreedyTokens()
    {
        if (Environment.GetEnvironmentVariable("DSV41_FULL_RUN") != "1")
        {
            _output.WriteLine("SKIPPED: set DSV41_FULL_RUN=1 for the full 40-layer run");
            return;
        }
        if (!HaveWeights(out string dir)) return;

        using CpuBackend backend = new();
        Stopwatch sw = Stopwatch.StartNew();
        using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(backend, dir, new DeepSeekV41LoadOptions(MaxTokens: 64));
        using FileStream tokStream = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokStream, bosToken: DeepSeekV41TextModel.BosLiteral, eosToken: DeepSeekV41TextModel.EosLiteral);
        _output.WriteLine($"load {sw.Elapsed.TotalSeconds:F1}s, peak RSS {Process.GetCurrentProcess().PeakWorkingSet64 / (1L << 30)} GiB");

        DeepSeekV41HostModel model = loaded.Model;
        List<int> ids = [tokenizer.BosId!.Value, .. tokenizer.EncodeOrdinary("The capital of France is")];
        DeepSeekV41SequenceState state = model.CreateState(64);
        float[] hidden = new float[ids.Count * model.Dim];
        sw.Restart();
        model.Forward(ids.ToArray(), state, hidden);
        _output.WriteLine($"prefill {ids.Count} tokens {sw.Elapsed.TotalSeconds:F1}s");
        float[] logits = model.Logits(hidden.AsSpan((ids.Count - 1) * model.Dim, model.Dim));
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));

        for (int step = 0; step < 8; step++)
        {
            int next = Array.IndexOf(logits, logits.Max());
            ids.Add(next);
            _output.WriteLine($"step {step}: {next} {tokenizer.Decode([next]).Replace("\n", "\\n")}");
            sw.Restart();
            float[] h = new float[model.Dim];
            model.Forward([next], state, h);
            logits = model.Logits(h);
            _output.WriteLine($"  {sw.Elapsed.TotalSeconds:F1}s/token, peak RSS {Process.GetCurrentProcess().PeakWorkingSet64 / (1L << 30)} GiB");
            Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        }
    }

    /// <summary>Dequantized row slices dumped by <c>tests/python-reference/deepseek_v41/dump_real_slices.py</c> (set <c>DSV41_REAL_SLICES</c> to its output directory).</summary>
    [Fact]
    public void RealTensorSlices_DequantizeLikeTheReference()
    {
        string dir = ModelDirectory();
        string? slices = Environment.GetEnvironmentVariable("DSV41_REAL_SLICES");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(dir, "config.json"), slices ?? "DSV41_REAL_SLICES-unset")) return;

        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(dir);
        foreach (System.Text.Json.JsonElement job in System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(slices!, "manifest.json"))).RootElement.EnumerateArray())
        {
            string key = job.GetProperty("key").GetString()!;
            int r0 = job.GetProperty("r0").GetInt32(), r1 = job.GetProperty("r1").GetInt32(), cols = job.GetProperty("cols").GetInt32();
            float[] expected = new float[(r1 - r0) * cols];
            Buffer.BlockCopy(File.ReadAllBytes(Path.Combine(slices!, job.GetProperty("file").GetString()!)), 0, expected, 0, expected.Length * 4);

            float[] all = WeightDequantizer.ToF32(checkpoint.GetWeight(key), checkpoint.GetQuant(key));
            Assert.Equal(0, all.Length % cols);
            int worst = 0; double maxDiff = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                double diff = Math.Abs(all[(long)r0 * cols + i] - expected[i]);
                if (diff > maxDiff) { maxDiff = diff; worst = i; }
            }
            _output.WriteLine($"{key}[{r0}:{r1}] cols {cols} (logical {all.Length / cols} rows) maxDiff {maxDiff:E2} at {worst}");
            Assert.Equal(0.0, maxDiff);
        }
    }

    // Gates fixed before the first comparison ran. exact: the upstream model in float32 with activation quantization removed and an exact
    // softmax, so the F32 host reference should agree to rounding; a miss is a structural difference. ports: the unmodified pure-torch ports
    // with FP8 activation quantization, which the host reference does not model; the numbers are a precision envelope and only guard
    // against gross divergence.
    private const double ExactHiddenRelL2 = 2e-3;
    private const double ExactLogitsCosine = 0.9999;
    private const double PortsCosineFloor = 0.95;
    // added after the first run as a regression guard, not a pre-set gate; the measured overlap is 10
    private const int ExactTop10Overlap = 9;

    /// <summary>The first N layers of the real checkpoint through the host reference against the UNMODIFIED upstream model on the same real weights, dumped by
    /// <c>tests/python-reference/deepseek_v41/dump_real_layers.py</c> (set <c>DSV41_ORACLE_DIR</c> to its output directory; its <c>meta.json</c> names the layers, mode and ids).</summary>
    [Fact]
    public void RealLayers_MatchTheUpstreamModel()
    {
        string dir = ModelDirectory();
        string? oracle = Environment.GetEnvironmentVariable("DSV41_ORACLE_DIR");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(dir, "config.json"), oracle is null ? "DSV41_ORACLE_DIR-unset" : Path.Combine(oracle, "meta.json"))) return;

        using System.Text.Json.JsonDocument meta = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(oracle!, "meta.json")));
        int layers = meta.RootElement.GetProperty("layers").GetInt32();
        string mode = meta.RootElement.GetProperty("mode").GetString()!;
        Assert.True(mode is "exact" or "ports", $"unknown oracle mode '{mode}' in meta.json");
        string? hostDump = Environment.GetEnvironmentVariable("DSV41_HOST_DUMP");
        if (string.IsNullOrEmpty(hostDump)) hostDump = null;
        Assert.True(hostDump is null || layers == 1, "DSV41_HOST_DUMP writes one file per stage and supports a single layer");
        int[] ids = meta.RootElement.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();

        using CpuBackend backend = new();
        using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(backend, dir, new DeepSeekV41LoadOptions(MaxTokens: 64, MaxLayers: layers));
        using FileStream tokStream = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokStream, bosToken: DeepSeekV41TextModel.BosLiteral, eosToken: DeepSeekV41TextModel.EosLiteral);
        int[] hostIds = [tokenizer.BosId!.Value, .. tokenizer.EncodeOrdinary("The capital of France is")];
        _output.WriteLine($"host ids [{string.Join(",", hostIds)}] oracle ids [{string.Join(",", ids)}]");
        Assert.Equal(ids, hostIds);

        DeepSeekV41HostModel model = loaded.Model;
        DeepSeekV41SequenceState state = model.CreateState(64);
        float[] hidden = new float[ids.Length * model.Dim];
        Stopwatch sw = Stopwatch.StartNew();
        if (hostDump is not null)
        {
            Directory.CreateDirectory(hostDump);
            model.SetProbe((_, stage, values) => File.WriteAllBytes(Path.Combine(hostDump, $"{stage}.f32"), ToBytes(values)));
        }
        try { model.Forward(ids, state, hidden); }
        finally { model.SetProbe(null); }
        float[] logits = model.Logits(hidden.AsSpan((ids.Length - 1) * model.Dim, model.Dim));
        _output.WriteLine($"host {layers}-layer forward {sw.Elapsed.TotalSeconds:F1}s, RSS {Rss()}");

        if (hostDump is not null)
        {
            File.WriteAllBytes(Path.Combine(hostDump, "final.f32"), ToBytes(hidden));
            File.WriteAllBytes(Path.Combine(hostDump, "logits.f32"), ToBytes(logits));
        }

        float[] refHidden = ReadF32(Path.Combine(oracle!, "final.f32"), hidden.Length);
        float[] refLogits = ReadF32(Path.Combine(oracle!, "logits.f32"), logits.Length);
        double hiddenRel = RelL2(hidden, refHidden), logitsRel = RelL2(logits, refLogits);
        double hiddenCos = Cosine(hidden, refHidden), logitsCos = Cosine(logits, refLogits);
        int hostTop = ArgMax(logits), refTop = ArgMax(refLogits);
        int overlap = TopK(logits, 10).Intersect(TopK(refLogits, 10)).Count();
        _output.WriteLine($"[{mode}] hidden relL2 {hiddenRel:E3} cos {hiddenCos:F6}; logits relL2 {logitsRel:E3} cos {logitsCos:F6}; "
            + $"argmax host {hostTop} oracle {refTop}; top-10 overlap {overlap}/10; max|dlogit| {MaxAbsDiff(logits, refLogits):E3}");

        if (mode == "exact")
        {
            Assert.True(hiddenRel <= ExactHiddenRelL2, $"hidden relL2 {hiddenRel:E3} > {ExactHiddenRelL2:E1}");
            Assert.True(logitsCos >= ExactLogitsCosine, $"logits cosine {logitsCos:F6} < {ExactLogitsCosine}");
            Assert.Equal(refTop, hostTop);
            Assert.True(overlap >= ExactTop10Overlap, $"top-10 overlap {overlap}/10 < {ExactTop10Overlap}");
        }
        else
        {
            Assert.True(hiddenCos >= PortsCosineFloor, $"hidden cosine {hiddenCos:F6} < {PortsCosineFloor}");
            Assert.True(logitsCos >= PortsCosineFloor, $"logits cosine {logitsCos:F6} < {PortsCosineFloor}");
        }
    }

    private static byte[] ToBytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private static float[] ReadF32(string path, int expected)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Assert.Equal(expected * 4, bytes.Length);
        float[] values = new float[expected];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static double RelL2(float[] a, float[] b)
    {
        double diff = 0, norm = 0;
        for (int i = 0; i < a.Length; i++) { double d = (double)a[i] - b[i]; diff += d * d; norm += (double)b[i] * b[i]; }
        return Math.Sqrt(diff / Math.Max(norm, 1e-30));
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / Math.Max(Math.Sqrt(na * nb), 1e-30);
    }

    private static double MaxAbsDiff(float[] a, float[] b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs((double)a[i] - b[i]));
        return max;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static int[] TopK(float[] v, int k) => Enumerable.Range(0, v.Length).OrderByDescending(i => v[i]).Take(k).ToArray();
}
