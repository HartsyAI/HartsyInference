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

    // Gates fixed before the first comparison ran (layer 0). exact: the upstream model in float32 with activation quantization removed and an exact
    // softmax, so the F32 host reference should agree to rounding; a miss is a structural difference. ports: the unmodified pure-torch ports
    // with FP8 activation quantization, which the host reference does not model; the numbers are a precision envelope and only guard
    // against gross divergence. The multi-layer comparison applies the same gates to every layer, sublayer and decode step; they were not
    // loosened for depth.
    private const double ExactHiddenRelL2 = 2e-3;
    private const double ExactLogitsCosine = 0.9999;
    private const double PortsCosineFloor = 0.95;
    // added after the first run as regression guards, not pre-set gates; measured: overlap 10, logits relL2 2.8e-6. Cosine and argmax are blind to a
    // uniform rescale of the head, so the logits' magnitude is gated too, at the hidden tolerance (logits are one linear map of the hidden state).
    private const int ExactTop10Overlap = 9;
    private const double ExactLogitsRelL2 = 2e-3;
    // the final RMS norm is scale-invariant, so the hidden and logits checks cannot see a uniformly rescaled residual stream; the raw output of every
    // block, which is what the next layer receives, is gated separately (measured 3e-6 at layer 0), as is every tapped sublayer
    private const double ExactBlockRelL2 = 2e-3;
    // ports is an envelope, so its magnitude bound is deliberately wide: about 10x the measured 2e-2, enough to catch a scale or gross error and not a precision regression.
    // It applies to one layer; deeper ports runs are reported and only checked for finite values.
    private const double PortsRelL2Ceiling = 0.2;
    // Quantized exact mode at depth (fixed before the first deep run, after a 4-layer run showed the cause). The window and compressed caches are FP8/FP4: a
    // float-noise difference of ~1e-6 before quantization flips an isolated element by one step (measured: 2 of 3072 window elements differ at layer 3, the
    // compressed cache is bit-identical), which moves a sublayer by 1e-4..1e-3 and so can pass the strict 2e-3 gate by luck or fail it for a benign reason.
    // The strict check at depth is the structural mode; these gates only have to catch real divergence, which shows as cosine well below 0.99.
    private const double QuantizedBlockRelL2 = 5e-2;
    private const double QuantizedHiddenRelL2 = 5e-2;
    private const double QuantizedLogitsCosine = 0.999;
    private const int QuantizedTop10Overlap = 8;
    // a different argmax is tolerated only when the host picked the oracle's runner-up and the oracle's two best logits were closer than this
    private const double NearTieLogitGap = 0.05;
    // Structural mode, amended after the first 1300-token run (which failed the cache gate): with that many tokens a routing near-tie is expected, where the 6th and
    // 7th biased gate scores of a token are within float noise and two correct implementations pick different experts. The oracle records that gap per token and
    // layer. Parity stays strict up to the first layer holding a near-tie; at that layer every token WITHOUT a near-tie must still match strictly (its input and its
    // attention are clean), the tie tokens are reported, and deeper layers and the final outputs get the flip-aware gates above.
    private const double NearTieMargin = 1e-5;

    private static readonly string[] SublayerStages = ["attn_in", "attn_out", "ffn_in", "ffn_out"];

    /// <summary>The first N layers of the real checkpoint through the host reference against the UNMODIFIED upstream model on the same real weights, dumped by
    /// <c>tests/python-reference/deepseek_v41/dump_real_layers.py</c> (set <c>DSV41_ORACLE_DIR</c> to its output directory; its <c>meta.json</c> names the layers,
    /// mode, ids and the oracle's greedy tokens). The host replays the oracle's tokens through the decode steps (teacher forcing), so one early disagreement cannot
    /// cascade, and every layer's block output (plus its sublayers when the dump has them) is compared on every call.</summary>
    [Fact]
    public void RealLayers_MatchTheUpstreamModel()
    {
        string dir = ModelDirectory();
        string? oracle = Environment.GetEnvironmentVariable("DSV41_ORACLE_DIR");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(dir, "config.json"), oracle is null ? "DSV41_ORACLE_DIR-unset" : Path.Combine(oracle, "meta.json"))) return;

        using System.Text.Json.JsonDocument meta = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(oracle!, "meta.json")));
        System.Text.Json.JsonElement root = meta.RootElement;
        int layers = root.GetProperty("layers").GetInt32();
        string mode = root.GetProperty("mode").GetString()!;
        Assert.True(mode is "exact" or "structural" or "ports", $"unknown oracle mode '{mode}' in meta.json");
        int steps = root.GetProperty("steps").GetInt32();
        bool stageTaps = root.GetProperty("stage_taps").GetBoolean();
        int[] ids = root.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int[] generated = root.GetProperty("generated").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        Assert.Equal(steps + 1, generated.Length);
        string? hostDump = Environment.GetEnvironmentVariable("DSV41_HOST_DUMP");
        if (string.IsNullOrEmpty(hostDump)) hostDump = null;

        int maxTokens = Math.Max(64, ids.Length + steps + 1);
        using CpuBackend backend = new();
        Stopwatch sw = Stopwatch.StartNew();
        using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(backend, dir, new DeepSeekV41LoadOptions(MaxTokens: maxTokens, MaxLayers: layers, QuantizeLatents: mode != "structural"));
        _output.WriteLine($"host load ({layers} layers) {sw.Elapsed.TotalSeconds:F1}s, RSS {Rss()}");
        // both sides run the ids the oracle recorded; the tokenizer is cross-checked only when the oracle says which text they came from
        if (root.TryGetProperty("prompt", out System.Text.Json.JsonElement promptElement) && promptElement.ValueKind == System.Text.Json.JsonValueKind.String)
        {
            using FileStream tokStream = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
            ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokStream, bosToken: DeepSeekV41TextModel.BosLiteral, eosToken: DeepSeekV41TextModel.EosLiteral);
            int[] hostIds = [tokenizer.BosId!.Value, .. tokenizer.EncodeOrdinary(promptElement.GetString()!)];
            _output.WriteLine($"host ids [{string.Join(",", hostIds.Take(12))}...] oracle ids [{string.Join(",", ids.Take(12))}...] ({ids.Length} tokens)");
            Assert.Equal(ids, hostIds);
        }
        else _output.WriteLine($"custom oracle ids ({ids.Length} tokens); tokenizer not cross-checked");

        DeepSeekV41HostModel model = loaded.Model;
        DeepSeekV41SequenceState state = model.CreateState(maxTokens);
        Dictionary<(int Layer, string Stage), float[]> seen = [];
        // Tie and divergence state carries across calls: a near-tie in the prefill makes the caches of every later decode step depend on it.
        int historyTieLayer = -1;
        int historyDivergedPos = int.MaxValue;
        List<string> failures = [];
        bool strict = mode == "structural" || (mode == "exact" && layers == 1);
        bool quantizedDeep = mode == "exact" && layers > 1;
        model.SetProbe((layer, stage, values) => seen[(layer, stage)] = values);
        try
        {
            float[] hidden = new float[ids.Length * model.Dim];
            sw.Restart();
            model.Forward(ids, state, hidden);
            float[] logits = model.Logits(hidden.AsSpan((ids.Length - 1) * model.Dim, model.Dim));
            _output.WriteLine($"host prefill {ids.Length} tokens {sw.Elapsed.TotalSeconds:F1}s, RSS {Rss()}");
            CompareCall("prefill", hidden, logits);
            for (int j = 1; j <= steps; j++)
            {
                float[] one = new float[model.Dim];
                sw.Restart();
                model.Forward([generated[j - 1]], state, one);
                logits = model.Logits(one);
                _output.WriteLine($"host step {j} {sw.Elapsed.TotalSeconds:F1}s, RSS {Rss()}");
                CompareCall($"step{j}", one, logits);
            }
        }
        finally { model.SetProbe(null); }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));

        void CompareCall(string call, float[] hidden, float[] logits)
        {
            string callDir = Path.Combine(oracle!, call);
            string? dumpDir = hostDump is null ? null : Path.Combine(hostDump, call);
            if (dumpDir is not null)
            {
                Directory.CreateDirectory(dumpDir);
                foreach (((int layer, string stage), float[] values) in seen)
                    File.WriteAllBytes(Path.Combine(dumpDir, $"{(stage == "out" ? "block" : stage)}{layer}.f32"), ToBytes(values));
                File.WriteAllBytes(Path.Combine(dumpDir, "final.f32"), ToBytes(hidden));
                File.WriteAllBytes(Path.Combine(dumpDir, "logits.f32"), ToBytes(logits));
            }

            int positions = hidden.Length / model.Dim;
            int callStart = state.Length - positions;
            int callTieLayer = -1;
            bool[] callTie = new bool[positions];
            List<string> tieNotes = [];
            if (mode == "structural")
                for (int layer = 0; layer < layers; layer++)
                {
                    string marginPath = Path.Combine(callDir, $"route_margin{layer}.f32");
                    if (!File.Exists(marginPath)) continue;
                    float[] margins = ReadF32(marginPath, positions);
                    int[] low = Enumerable.Range(0, positions).Where(t => margins[t] < NearTieMargin).ToArray();
                    if (low.Length == 0) continue;
                    tieNotes.Add($"layer {layer}: " + string.Join(", ", low.Take(6).Select(t => $"token {t} margin {margins[t]:E1}")) + (low.Length > 6 ? $" (+{low.Length - 6} more)" : ""));
                    if (callTieLayer >= 0) continue;
                    callTieLayer = layer;
                    foreach (int t in low) callTie[t] = true;
                }
            if (callTieLayer >= 0) historyTieLayer = historyTieLayer < 0 ? callTieLayer : Math.Min(historyTieLayer, callTieLayer);
            int tieLayer = historyTieLayer;
            // only this call's own tie tokens are excluded, and only at the layer where this call's ties sit
            bool[] tie = callTieLayer == tieLayer ? callTie : new bool[positions];
            bool flipAware = quantizedDeep || tieLayer >= 0;
            _output.WriteLine($"[{mode}] {call}: routing near-ties (margin < {NearTieMargin:E0}): " + (tieNotes.Count == 0 ? "none" : string.Join("; ", tieNotes)));

            // allowed relL2 for one stage of one layer, and whether the near-tie tokens are left out (only the stages that depend on the layer's own routing)
            (double Gate, bool SkipTies) StageGate(int layer, string stage)
            {
                if (quantizedDeep) return (QuantizedBlockRelL2, false);
                if (!strict && mode != "structural") return (double.PositiveInfinity, false);
                if (tieLayer < 0 || layer < tieLayer) return (ExactBlockRelL2, false);
                if (layer > tieLayer) return (QuantizedBlockRelL2, false);
                return (ExactBlockRelL2, stage is "out" or "ffn_out");
            }

            double worstBlock = 0, worstStage = 0;
            string firstOver = "none";
            List<string> perLayer = [];
            for (int layer = 0; layer < layers; layer++)
            {
                Assert.True(seen.TryGetValue((layer, "out"), out float[]? blockOut), $"{call}: no block output captured for layer {layer}");
                float[] refBlock = ReadF32(Path.Combine(callDir, $"block{layer}.f32"), blockOut!.Length);
                double blockRel = RelL2(blockOut, refBlock);
                (double blockGate, bool skipTies) = StageGate(layer, "out");
                double gated = skipTies ? RelL2Rows(blockOut, refBlock, positions, tie) : blockRel;
                perLayer.Add($"{blockRel:E0}");
                worstBlock = Math.Max(worstBlock, blockRel);
                if (gated > blockGate && firstOver == "none") firstOver = $"layer {layer} block {gated:E2}" + (skipTies ? " (tie tokens excluded)" : "");
                if (!stageTaps) continue;
                foreach (string stage in SublayerStages)
                {
                    float[] values = seen[(layer, stage)];
                    float[] refValues = ReadF32(Path.Combine(callDir, $"{stage}{layer}.f32"), values.Length);
                    double rel = RelL2(values, refValues);
                    (double stageGate, bool skip) = StageGate(layer, stage);
                    double stageGated = skip ? RelL2Rows(values, refValues, positions, tie) : rel;
                    worstStage = Math.Max(worstStage, rel);
                    if (stageGated > stageGate && firstOver == "none") firstOver = $"layer {layer} {stage} {stageGated:E2}" + (skip ? " (tie tokens excluded)" : "");
                }
            }

            // a layer's caches are built from its own input, so they are clean up to and including the first tie layer
            double worstWindow = 0, worstCompress = 0, gatedCache = 0;
            string cacheNote = "";
            for (int layer = 0; layer < layers; layer++)
            {
                bool cacheClean = tieLayer < 0 || layer <= tieLayer;
                DeepSeekV41AttentionState layerState = state.Layers[layer];
                float[] refWindow = ReadF32(Path.Combine(callDir, $"cache_window{layer}.f32"), layerState.Window.Length);
                double windowRel = RelL2(layerState.Window, refWindow);
                worstWindow = Math.Max(worstWindow, windowRel);
                if (cacheClean) gatedCache = Math.Max(gatedCache, windowRel);
                string compressPath = Path.Combine(callDir, $"cache_compress{layer}.f32");
                if (!File.Exists(compressPath)) continue;
                float[] refCompress = ReadF32(compressPath, (int)(new FileInfo(compressPath).Length / 4));
                float[] hostCompress = layerState.CompressKv![..refCompress.Length];
                double compressRel = RelL2(hostCompress, refCompress);
                int differing = 0;
                for (int i = 0; i < refCompress.Length; i++) if (hostCompress[i] != refCompress[i]) differing++;
                worstCompress = Math.Max(worstCompress, compressRel);
                if (cacheClean) gatedCache = Math.Max(gatedCache, compressRel);
                cacheNote += $" L{layer}:{refCompress.Length / 512}rows relL2 {compressRel:E1} differing {differing}/{refCompress.Length}";
                if (dumpDir is not null) File.WriteAllBytes(Path.Combine(dumpDir, $"cache_compress{layer}.f32"), ToBytes(hostCompress));
            }
            if (dumpDir is not null)
                for (int layer = 0; layer < layers; layer++)
                    File.WriteAllBytes(Path.Combine(dumpDir, $"cache_window{layer}.f32"), ToBytes(state.Layers[layer].Window));
            _output.WriteLine($"[{mode}] {call}: cache window worst relL2 {worstWindow:E1}; compressed{cacheNote}");

            // expert selection itself, token by token. Token t at layer L depends only on positions <= t at earlier layers, so before any layer has diverged
            // every disagreement must sit at a token whose oracle routing margin is inside float noise. After a divergence at position p, tokens >= p may
            // legitimately flip through attention, so only disagreements before that are held to the margin rule.
            int routeLayers = 0, nearTie = 0, propagated = 0;
            double largestNearTie = 0;
            List<string> unexplained = [];
            for (int layer = 0; layer < layers; layer++)
            {
                string idxPath = Path.Combine(callDir, $"route_idx{layer}.f32");
                if (!File.Exists(idxPath) || !seen.TryGetValue((layer, "route"), out float[]? hostRoute)) continue;
                routeLayers++;
                float[] oracleRoute = ReadF32(idxPath, hostRoute.Length);
                float[] margin = ReadF32(Path.Combine(callDir, $"route_margin{layer}.f32"), positions);
                int k = hostRoute.Length / positions, layerFirst = int.MaxValue;
                for (int t = 0; t < positions; t++)
                {
                    HashSet<int> host = [], oracleSet = [];
                    for (int j = 0; j < k; j++)
                    {
                        host.Add((int)hostRoute[t * k + j]);
                        oracleSet.Add((int)oracleRoute[t * k + j]);
                    }
                    if (host.SetEquals(oracleSet)) continue;
                    int global = callStart + t;
                    if (global >= historyDivergedPos) { propagated++; continue; }
                    if (margin[t] < NearTieMargin) { nearTie++; largestNearTie = Math.Max(largestNearTie, margin[t]); }
                    else if (unexplained.Count < 6) unexplained.Add($"layer {layer} position {global} margin {margin[t]:E2} host [{string.Join(",", host)}] oracle [{string.Join(",", oracleSet)}]");
                    layerFirst = Math.Min(layerFirst, global);
                }
                historyDivergedPos = Math.Min(historyDivergedPos, layerFirst);
            }
            if (routeLayers > 0)
                _output.WriteLine($"[{mode}] {call}: expert selection: {nearTie} near-tie disagreements (largest margin {largestNearTie:E1}), {propagated} downstream of an earlier divergence, {unexplained.Count} unexplained");
            if (mode == "structural" && unexplained.Count > 0)
                failures.Add($"{call}: expert selection differs at a token with a clear routing margin and no earlier divergence: {string.Join("; ", unexplained)}");

            float[] refHidden = ReadF32(Path.Combine(callDir, "final.f32"), hidden.Length);
            float[] refLogits = ReadF32(Path.Combine(callDir, "logits.f32"), logits.Length);
            double hiddenRel = RelL2(hidden, refHidden), logitsRel = RelL2(logits, refLogits), logitsCos = Cosine(logits, refLogits);
            int hostTop = ArgMax(logits), refTop = ArgMax(refLogits);
            int overlap = TopK(logits, 10).Intersect(TopK(refLogits, 10)).Count();
            _output.WriteLine($"[{mode}] {call}: worst block relL2 {worstBlock:E2}" + (stageTaps ? $", worst sublayer {worstStage:E2}" : "") + $", first over gate: {firstOver}; "
                + $"final hidden relL2 {hiddenRel:E2}; logits relL2 {logitsRel:E2} cos {logitsCos:F6}; argmax host {hostTop} oracle {refTop}"
                + (call != "prefill" ? $" (oracle token {generated[int.Parse(call[4..])]})" : "") + $"; top-10 overlap {overlap}/10");
            if (call == "prefill" || call == $"step{steps}") _output.WriteLine($"[{mode}] {call} block relL2 by layer: {string.Join(" ", perLayer)}");

            Assert.All(hidden, v => Assert.True(float.IsFinite(v)));
            Assert.All(logits, v => Assert.True(float.IsFinite(v)));
            if (firstOver != "none") failures.Add($"{call}: first block or sublayer over its gate: {firstOver}");
            if (strict && !flipAware)
            {
                if (hiddenRel > ExactHiddenRelL2) failures.Add($"{call}: final hidden relL2 {hiddenRel:E3} > {ExactHiddenRelL2:E1}");
                if (logitsCos < ExactLogitsCosine) failures.Add($"{call}: logits cosine {logitsCos:F6} < {ExactLogitsCosine}");
                if (logitsRel > ExactLogitsRelL2) failures.Add($"{call}: logits relL2 {logitsRel:E3} > {ExactLogitsRelL2:E1}");
                if (hostTop != refTop) failures.Add($"{call}: argmax host {hostTop} != oracle {refTop}");
                if (overlap < ExactTop10Overlap) failures.Add($"{call}: top-10 overlap {overlap}/10 < {ExactTop10Overlap}");
            }
            else if (flipAware)
            {
                if (hiddenRel > QuantizedHiddenRelL2) failures.Add($"{call}: final hidden relL2 {hiddenRel:E3} > {QuantizedHiddenRelL2:E1}");
                if (logitsCos < QuantizedLogitsCosine) failures.Add($"{call}: logits cosine {logitsCos:F6} < {QuantizedLogitsCosine}");
                if (overlap < QuantizedTop10Overlap) failures.Add($"{call}: top-10 overlap {overlap}/10 < {QuantizedTop10Overlap}");
                if (hostTop != refTop)
                {
                    int runnerUp = SecondBest(refLogits, refTop);
                    double gap = refLogits[refTop] - refLogits[runnerUp];
                    if (hostTop == runnerUp && gap < NearTieLogitGap) _output.WriteLine($"[{mode}] {call}: near-tie, host picked the oracle's runner-up {runnerUp} (gap {gap:F4})");
                    else failures.Add($"{call}: argmax host {hostTop} != oracle {refTop} (oracle runner-up {runnerUp}, gap {gap:F4})");
                }
            }
            else if (mode == "ports" && layers == 1)
            {
                double hiddenCos = Cosine(hidden, refHidden);
                if (worstBlock > PortsRelL2Ceiling || hiddenRel > PortsRelL2Ceiling || logitsRel > PortsRelL2Ceiling)
                    failures.Add($"{call}: ports relL2 above {PortsRelL2Ceiling} (block {worstBlock:E2}, hidden {hiddenRel:E2}, logits {logitsRel:E2})");
                if (hiddenCos < PortsCosineFloor || logitsCos < PortsCosineFloor)
                    failures.Add($"{call}: ports cosine below {PortsCosineFloor} (hidden {hiddenCos:F6}, logits {logitsCos:F6})");
            }
            if (mode == "structural" && gatedCache > ExactBlockRelL2)
                failures.Add($"{call}: structural cache state differs up to the first tie layer (worst {gatedCache:E2} > {ExactBlockRelL2:E0})");
            seen.Clear();
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

    private static double RelL2Rows(float[] a, float[] b, int rows, bool[] skip)
    {
        int width = a.Length / rows;
        double diff = 0, norm = 0;
        for (int r = 0; r < rows; r++)
        {
            if (skip[r]) continue;
            for (int i = r * width; i < (r + 1) * width; i++) { double d = (double)a[i] - b[i]; diff += d * d; norm += (double)b[i] * b[i]; }
        }
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

    private static int SecondBest(float[] v, int best)
    {
        int second = best == 0 ? 1 : 0;
        for (int i = 0; i < v.Length; i++) if (i != best && v[i] > v[second]) second = i;
        return second;
    }

    private static int[] TopK(float[] v, int k) => Enumerable.Range(0, v.Length).OrderByDescending(i => v[i]).Take(k).ToArray();
}
