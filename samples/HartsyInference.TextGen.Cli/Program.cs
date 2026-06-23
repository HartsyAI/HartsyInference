using System.Diagnostics;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

// ── HartsyInference.LLM text-generation harness ──────────────────────────────────────────────────
// Drives the config-driven GenericTransformer through the TextGenerationPipeline on a real checkpoint and
// reports throughput + GPU residency.
//
// Usage:  hartsyinference-textgen [arch] [backend] [genTokens] ["prompt"]
//   arch    = qwen2 | qwen3 | gguf      (default qwen2)
//   backend = cuda | cpu              (default cuda)
//   genTokens (default 64), prompt (default a one-liner)
//
// Model source via HARTSY_MODEL_DIR:
//   arch=qwen2/qwen3 → a safetensors checkpoint folder (defaults to cached VibeVoice-1.5B for qwen2).
//   arch=gguf        → a .gguf file path; config is inferred from GGUF metadata (any Qwen2/Qwen3/Llama LLM).

string arch = args.Length > 0 ? args[0] : "qwen2";
string backendName = args.Length > 1 ? args[1] : "cuda";
int genTokens = args.Length > 2 && int.TryParse(args[2], out int g) ? g : 64;
string prompt = args.Length > 3 ? args[3] : "In one sentence, what is a transformer in machine learning?";

using IBackend backend = backendName == "cpu"
    ? new CpuBackend()
    : new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx"));
CudaBackend? cuda = backend as CudaBackend;

GenericTransformer model;
GgufLanguageModel? ggufModel = null;
SafeTensorsShardLoader? safeLoader = null;
ILlmTokenizer tokenizer;
IChatTemplate? template = null;

if (arch == "gguf")
{
    string ggufPath = Environment.GetEnvironmentVariable("HARTSY_MODEL_DIR")
        ?? throw new InvalidOperationException("Set HARTSY_MODEL_DIR to a .gguf file path for arch=gguf.");
    if (!File.Exists(ggufPath))
    {
        Console.Error.WriteLine($"GGUF file not found: {ggufPath}");
        return 1;
    }
    Console.WriteLine($"=== HartsyInference.LLM — arch=gguf, backend={backendName}, genTokens={genTokens} ===");
    Console.WriteLine($"Loading GGUF {ggufPath} ...");
    bool lowVram = Environment.GetEnvironmentVariable("HARTSY_LOWVRAM") == "1";
    ggufModel = GgufLanguageModel.Load(ggufPath, lowVram);
    model = ggufModel.Transformer;
    TransformerConfig c = ggufModel.Config;
    Console.WriteLine($"  lowVramQuant={lowVram} (set HARTSY_LOWVRAM=1 to keep weights compressed on-device)");
    Console.WriteLine($"  arch={ggufModel.Architecture} hidden={c.HiddenSize} layers={c.NumLayers} heads={c.NumHeads}/{c.NumKvHeads} headDim={c.HeadDim} vocab={c.VocabSize} bias={c.AttentionBias} qkNorm={c.QkNorm} tied={c.TieWordEmbeddings}");
    tokenizer = ggufModel.Tokenizer;
    template = ggufModel.Template;
    Console.WriteLine($"  tokenizer=GGUF-native, chat-template={template.Name}");
}
else
{
    (TransformerConfig cfg, string repoDir, string prefix) = ResolveModel(arch);
    if (!Directory.Exists(repoDir) || Directory.GetFiles(repoDir, "*.safetensors").Length == 0)
    {
        Console.Error.WriteLine($"No safetensors weights found under {repoDir}.");
        Console.Error.WriteLine("Set HARTSY_MODEL_DIR to a Qwen2.5/Qwen3-Instruct checkpoint folder.");
        return 1;
    }
    Console.WriteLine($"=== HartsyInference.LLM — arch={arch}, backend={backendName}, genTokens={genTokens} ===");
    Console.WriteLine($"Loading {repoDir} (prefix '{prefix}') ...");
    safeLoader = new SafeTensorsShardLoader();
    safeLoader.LoadDirectory(repoDir);
    Dictionary<string, Tensor> weights = safeLoader.GetAllTensors();

    // Checkpoints ship bf16/f16 weights. The CUDA backend dequantizes on-device, but the CPU backend is
    // F32-only, so cast up front for cpu (otherwise the F32 kernels would read past the half-width buffers).
    if (cuda is null)
    {
        foreach (string key in weights.Keys.ToList())
            if (weights[key].DType != DType.F32)
                weights[key] = weights[key].CastTo(DType.F32);
    }

    string vocabPath = Path.Combine(repoDir, "vocab.json");
    string mergesPath = Path.Combine(repoDir, "merges.txt");
    tokenizer = File.Exists(vocabPath) && File.Exists(mergesPath)
        ? new Qwen2Tokenizer(vocabPath, mergesPath)
        : new Qwen2Tokenizer();

    GenericTransformer built = new(cfg);
    built.LoadWeights(weights, prefix);
    model = built;
}

if (cuda is not null)
{
    Console.WriteLine("Preloading weights to GPU ...");
    backend.PreloadWeights(model.EnumerateWeights());
}

TextGenerationPipeline pipeline = new(model, tokenizer, backend, template);
GenerationRequest request = new()
{
    Prompt = prompt,
    MaxTokens = genTokens,
    Sampling = SamplingOptions.Default with { Greedy = true },
};
if (Environment.GetEnvironmentVariable("HARTSY_RAW") == "1")
{
    // Bypass the chat template: feed the prompt as raw tokens (BOS + text) to isolate the forward pass.
    int[] raw = tokenizer.Encode(prompt, addSpecial: true);
    Console.WriteLine($"RAW ids ({raw.Length}): {string.Join(",", raw)}");
    request = request with { Prompt = null, RawTokenIds = raw };
}

if (Environment.GetEnvironmentVariable("HARTSY_DEBUG_PROMPT") == "1")
{
    List<ChatMessage> dbgMsgs = [ChatMessage.User(prompt)];
    int[] pids = (template ?? new ChatMlTemplate()).Encode(tokenizer, dbgMsgs, addGenerationPrompt: true);
    Console.WriteLine($"DEBUG prompt ids ({pids.Length}): {string.Join(",", pids)}");
    Console.WriteLine($"DEBUG prompt decode: [{tokenizer.Decode(pids)}]");
}

cuda?.ResetD2hSyncCount();
backend.Sync();
Stopwatch sw = Stopwatch.StartNew();
GenerationResult result = pipeline.Generate(request);
backend.Sync();
sw.Stop();

double sec = sw.Elapsed.TotalSeconds;
int n = Math.Max(1, result.TokenIds.Count);
Console.WriteLine();
Console.WriteLine($"--- {arch} / {backendName} ---");
Console.WriteLine($"  prompt tokens : {result.PromptTokens}");
Console.WriteLine($"  generated     : {result.TokenIds.Count} tokens in {sec:F2}s ({n / sec:F2} tok/s overall)");
Console.WriteLine($"  stopped on EOS: {result.StoppedOnStopToken}");
if (cuda is not null)
    Console.WriteLine($"  total D2H syncs (incl. per-token logits reads): {cuda.GetD2hSyncCount()}");
Console.WriteLine();
Console.WriteLine("=== token ids ===");
Console.WriteLine(string.Join(",", result.TokenIds));
Console.WriteLine("=== text ===");
Console.WriteLine(result.Text);

// ── Optional continuous-batching pass: HARTSY_BATCH=N runs N copies of the prompt through the scheduler,
// verifies each output is token-identical to the single-sequence result above, and reports aggregate tok/s.
if (int.TryParse(Environment.GetEnvironmentVariable("HARTSY_BATCH"), out int batchN) && batchN > 1)
{
    ContinuousBatchScheduler scheduler = new(model, tokenizer, backend, template);
    List<GenerationRequest> batch = new(batchN);
    for (int i = 0; i < batchN; i++) batch.Add(request);

    cuda?.ResetD2hSyncCount();
    backend.Sync();
    Stopwatch bsw = Stopwatch.StartNew();
    IReadOnlyList<GenerationResult> batchResults = scheduler.GenerateBatch(batch);
    backend.Sync();
    bsw.Stop();

    double bsec = bsw.Elapsed.TotalSeconds;
    int totalTokens = 0;
    bool allMatch = true;
    foreach (GenerationResult r in batchResults)
    {
        totalTokens += r.TokenIds.Count;
        if (!r.TokenIds.SequenceEqual(result.TokenIds)) allMatch = false;
    }

    // T1 discriminator: all requests are identical, so the only difference between the batch sequences is
    // their position in the batched GEMM (M-row). If they diverge FROM EACH OTHER the batch path is
    // nondeterministic (a real bug). If they agree with each other but differ from single-seq, the cause is
    // M=B vs M=1 GEMM-kernel numerics (benign). Report seq-to-seq identity and the first divergence step.
    List<int> seq0 = [.. batchResults[0].TokenIds];
    bool seqsMatchEachOther = batchResults.All(r => r.TokenIds.SequenceEqual(seq0));
    int firstSeqDivergence = -1;
    for (int s = 1; s < batchResults.Count && firstSeqDivergence < 0; s++)
    {
        IReadOnlyList<int> rs = batchResults[s].TokenIds;
        int m = Math.Min(seq0.Count, rs.Count);
        for (int t = 0; t < m; t++) if (rs[t] != seq0[t]) { firstSeqDivergence = t; break; }
        if (firstSeqDivergence < 0 && rs.Count != seq0.Count) firstSeqDivergence = m;
    }
    int firstVsSingle = -1;
    {
        int m = Math.Min(seq0.Count, result.TokenIds.Count);
        for (int t = 0; t < m; t++) if (seq0[t] != result.TokenIds[t]) { firstVsSingle = t; break; }
        if (firstVsSingle < 0 && seq0.Count != result.TokenIds.Count) firstVsSingle = m;
    }

    Console.WriteLine();
    Console.WriteLine($"--- continuous batch (N={batchN}) ---");
    Console.WriteLine($"  aggregate     : {totalTokens} tokens in {bsec:F2}s ({totalTokens / bsec:F2} tok/s aggregate)");
    Console.WriteLine($"  per-seq tok/s : {totalTokens / bsec / batchN:F2}  (single-seq was {n / sec:F2})");
    Console.WriteLine($"  speedup vs N× single-seq : {(totalTokens / bsec) / (n / sec):F2}x");
    Console.WriteLine($"  token-identical to single-seq: {allMatch}  (first divergence step: {firstVsSingle})");
    Console.WriteLine($"  batch seqs identical to EACH OTHER: {seqsMatchEachOther}  (first divergence step: {firstSeqDivergence})");
    Console.WriteLine($"  >>> diagnosis: {(seqsMatchEachOther ? (allMatch ? "batch == single (no divergence)" : "deterministic batch, differs from single => GEMM M-variance (numerics)") : "NONDETERMINISTIC batch => real bug")}");
    if (cuda is not null)
        Console.WriteLine($"  total D2H syncs: {cuda.GetD2hSyncCount()}");
}

// ── HARTSY_DIAG=1: logit-level diagnosis of single vs batched decode (T2 bn=1 equivalence, T3 delta).
// Replays the single-seq token trajectory through both the single decode path (Forward, t=1) and the batched
// decode path (ForwardBatchDecode with B copies), comparing the post-projection logit rows step by step.
if (Environment.GetEnvironmentVariable("HARTSY_DIAG") == "1")
{
    TransformerConfig dcfg = model.Config;
    int[] pIds = result.PromptTokens > 0
        ? (template ?? new ChatMlTemplate()).Encode(tokenizer, [ChatMessage.User(prompt)], addGenerationPrompt: true)
        : tokenizer.Encode(prompt, addSpecial: true);
    int diagB = 2;
    int vocab = dcfg.VocabSize;
    int maxSeq = pIds.Length + result.TokenIds.Count + 2;

    // Single path: one FixedKvCache; batch path: diagB caches fed identical tokens.
    using FixedKvCache sCache = new(dcfg.NumLayers, 1, dcfg.NumKvHeads, dcfg.HeadDim, maxSeq);
    FixedKvCache[] bCaches = new FixedKvCache[diagB];
    for (int i = 0; i < diagB; i++) bCaches[i] = new FixedKvCache(dcfg.NumLayers, 1, dcfg.NumKvHeads, dcfg.HeadDim, maxSeq);

    // Prefill both with the prompt.
    int sNext, bNext;
    using (Tensor h = model.Forward(backend, pIds, 0, sCache))
    using (Tensor lg = model.ProjectLogits(backend, h, pIds.Length))
        sNext = ArgMax(lg.AsReadOnlySpan<float>(), (pIds.Length - 1) * vocab, vocab);
    for (int i = 0; i < diagB; i++)
        using (Tensor h = model.Forward(backend, pIds, 0, bCaches[i]))
            _ = h; // prefill only; batch path re-derives its own first token below
    bNext = sNext;

    Console.WriteLine();
    Console.WriteLine($"=== HARTSY_DIAG: single vs batched (B={diagB}) logit comparison ===");
    Console.WriteLine("step  argmax(single)  argmax(batch)  maxAbsDiff   single top1-top2 margin   flip?");
    int flips = 0;
    int steps = Math.Min(result.TokenIds.Count, 32);
    for (int step = 0; step < steps; step++)
    {
        // Single decode of one token.
        float[] sRow;
        using (Tensor h = model.Forward(backend, [sNext], sCache.CurrentLength, sCache))
        using (Tensor lg = model.ProjectLogits(backend, h, 1))
            sRow = lg.AsReadOnlySpan<float>().Slice(0, vocab).ToArray();

        // Batched decode of the SAME token across B identical sequences.
        float[] bRow;
        int[] toks = new int[diagB];
        int[] poss = new int[diagB];
        for (int i = 0; i < diagB; i++) { toks[i] = bNext; poss[i] = bCaches[i].CurrentLength; }
        using (Tensor emb = new(new TensorShape(1, diagB, dcfg.HiddenSize), DType.F32))
        {
            model.EmbedLookup(emb, toks);
            using Tensor h = model.ForwardBatchDecode(backend, emb, poss, bCaches);
            using Tensor lg = model.ProjectLogits(backend, h, diagB);
            bRow = lg.AsReadOnlySpan<float>().Slice(0, vocab).ToArray(); // row 0
        }

        int sArg = ArgMax(sRow, 0, vocab), bArg = ArgMax(bRow, 0, vocab);
        float maxDiff = 0f; for (int j = 0; j < vocab; j++) maxDiff = MathF.Max(maxDiff, MathF.Abs(sRow[j] - bRow[j]));
        float margin = Top1MinusTop2(sRow, vocab);
        bool flip = sArg != bArg; if (flip) flips++;
        if (flip || step < 3 || maxDiff > margin * 0.5f)
            Console.WriteLine($"{step,4}  {sArg,13}  {bArg,12}  {maxDiff,10:E3}   {margin,22:E3}   {(flip ? "FLIP" : "")}");
        sNext = sArg; bNext = bArg; // follow each path's own greedy choice
    }
    Console.WriteLine($"  total argmax flips in {steps} steps: {flips}");
    Console.WriteLine("  interpretation: tiny maxAbsDiff (~1e-3 or less) with flips only where margin <= maxAbsDiff confirms");
    Console.WriteLine("  benign GEMM M-variance (M=B vs M=1 kernel rounding), not a logic bug.");
    for (int i = 0; i < diagB; i++) bCaches[i].Dispose();
}

(tokenizer as IDisposable)?.Dispose();
ggufModel?.Dispose();
if (ggufModel is null) model.Dispose();
safeLoader?.Dispose();
return 0;

static int ArgMax(ReadOnlySpan<float> v, int offset, int n)
{
    int best = 0; float bv = v[offset];
    for (int i = 1; i < n; i++) { float x = v[offset + i]; if (x > bv) { bv = x; best = i; } }
    return best;
}

static float Top1MinusTop2(ReadOnlySpan<float> v, int n)
{
    float t1 = float.NegativeInfinity, t2 = float.NegativeInfinity;
    for (int i = 0; i < n; i++) { float x = v[i]; if (x > t1) { t2 = t1; t1 = x; } else if (x > t2) t2 = x; }
    return t1 - t2;
}

static (TransformerConfig, string, string) ResolveModel(string arch)
{
    string? overrideDir = Environment.GetEnvironmentVariable("HARTSY_MODEL_DIR");
    if (arch == "qwen3")
        return (TransformerConfig.Qwen3_0_6B, overrideDir ?? "/tmp/qwen3-06b", "model");

    // qwen2: explicit dir → Qwen2.5-0.5B preset; else the cached VibeVoice-1.5B backbone.
    if (!string.IsNullOrEmpty(overrideDir))
        return (TransformerConfig.Qwen2_5_0_5B, overrideDir, "model");

    string vibe = AudioModelCache.GetRepoDirectory("vibevoice/VibeVoice-1.5B");
    string legacy = vibe.Replace("hartsyinference", "sharpinference");
    if (!File.Exists(Path.Combine(vibe, "model-00001-of-00003.safetensors"))
        && File.Exists(Path.Combine(legacy, "model-00001-of-00003.safetensors")))
        vibe = legacy;
    return (TransformerConfig.Qwen2_5_1_5B, vibe, "model.language_model");
}
