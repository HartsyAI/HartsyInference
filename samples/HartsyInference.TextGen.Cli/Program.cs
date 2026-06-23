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
    Console.WriteLine();
    Console.WriteLine($"--- continuous batch (N={batchN}) ---");
    Console.WriteLine($"  aggregate     : {totalTokens} tokens in {bsec:F2}s ({totalTokens / bsec:F2} tok/s aggregate)");
    Console.WriteLine($"  per-seq tok/s : {totalTokens / bsec / batchN:F2}  (single-seq was {n / sec:F2})");
    Console.WriteLine($"  speedup vs N× single-seq : {(totalTokens / bsec) / (n / sec):F2}x");
    Console.WriteLine($"  token-identical to single-seq: {allMatch}");
    if (cuda is not null)
        Console.WriteLine($"  total D2H syncs: {cuda.GetD2hSyncCount()}");
}

(tokenizer as IDisposable)?.Dispose();
ggufModel?.Dispose();
if (ggufModel is null) model.Dispose();
safeLoader?.Dispose();
return 0;

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
