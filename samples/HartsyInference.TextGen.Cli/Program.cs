using System.Diagnostics;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
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
//   arch    = qwen2 | qwen3            (default qwen2)
//   backend = cuda | cpu              (default cuda)
//   genTokens (default 64), prompt (default a one-liner)
//
// Model dir: HARTSY_MODEL_DIR (a Qwen2.5/Qwen3 checkpoint folder). Defaults to the cached VibeVoice-1.5B
// Qwen2.5 backbone when arch=qwen2 and no dir is set (proves residency, but emits TTS codec tokens).

string arch = args.Length > 0 ? args[0] : "qwen2";
string backendName = args.Length > 1 ? args[1] : "cuda";
int genTokens = args.Length > 2 && int.TryParse(args[2], out int g) ? g : 64;
string prompt = args.Length > 3 ? args[3] : "In one sentence, what is a transformer in machine learning?";

(TransformerConfig cfg, string repoDir, string prefix) = ResolveModel(arch);
if (!Directory.Exists(repoDir) || Directory.GetFiles(repoDir, "*.safetensors").Length == 0)
{
    Console.Error.WriteLine($"No safetensors weights found under {repoDir}.");
    Console.Error.WriteLine("Set HARTSY_MODEL_DIR to a Qwen2.5/Qwen3-Instruct checkpoint folder.");
    return 1;
}

Console.WriteLine($"=== HartsyInference.LLM — arch={arch}, backend={backendName}, genTokens={genTokens} ===");
Console.WriteLine($"Loading {repoDir} (prefix '{prefix}') ...");
using SafeTensorsShardLoader loader = new();
loader.LoadDirectory(repoDir);
Dictionary<string, Tensor> weights = loader.GetAllTensors();

string vocabPath = Path.Combine(repoDir, "vocab.json");
string mergesPath = Path.Combine(repoDir, "merges.txt");
using Qwen2Tokenizer tokenizer = File.Exists(vocabPath) && File.Exists(mergesPath)
    ? new Qwen2Tokenizer(vocabPath, mergesPath)
    : new Qwen2Tokenizer();

using IBackend backend = backendName == "cpu"
    ? new CpuBackend()
    : new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx"));
CudaBackend? cuda = backend as CudaBackend;

using GenericTransformer model = new(cfg);
model.LoadWeights(weights, prefix);
if (cuda is not null)
{
    Console.WriteLine("Preloading weights to GPU ...");
    backend.PreloadWeights(model.EnumerateWeights());
}

TextGenerationPipeline pipeline = new(model, tokenizer, backend);
GenerationRequest request = new()
{
    Prompt = prompt,
    MaxTokens = genTokens,
    Sampling = SamplingOptions.Default with { Greedy = true },
};

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
