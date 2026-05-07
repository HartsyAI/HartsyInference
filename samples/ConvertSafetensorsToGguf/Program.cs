using SharpInference.ModelHandler.Gguf;

if (args.Length < 3)
{
    PrintUsage();
    return 1;
}

string input = args[0];
string output = args[1];
string policyName = args[2];
string architecture = args.Length >= 4 ? args[3] : "passthrough";

GgufQuantPolicy policy = policyName.ToLowerInvariant() switch
{
    "q8_0" => GgufQuantPolicy.Q8_0,
    "q4_k_s" => GgufQuantPolicy.Q4_K_S,
    "q4_k_m" => GgufQuantPolicy.Q4_K_M,
    "q5_k_m" => GgufQuantPolicy.Q5_K_M,
    "q6_k" => GgufQuantPolicy.Q6_K,
    _ => Unsupported(policyName),
};

if (!File.Exists(input))
{
    Console.Error.WriteLine($"Input file not found: {input}");
    return 2;
}

Console.WriteLine($"[convert-safetensors-to-gguf] {Path.GetFileName(input)} → {Path.GetFileName(output)} [{policyName.ToUpperInvariant()}] arch={architecture}");
DateTime start = DateTime.UtcNow;

GgufQuantizationReport report = GgufQuantizer.ConvertSafetensorsToGguf(
    input, output, policy, architecture);

TimeSpan elapsed = DateTime.UtcNow - start;
Console.WriteLine($"  passthrough: {report.PassthroughCount}");
Console.WriteLine($"  cast (→ F16/F32): {report.CastCount}");
Console.WriteLine($"  quantized: {report.QuantizedCount}");
foreach (KeyValuePair<string, int> kv in report.QuantTotals)
{
    Console.WriteLine($"    {kv.Key}: {kv.Value} tensors");
}
Console.WriteLine($"  output: {report.OutputBytes / (1024 * 1024)} MB");
Console.WriteLine($"  elapsed: {elapsed.TotalSeconds:F1}s");
return 0;

static void PrintUsage()
{
    Console.Error.WriteLine("Usage: convert-safetensors-to-gguf <input.safetensors> <output.gguf> <policy> [architecture]");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Policies (output size approx. vs F16 source):");
    Console.Error.WriteLine("  q8_0    — uniform Q8_0 on backbone, F16 norms                    ~50% of F16");
    Console.Error.WriteLine("  q4_k_s  — uniform Q4_K, F16 norms                                ~25% of F16  (cheapest)");
    Console.Error.WriteLine("  q4_k_m  — Q4_K backbone + Q6_K for V/output                      ~30% of F16  (most popular)");
    Console.Error.WriteLine("  q5_k_m  — Q5_K backbone + Q6_K for V/output                      ~37% of F16");
    Console.Error.WriteLine("  q6_k    — uniform Q6_K, F16 norms                                ~44% of F16  (near-lossless)");
    Console.Error.WriteLine();
    Console.Error.WriteLine("These five policies are the only supported authoring modes. Q4_0/Q4_1/Q5_0/Q5_1/Q8_1/");
    Console.Error.WriteLine("Q2_K/Q3_K and the i-quant family are READ-ONLY — use llama.cpp's `quantize` tool to author them.");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Architecture (sets `general.architecture` metadata; case-insensitive):");
    Console.Error.WriteLine("  flux | flux2 | sdxl | sd3 | sd15 | flite | chroma | auraflow | zimage |");
    Console.Error.WriteLine("  ernie_image | hunyuan_image | qwen_image | llama | passthrough");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Example:");
    Console.Error.WriteLine("  convert-safetensors-to-gguf flux1-schnell.safetensors flux1-schnell-Q4_K_M.gguf q4_k_m flux");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Full docs: docs/Research/GGUF_QUANTIZER_USAGE.md");
}

static GgufQuantPolicy Unsupported(string name)
{
    Console.Error.WriteLine($"Unknown policy '{name}'. Supported: q8_0 | q4_k_s | q4_k_m | q5_k_m | q6_k.");
    Console.Error.WriteLine("(Other ggml types are read-only — see docs/Research/GGUF_QUANTIZER_USAGE.md for details.)");
    Environment.Exit(3);
    return null!;
}
