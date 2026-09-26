using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
using HartsyInference.Core.Tensors;
using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

const string HelpText = """
CheckpointRepacker - turn a model checkpoint into a self-describing .safetensors file.

  PyTorch pickles (.pt .pth .bin .th .ckpt) are converted. An existing .safetensors keeps its tensors
  byte-for-byte and only has its identity metadata rewritten. Either way the output carries the
  architecture, title, author and license SwarmUI and Hartsy read, plus a .manifest.json with hashes.

USAGE
  CheckpointRepacker <input> <output.safetensors | output-folder/> --model <id> [options]
  CheckpointRepacker <input> [<input> ...] <output> --model <id> [--dtype bf16] [options]
  CheckpointRepacker --list-models

  Give a folder as the output and the file is named for you, the way Hartsy names models:
  <model>[-<variant>][-<part>]_<precision>.safetensors, e.g. whisper-large-v3_fp16.safetensors.

IDENTITY
  --model <id>          Engine model id, e.g. kokoro, whisper, dia. See --list-models. Required.
  --variant <name>      The AudioLab model id, e.g. large-v3, 1.7B-Base, default. AudioLab only lists a
                        file whose header names it. Also goes in the title and file name.
  --component <name>    main (default) for the weights a user selects. For a companion file name its
                        part instead (codec, vocoder, voice, tokenizer...): companions get no
                        architecture, so they never show up as a model of their own.
  --source-repo <repo>  Upstream repo to record as provenance. Defaults to the model's known upstream.
  --meta KEY=VALUE      Add or override one metadata entry. Repeatable.
  --meta-json <file>    Add or override entries from a JSON object of strings.
  --no-metadata         Write no identity. SwarmUI then cannot classify the file; not for publishing.
  --stands-in-for <path>  An upstream file this one replaces, as a path under the audio models root (the
                        engine's own cache layout). Repeatable. The engine links the file into each such path
                        that is missing, so its loaders find it without knowing its Hartsy name.
                        e.g. --stands-in-for tts/nari-labs--Dia-1.6B-0626/pytorch_model.bin

MERGE AND CAST (safetensors)
  Several inputs, a *.safetensors.index.json, or a folder of shards are merged into one file, streaming
  tensor by tensor. Keys must not collide across inputs.
  --dtype <bf16|fp16|fp32>  Cast float tensors (F32/BF16/F16), rounding to nearest even exactly as PyTorch
                        does, so the result matches a torch-made repack bit for bit. Integer tensors pass.
  --prefix <p>          Prepend <p> to every key, e.g. model.diffusion_model. when bundling components.
  --drop <pattern>      Leave out tensors whose key matches (* and ? wildcards). Repeatable. e.g. lm_head.weight
                        for a head tied to the embeddings.
  --keep-dtype <pattern>  Keep matching tensors in their stored dtype through --dtype. Repeatable.
                        e.g. --keep-dtype '*norm*' keeps norms in F32 under a bf16 cast.

LORA BAKING (safetensors)
  --lora <file>[:<strength>]  Merge an adapter into the weights it modifies and write a standalone model.
                        Repeatable; applied in order. Reads PEFT, kohya, diffusers and LyCORIS (LoHa, LoKr,
                        DoRA, full-weight diffs) naming, and matches each module to a weight by name. A PEFT
                        adapter_config.json beside the file supplies alpha and rsLoRA. The merge is the float32
                        arithmetic single-threaded PyTorch does, so it matches such a merge bit for bit.
  --lora-alpha <a>      Alpha for every module when neither the file nor an adapter_config.json has one.

KEY LAYOUT (pickles only)
  --recursive           Keep every nested state dict, prefixing keys with the dict name (bert.x ...).
  --allow-partial       Keep only the first nested state dict even though others hold tensors.
  --strip-prefix <p>    Remove a leading <p> from every key.
  --rename FROM=TO      Replace FROM with TO anywhere in a key. Repeatable. e.g. --rename .module.=.

RECIPES
  --recipe <file.json>  Build the output from a checked-in recipe: components (with prefixes and regex renames),
                        fused tensors, copies, drops and embedded files or tokenizers. Recipe fields supply
                        --model/--variant/--component/--source-repo/--dtype unless given here. Takes only the output path.
  --source-root <dir>   Where a recipe's source_file paths resolve. Default: the recipe's folder.

TOKENIZER
  --tokenizer-from-tiktoken <file.tiktoken> <tokenizer.json>
                        Convert a tiktoken rank file to a HuggingFace tokenizers JSON (byte-level BPE with
                        merges recovered from the ranks), identical to what transformers' converter writes.
  --pattern <regex>     Pre-tokenizer split pattern. Defaults to the Qwen2 pattern.
  --normalizer <NFC|none>  Unicode normalizer. Default NFC.

OUTPUT
  --force               Replace the output if it already exists.
  -h, --help            Show this help.

EXAMPLES
  CheckpointRepacker kokoro-v1_0.pth kokoro-82m.safetensors --model kokoro --recursive --rename .module.=.
  CheckpointRepacker model.safetensors out/model.safetensors --model whisper --variant large-v3
  CheckpointRepacker dac.pth dac.safetensors --model dia --component codec
  CheckpointRepacker xl-turbo/model.safetensors.index.json out/ --model acestep --variant xl-turbo --dtype bf16
  CheckpointRepacker base.safetensors out/ --model orpheus --lora adapter_model.safetensors:0.8 --dtype bf16
""";

// The tool prints its own progress; the engine's info lines would repeat it.
Logs.MinLevel = LogLevel.Warning;
string[] pickleExtensions = [".pt", ".pth", ".bin", ".th", ".ckpt"];
string[] valueOptions = ["--model", "--variant", "--component", "--source-repo", "--meta", "--meta-json", "--strip-prefix", "--rename", "--dtype", "--prefix", "--drop", "--keep-dtype", "--pattern", "--normalizer", "--recipe", "--source-root", "--lora", "--lora-alpha", "--stands-in-for"];
string[] flagOptions = ["--recursive", "--allow-partial", "--no-metadata", "--force", "--help", "-h", "--list-models", "--tokenizer-from-tiktoken"];

List<string> positionals = [];
Dictionary<string, List<string>> values = new(StringComparer.Ordinal);
HashSet<string> flags = new(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i++)
{
    string arg = args[i];
    if (!arg.StartsWith('-') || arg == "-")
    {
        positionals.Add(arg);
    }
    else if (flagOptions.Contains(arg))
    {
        flags.Add(arg);
    }
    else if (valueOptions.Contains(arg))
    {
        if (i + 1 >= args.Length)
        {
            return Usage($"{arg} needs a value.");
        }
        if (!values.TryGetValue(arg, out List<string>? list))
        {
            values[arg] = list = [];
        }
        list.Add(args[++i]);
    }
    else
    {
        string? near = valueOptions.Concat(flagOptions).Where(o => o.StartsWith("--", StringComparison.Ordinal))
            .OrderBy(o => Distance(arg, o)).FirstOrDefault(o => Distance(arg, o) <= 3);
        return Usage(near is null ? $"Unknown option '{arg}'." : $"Unknown option '{arg}'. Did you mean '{near}'?");
    }
}

if (flags.Contains("--help") || flags.Contains("-h"))
{
    Console.WriteLine(HelpText);
    return 0;
}
if (flags.Contains("--list-models"))
{
    Console.WriteLine($"{"id",-20} {"class (modelspec.architecture)",-30} {"license",-18} name");
    foreach (ArtifactIdentity id in ModelIdentityCatalog.All.Values.OrderBy(v => v.Tags.FirstOrDefault()).ThenBy(v => v.EngineId))
    {
        string variants = id.VariantClassIds.Count > 0 ? $"  [variants: {string.Join(", ", id.VariantClassIds.Keys)}]" : "";
        Console.WriteLine($"{id.EngineId,-20} {id.SwarmClassId,-30} {id.License,-18} {id.DisplayName}{variants}");
    }
    return 0;
}
if (flags.Contains("--tokenizer-from-tiktoken"))
{
    if (positionals.Count != 2)
    {
        return Usage("--tokenizer-from-tiktoken takes a .tiktoken file and an output .json path.");
    }
    const string Qwen2Pattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    string? norm = values.GetValueOrDefault("--normalizer")?.Last();
    byte[] tokenizerJson = TiktokenConverter.ToHuggingFaceJson(positionals[0], values.GetValueOrDefault("--pattern")?.Last() ?? Qwen2Pattern,
        norm is null ? "NFC" : norm.Equals("none", StringComparison.OrdinalIgnoreCase) ? null : norm);
    if (File.Exists(positionals[1]) && !flags.Contains("--force"))
    {
        Console.Error.WriteLine($"Output already exists: {positionals[1]}. Add --force to replace it.");
        return 1;
    }
    File.WriteAllBytes(positionals[1], tokenizerJson);
    Console.WriteLine($"Wrote {positionals[1]} ({Size(tokenizerJson.Length)}), sha256 {System.Convert.ToHexString(SHA256.HashData(tokenizerJson)).ToLowerInvariant()}");
    return 0;
}
Recipe? recipe = null;
string recipeRoot = "";
string recipeDir = "";
if (values.GetValueOrDefault("--recipe")?.Last() is string recipePath)
{
    if (positionals.Count != 1)
    {
        return Usage("--recipe takes only the output path; the recipe names its own sources.");
    }
    try
    {
        recipe = Recipe.Load(recipePath);
    }
    catch (Exception ex) when (ex is IOException or JsonException or InvalidDataException)
    {
        Console.Error.WriteLine($"Could not read recipe {recipePath}: {ex.Message}");
        return 1;
    }
    recipeDir = Path.GetDirectoryName(Path.GetFullPath(recipePath))!;
    recipeRoot = values.GetValueOrDefault("--source-root")?.Last() ?? recipeDir;
    void Default(string option, string? value)
    {
        if (value is not null && !values.ContainsKey(option))
            values[option] = [value];
    }
    Default("--model", recipe.Model);
    Default("--variant", recipe.Variant);
    Default("--component", recipe.Component);
    Default("--source-repo", recipe.SourceRepo);
    Default("--dtype", recipe.Dtype);
    if (recipe.KeepDtype.Count > 0 && !values.ContainsKey("--keep-dtype"))
        values["--keep-dtype"] = [.. recipe.KeepDtype];
}
else if (positionals.Count < 2)
{
    return Usage("Give an input checkpoint and an output .safetensors path.");
}

List<string> inputs = [.. positionals.Take(positionals.Count - 1).Select(Path.GetFullPath)];
string input = recipe is null ? inputs[0] : Path.GetFullPath(Path.Combine(recipeRoot, recipe.Components.FirstOrDefault()?.SourceFile ?? ""));
string outputArg = positionals[^1];
string output = Path.GetFullPath(outputArg);
bool autoName = outputArg.EndsWith('/') || outputArg.EndsWith('\\') || Directory.Exists(output);
string? dtypeName = values.GetValueOrDefault("--dtype")?.Last()?.ToLowerInvariant();
DType? castTo = dtypeName switch
{
    null => null,
    "bf16" => DType.BF16,
    "fp16" or "f16" => DType.F16,
    "fp32" or "f32" => DType.F32,
    _ => null,
};
if (dtypeName is not null && castTo is null)
{
    return Usage($"--dtype '{dtypeName}' is not supported; use bf16, fp16 or fp32.");
}
bool IsShardSet(string p) => p.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase)
    || (Directory.Exists(p) && Directory.EnumerateFiles(p, "*.safetensors").Any());
List<(string Path, float Strength, float? Alpha, bool RsLora)> loras = [];
float? loraAlpha = null;
if (values.GetValueOrDefault("--lora-alpha")?.Last() is string alphaText)
{
    if (!float.TryParse(alphaText, NumberStyles.Float, CultureInfo.InvariantCulture, out float a) || !(a > 0))
        return Usage($"--lora-alpha '{alphaText}' is not a positive number.");
    loraAlpha = a;
}
foreach (string spec in values.GetValueOrDefault("--lora") ?? [])
{
    int colon = spec.LastIndexOf(':');
    float strength = 1.0f;
    string file = spec;
    if (colon > 1 && float.TryParse(spec[(colon + 1)..], NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed))
    {
        strength = parsed;
        file = spec[..colon];
    }
    file = Path.GetFullPath(file);
    if (!File.Exists(file) || !file.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
    {
        Console.Error.WriteLine(File.Exists(file) ? $"--lora takes a .safetensors adapter; convert '{Path.GetFileName(file)}' first." : $"LoRA not found: {file}");
        return 1;
    }
    (float? alpha, bool rs, string? problem) = ReadPeftConfig(Path.Combine(Path.GetDirectoryName(file)!, "adapter_config.json"));
    if (problem is not null)
    {
        Console.Error.WriteLine(problem);
        return 1;
    }
    loras.Add((file, strength, loraAlpha ?? alpha, rs));
}
bool mergeMode = recipe is not null || inputs.Count > 1 || castTo is not null || values.ContainsKey("--prefix") || values.ContainsKey("--drop")
    || loras.Count > 0 || inputs.Any(IsShardSet);
if (mergeMode)
{
    foreach (string p in inputs)
    {
        if (!IsShardSet(p) && !(File.Exists(p) && p.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)))
        {
            Console.Error.WriteLine(File.Exists(p) || Directory.Exists(p)
                ? $"'{Path.GetFileName(p)}' can't be merged: merging takes .safetensors files, a *.safetensors.index.json or a folder of shards. Convert a pickle first."
                : $"Input not found: {p}");
            return 1;
        }
    }
}
if (!mergeMode && Directory.Exists(input))
{
    string[] candidates = Directory.EnumerateFiles(input, "*", SearchOption.AllDirectories)
        .Where(f => pickleExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()) || f.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        .Select(f => Path.GetRelativePath(input, f)).Order().Take(20).ToArray();
    Console.Error.WriteLine($"'{input}' is a folder. Pass one checkpoint file inside it.");
    if (candidates.Length > 0)
    {
        Console.Error.WriteLine("Checkpoints found there:");
        foreach (string c in candidates)
        {
            Console.Error.WriteLine($"  {c}");
        }
    }
    return 1;
}
if (!mergeMode && !File.Exists(input))
{
    Console.Error.WriteLine($"Input not found: {input}");
    return 1;
}
string inputExtension = Path.GetExtension(input).ToLowerInvariant();
bool isSafeTensors = mergeMode || inputExtension == ".safetensors";
if (!isSafeTensors && !pickleExtensions.Contains(inputExtension))
{
    Console.Error.WriteLine($"'{Path.GetFileName(input)}' is not a supported checkpoint. Expected .safetensors or one of {string.Join(" ", pickleExtensions)}.");
    Console.Error.WriteLine(inputExtension is ".onnx" or ".gguf"
        ? "ONNX and GGUF stay in their own format; describe them with a .swarm.json sidecar (tools/repack/stamp_modelspec.py --sidecar-only)."
        : "");
    return 1;
}
if (!autoName && !output.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
{
    return Usage($"The output must end in .safetensors: {outputArg}");
}
if (!autoName && File.Exists(output) && !flags.Contains("--force"))
{
    Console.Error.WriteLine($"Output already exists: {output}");
    Console.Error.WriteLine("Add --force to replace it.");
    return 1;
}
if (isSafeTensors && (flags.Contains("--recursive") || flags.Contains("--allow-partial")
    || (!mergeMode && (values.ContainsKey("--strip-prefix") || values.ContainsKey("--rename")))))
{
    return Usage(mergeMode ? "--recursive and --allow-partial apply to pickles; merged safetensors keep their keys (use --rename or --strip-prefix)."
        : "Key layout options only apply to pickle checkpoints; a .safetensors input keeps its tensors as they are. Add --dtype or --prefix to rewrite it.");
}

Dictionary<string, string>? metadata = null;
ArtifactIdentity? resolvedIdentity = null;
string? modelId = values.GetValueOrDefault("--model")?.Last();
string component = values.GetValueOrDefault("--component")?.Last() ?? ArtifactProvenance.MainComponent;
string? variant = values.GetValueOrDefault("--variant")?.Last();
if (!flags.Contains("--no-metadata"))
{
    if (modelId is null)
    {
        return Usage("Say which model this is with --model <id> (see --list-models), or pass --no-metadata to write none.");
    }
    ArtifactIdentity? identity = ModelIdentityCatalog.Find(modelId);
    if (identity is null)
    {
        string? near = ModelIdentityCatalog.All.Keys.OrderBy(k => Distance(modelId, k)).FirstOrDefault(k => Distance(modelId, k) <= 3);
        Console.Error.WriteLine($"Unknown model id '{modelId}'.{(near is null ? "" : $" Did you mean '{near}'?")} Run --list-models for the full list.");
        return 1;
    }
    if (identity.VariantClassIds.Count > 0 && component == ArtifactProvenance.MainComponent && variant is null)
    {
        Console.Error.WriteLine($"Some '{modelId}' variants register as their own class ({string.Join(", ", identity.VariantClassIds.Keys)});");
        Console.Error.WriteLine("pass --variant so the right one is stamped. Other variants use the family class.");
        return 1;
    }
    resolvedIdentity = identity.ForVariant(variant);
}
Dictionary<string, string> overrides = new(StringComparer.Ordinal);
foreach (string file in values.GetValueOrDefault("--meta-json") ?? [])
{
    Dictionary<string, string>? loaded;
    try
    {
        loaded = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(file));
    }
    catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
    {
        Console.Error.WriteLine($"Could not read --meta-json {file}: {ex.Message}");
        return 1;
    }
    foreach (KeyValuePair<string, string> entry in loaded ?? [])
    {
        overrides[entry.Key] = entry.Value;
    }
}
foreach (string pair in values.GetValueOrDefault("--meta") ?? [])
{
    int split = pair.IndexOf('=', StringComparison.Ordinal);
    if (split <= 0)
    {
        return Usage($"--meta '{pair}' should look like KEY=VALUE.");
    }
    overrides[pair[..split]] = pair[(split + 1)..];
}

List<string> standsInFor = [.. values.GetValueOrDefault("--stands-in-for") ?? recipe?.StandsInFor ?? []];
foreach (string path in standsInFor)
{
    if (Path.IsPathRooted(path) || path.Replace('\\', '/').Split('/').Any(s => s is ".." or "." or ""))
        return Usage($"--stands-in-for '{path}' must be a path under the audio models root, e.g. tts/nari-labs--Dia-1.6B-0626/pytorch_model.bin.");
}
if (standsInFor.Count > 0)
    overrides["hartsy.stands_in_for"] = string.Join(";", standsInFor.Select(p => p.Replace('\\', '/')));

List<(string From, string To)> renames = [];
foreach (string pair in values.GetValueOrDefault("--rename") ?? [])
{
    int split = pair.IndexOf('=', StringComparison.Ordinal);
    if (split <= 0)
    {
        return Usage($"--rename '{pair}' should look like FROM=TO.");
    }
    renames.Add((pair[..split], pair[(split + 1)..]));
}
string? stripPrefix = values.GetValueOrDefault("--strip-prefix")?.Last();
bool layoutGiven = flags.Contains("--recursive") || flags.Contains("--allow-partial") || stripPrefix is not null || renames.Count > 0;
// Kokoro's main checkpoint is a dict of DataParallel-wrapped modules; this is the layout its loader reads.
if (!isSafeTensors && !layoutGiven && modelId == "kokoro" && component == ArtifactProvenance.MainComponent)
{
    Console.WriteLine("Using the kokoro layout: --recursive --rename .module.=.");
    flags.Add("--recursive");
    renames.Add((".module.", "."));
}

List<string> mergeSources = [];

// Built after the load succeeds: hashing a multi-gigabyte source for provenance is wasted on a file that won't convert.
Dictionary<string, string>? BuildMetadata()
{
    Dictionary<string, string>? built = null;
    if (resolvedIdentity is not null)
    {
        ArtifactProvenance provenance;
        if (mergeMode)
        {
            Console.WriteLine($"Hashing {mergeSources.Count} source file(s) for provenance...");
            string combined = string.Join('\n', mergeSources.Select(f => $"{Path.GetFileName(f)}:{ArtifactProvenance.HashFile(f)}"));
            provenance = new ArtifactProvenance
            {
                Converter = "HartsyInference.SafeTensorsMerger",
                Component = component,
                SourceRepo = values.GetValueOrDefault("--source-repo")?.Last(),
                SourceFile = string.Join(",", mergeSources.Select(SourceName)),
                SourceSha256 = mergeSources.Count == 1 ? ArtifactProvenance.HashFile(mergeSources[0])
                    : System.Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(combined))).ToLowerInvariant(),
                Precision = dtypeName,
                ModelId = variant,
            };
        }
        else
        {
            Console.WriteLine($"Hashing source {Path.GetFileName(input)} for provenance...");
            string converter = isSafeTensors ? "upstream-safetensors" : "HartsyInference.PickleCheckpointRepacker";
            provenance = ArtifactProvenance.FromSourceFile(converter, component, input,
                values.GetValueOrDefault("--source-repo")?.Last()) with { ModelId = variant };
        }
        built = ArtifactMetadata.ForRepack(resolvedIdentity, provenance);
        if (component != ArtifactProvenance.MainComponent && PartName() is string part)
        {
            built["modelspec.title"] = $"{resolvedIdentity.DisplayName} ({component}: {part})";
        }
        if (variant is not null)
        {
            if (component == ArtifactProvenance.MainComponent
                && !Alnum(resolvedIdentity.DisplayName).Contains(Alnum(variant), StringComparison.Ordinal))
            {
                built["modelspec.title"] = $"{resolvedIdentity.DisplayName} {variant}";
            }
        }
    }
    foreach (KeyValuePair<string, string> entry in overrides)
    {
        (built ??= new(StringComparer.Ordinal))[entry.Key] = entry.Value;
    }
    if (built is not null && !built.ContainsKey(ArtifactMetadata.HashKey))
    {
        built[ArtifactMetadata.HashKey] = "";
    }
    return built;
}

// A companion's own file stem, when it says more than "model": af_heart, t5-large. Generic stems name nothing.
string? PartName()
{
    string stem = ShardPattern().Replace(Path.GetFileNameWithoutExtension(input), "");
    string[] generic = ["model", "weights", "pytorch_model", "diffusion_pytorch_model", "checkpoint", "state_dict"];
    return generic.Contains(stem, StringComparer.OrdinalIgnoreCase) || stem.Equals(component, StringComparison.OrdinalIgnoreCase) ? null : stem;
}

// Names the output once the tensors' dtypes are known; null means "stop, already reported".
string? ResolveOutput(IEnumerable<(DType DType, long Elements)> tensors)
{
    if (!autoName)
    {
        return output;
    }
    if (resolvedIdentity is null)
    {
        Console.Error.WriteLine("Naming the file for you needs --model; give an explicit output file name with --no-metadata.");
        return null;
    }
    string precision = Precision(tensors);
    string? variantSlot = variant is null || variant.Equals("default", StringComparison.OrdinalIgnoreCase)
        || variant.Equals(precision, StringComparison.OrdinalIgnoreCase) ? null : variant;
    // Continuation shards share the first shard's stem so the set reads as one file split N ways.
    string? partSlot = component is ArtifactProvenance.MainComponent or "shard" ? null
        : PartName() is string part ? $"{component}-{part}" : component;
    string name = ArtifactNaming.FileName(resolvedIdentity.EngineId, variantSlot, precision, ".safetensors", partSlot);
    Match shard = ShardPattern().Match(Path.GetFileNameWithoutExtension(input));
    if (shard.Success)
    {
        name = name[..^".safetensors".Length] + shard.Value + ".safetensors";
    }
    string resolved = Path.Combine(output, name);
    if (File.Exists(resolved) && !flags.Contains("--force"))
    {
        Console.Error.WriteLine($"Output already exists: {resolved}");
        Console.Error.WriteLine("Add --force to replace it.");
        return null;
    }
    return resolved;
}

if (resolvedIdentity?.ProviderId is not null && component == ArtifactProvenance.MainComponent && variant is null)
{
    Console.WriteLine("Note: no --variant given. AudioLab lists a file only when its header names the model id");
    Console.WriteLine("      (e.g. --variant default, --variant large-v3); this one will be classified but not selectable.");
}

Stopwatch clock = Stopwatch.StartNew();
int tensorCount;
string payloadSha;
try
{
    if (mergeMode)
    {
        List<SafeTensorsLoader> loaders = [];
        List<Tensor> ownedTensors = [];
        using CpuBackend cpu = new();
        try
        {
            List<KeyValuePair<string, Tensor>> all = [];
            string prefix = values.GetValueOrDefault("--prefix")?.Last() ?? "";
            List<Regex> drops = [.. (values.GetValueOrDefault("--drop") ?? []).Select(Glob)];
            List<Regex> keeps = [.. (values.GetValueOrDefault("--keep-dtype") ?? []).Select(Glob)];
            int dropped = 0;
            List<Tensor> owned = [];
            if (recipe is not null)
            {
                Console.WriteLine($"Building '{recipe.Name}' from its recipe:");
                all.AddRange(recipe.Build(recipeDir, recipeRoot, loaders, owned, mergeSources, line => Console.WriteLine(line), cpu));
                foreach ((string key, string value) in recipe.Metadata)
                    overrides.TryAdd(key, value);
            }
            ownedTensors = owned;
            foreach (string p in inputs)
            {
                (List<KeyValuePair<string, Tensor>> tensors, List<SafeTensorsLoader> opened) = SafeTensorsMerger.Open(p);
                loaders.AddRange(opened);
                foreach ((string key, Tensor tensor) in tensors)
                {
                    string mapped = stripPrefix is not null && key.StartsWith(stripPrefix, StringComparison.Ordinal) ? key[stripPrefix.Length..] : key;
                    foreach ((string from, string to) in renames)
                        mapped = mapped.Replace(from, to, StringComparison.Ordinal);
                    if (drops.Any(d => d.IsMatch(mapped)))
                    {
                        dropped++;
                        continue;
                    }
                    all.Add(new(prefix + mapped, tensor));
                }
            }
            if (recipe is null)
                mergeSources.AddRange(loaders.Select(l => l.FilePath));
            foreach ((string file, float strength, float? alpha, bool rs) in loras)
            {
                (List<KeyValuePair<string, Tensor>> adapter, List<SafeTensorsLoader> opened) = SafeTensorsMerger.Open(file);
                loaders.AddRange(opened);
                mergeSources.Add(file);
                List<string> stray = [];
                List<LoraBaker.Patch> patches = LoraBaker.Group(adapter, stray);
                if (patches.Count == 0)
                {
                    Console.Error.WriteLine($"{Path.GetFileName(file)} has no LoRA tensors (keys like {string.Join(", ", stray.Take(3))}).");
                    return 1;
                }
                if (stray.Count > 0)
                    Console.WriteLine($"  {Path.GetFileName(file)}: ignoring {stray.Count} key(s) that are not adapter weights, e.g. {stray[0]}");
                Dictionary<string, Tensor> byKey = new(all, StringComparer.Ordinal);
                int changed = LoraBaker.Apply(byKey, patches, LoraBaker.AutoTargets(byKey.Keys, patches.Select(p => p.Root)),
                    new LoraBaker.Options { Strength = strength, Alpha = alpha, RsLora = rs, Backend = cpu }, owned);
                all = [.. all.Select(kv => new KeyValuePair<string, Tensor>(kv.Key, byKey[kv.Key]))];
                Console.WriteLine($"Baked {Path.GetFileName(file)} ({patches.Count} module(s), strength {strength.ToString(CultureInfo.InvariantCulture)}"
                    + (alpha is { } a ? $", alpha {a.ToString(CultureInfo.InvariantCulture)}" : "") + (rs ? ", rsLoRA" : "") + $") into {changed} weight(s).");
            }
            if (drops.Count > 0)
            {
                Console.WriteLine($"Dropping {dropped} tensor(s) matching {string.Join(", ", values["--drop"])}.");
                if (dropped == 0)
                {
                    Console.Error.WriteLine("No tensor matched --drop; check the pattern against the source keys.");
                    return 1;
                }
            }
            bool Keep(string key) => keeps.Any(k => k.IsMatch(key));
            if (ResolveOutput(all.Select(kv => (castTo is { } c && SafeTensorsMerger.IsCastable(kv.Value.DType) && !Keep(kv.Key) ? c : kv.Value.DType,
                kv.Value.Shape.ElementCount))) is not string named)
            {
                return 1;
            }
            output = named;
            metadata = BuildMetadata();
            long inBytes = mergeSources.Sum(f => new FileInfo(f).Length);
            Console.WriteLine($"Writing {all.Count} tensors from {loaders.Count} file(s) ({Size(inBytes)})"
                + (castTo is { } t ? $", cast to {t.Name}" : "") + $" to {Path.GetFileName(output)}...");
            payloadSha = SafeTensorsMerger.Write(output, all, castTo, metadata, keeps.Count > 0 ? Keep : null);
            tensorCount = all.Count;
        }
        finally
        {
            foreach (SafeTensorsLoader loader in loaders)
                loader.Dispose();
            foreach (Tensor tensor in ownedTensors)
                tensor.Dispose();
        }
    }
    else if (isSafeTensors)
    {
        using (SafeTensorsLoader header = new())
        {
            header.Load(input);
            if (ResolveOutput(header.Descriptors.Values.Select(d => (d.DType, d.Shape.ElementCount))) is not string named)
            {
                return 1;
            }
            output = named;
        }
        metadata = BuildMetadata();
        Console.WriteLine($"Rewriting metadata; {Size(new FileInfo(input).Length)} of tensors copied unchanged...");
        payloadSha = SafeTensorsWriter.RewriteMetadata(input, output, metadata ?? []);
        using SafeTensorsLoader check = new();
        check.Load(output);
        tensorCount = check.Descriptors.Count;
    }
    else
    {
        bool recursive = flags.Contains("--recursive");
        Console.WriteLine($"Reading {Path.GetFileName(input)} ({Size(new FileInfo(input).Length)})...");
        using PytorchPickleLoader loader = new();
        loader.Load(input, recursive);
        if (loader.SkippedTensorCount > 0 && !flags.Contains("--allow-partial"))
        {
            Console.Error.WriteLine($"Stopped: this checkpoint holds several state dicts, and converting it as-is would drop {loader.SkippedTensorCount} of {loader.SkippedTensorCount + loader.Descriptors.Count} tensors.");
            foreach ((string name, int count) in loader.WrapperTensorCounts)
            {
                Console.Error.WriteLine($"  {name,-28} {count,6} tensors");
            }
            Console.Error.WriteLine("Add --recursive to keep all of them (keys become <dict>.<key>), or --allow-partial to keep only the first.");
            Console.Error.WriteLine("Nothing was written.");
            return 1;
        }
        Func<string, string?>? keyMap = stripPrefix is null && renames.Count == 0 ? null : key =>
        {
            string mapped = stripPrefix is not null && key.StartsWith(stripPrefix, StringComparison.Ordinal) ? key[stripPrefix.Length..] : key;
            foreach ((string from, string to) in renames)
            {
                mapped = mapped.Replace(from, to, StringComparison.Ordinal);
            }
            return mapped;
        };
        if (ResolveOutput(loader.Descriptors.Values.Select(d => (d.DType, d.Shape.ElementCount))) is not string named)
        {
            return 1;
        }
        output = named;
        metadata = BuildMetadata();
        Console.WriteLine($"Writing {loader.Descriptors.Count} tensors to {Path.GetFileName(output)}...");
        tensorCount = PickleCheckpointRepacker.Repack(loader, output, keyMap, metadata);
        if (tensorCount != loader.Descriptors.Count)
        {
            Console.Error.WriteLine($"Warning: {loader.Descriptors.Count - tensorCount} tensors collapsed onto the same key after renaming.");
        }
        payloadSha = PayloadSha256(output);
    }
}
catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or InvalidDataException or IOException
    or JsonException or HartsyInference.Core.Exceptions.HartsyInferenceException)
{
    Console.Error.WriteLine($"Could not convert {Path.GetFileName(input)}: {ex.Message}");
    Console.Error.WriteLine(isSafeTensors ? "" : "If this is not a PyTorch checkpoint, it cannot be converted with this tool.");
    return 1;
}

FileInfo sourceInfo = new(mergeMode ? mergeSources[0] : input);
FileInfo outputInfo = new(output);
string outputSha = Sha256(output);
var manifest = new
{
    schema = 1,
    conversion = mergeMode ? "safetensors-merge" : isSafeTensors ? "safetensors-metadata-rewrite" : "pytorch-pickle-to-safetensors",
    converter = mergeMode ? "HartsyInference.SafeTensorsMerger" : isSafeTensors ? "HartsyInference.SafeTensorsWriter.RewriteMetadata"
        : "HartsyInference.ModelAssets.PickleCheckpointRepacker",
    model = modelId,
    variant,
    component,
    source = new { file = mergeMode ? string.Join(",", mergeSources.Select(Path.GetFileName)) : sourceInfo.Name,
        bytes = mergeMode ? mergeSources.Sum(f => new FileInfo(f).Length) : sourceInfo.Length,
        sha256 = metadata?.GetValueOrDefault("hartsy.source_sha256") ?? (mergeMode ? "" : Sha256(input)) },
    sources = mergeMode ? mergeSources.Select(f => new { file = Path.GetFileName(f), bytes = new FileInfo(f).Length }).ToArray() : null,
    output = new { file = outputInfo.Name, bytes = outputInfo.Length, sha256 = outputSha, payload_sha256 = payloadSha, tensor_count = tensorCount },
    recursive_flatten = flags.Contains("--recursive"),
    stripped_prefix = stripPrefix,
    renames = renames.Select(r => $"{r.From}={r.To}").ToArray(),
    dtype_cast = castTo?.Name,
    embedded_metadata = ReadMetadata(output),
};
string manifestPath = Path.ChangeExtension(output, ".manifest.json");
File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
Console.WriteLine();
Console.WriteLine($"Done in {clock.Elapsed.TotalSeconds.ToString("0.0", CultureInfo.InvariantCulture)} s");
Console.WriteLine($"  output    {output}");
Console.WriteLine($"  tensors   {tensorCount}   size {Size(outputInfo.Length)}");
if (metadata is not null)
{
    Console.WriteLine($"  model     {metadata.GetValueOrDefault("modelspec.title")}  [{metadata.GetValueOrDefault("modelspec.architecture") ?? $"component: {component}"}]");
    Console.WriteLine($"  license   {metadata.GetValueOrDefault("modelspec.license") ?? "(on the main file)"}");
}
else
{
    Console.WriteLine("  model     no identity written (--no-metadata)");
}
Console.WriteLine($"  sha256    {outputSha}");
Console.WriteLine($"  manifest  {manifestPath}");
return 0;

// A recipe's sources are often all model.safetensors, so they are named by their path under the source root.
string SourceName(string file)
{
    string relative = recipe is null ? "" : Path.GetRelativePath(recipeRoot, file);
    return relative.Length == 0 || relative.StartsWith("..", StringComparison.Ordinal) || Path.IsPathRooted(relative) ? Path.GetFileName(file) : relative.Replace('\\', '/');
}

// PEFT keeps alpha out of the weights file; a pattern of per-module ranks or alphas is refused rather than flattened.
static (float? Alpha, bool RsLora, string? Problem) ReadPeftConfig(string path)
{
    if (!File.Exists(path))
        return (null, false, null);
    using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(path));
    JsonElement root = doc.RootElement;
    foreach (string pattern in new[] { "rank_pattern", "alpha_pattern" })
    {
        if (root.TryGetProperty(pattern, out JsonElement p) && p.ValueKind == JsonValueKind.Object && p.EnumerateObject().Any())
            return (null, false, $"{path} sets {pattern}; per-module ranks or alphas are not supported. Give --lora-alpha only if every module shares one.");
    }
    float? alpha = root.TryGetProperty("lora_alpha", out JsonElement a) && a.ValueKind == JsonValueKind.Number ? a.GetSingle() : null;
    bool rs = root.TryGetProperty("use_rslora", out JsonElement r) && r.ValueKind == JsonValueKind.True;
    return (alpha, rs, null);
}

static int Usage(string problem)
{
    Console.Error.WriteLine(problem);
    Console.Error.WriteLine("Run with --help for usage and examples.");
    return 2;
}

// The dtype holding the most bytes names the file's precision, in HartsyWeb's ModelPrecision vocabulary.
static string Precision(IEnumerable<(DType DType, long Elements)> tensors)
{
    string? top = tensors.GroupBy(t => t.DType.Name).OrderByDescending(g => g.Sum(t => t.Elements * t.DType.SizeInBytes))
        .Select(g => g.Key).FirstOrDefault();
    return top?.ToUpperInvariant() switch
    {
        "F32" => "fp32",
        "BF16" => "bf16",
        "F16" => "fp16",
        "F8_E4M3" or "F8_E5M2" => "fp8",
        "I8" => "int8",
        "F64" => "fp64",
        null => "fp32",
        string other => other.ToLowerInvariant(),
    };
}

static Regex Glob(string pattern) =>
    new("^" + Regex.Escape(pattern).Replace("\\*", ".*").Replace("\\?", ".") + "$", RegexOptions.CultureInvariant);

static Regex ShardPattern() => new(@"-\d{5}-of-\d{5}$");

static string Alnum(string text) => new(text.Where(char.IsLetterOrDigit).Select(char.ToLowerInvariant).ToArray());

static string Size(long bytes) => bytes >= 1L << 30
    ? (bytes / (double)(1L << 30)).ToString("0.00", CultureInfo.InvariantCulture) + " GB"
    : (bytes / (double)(1L << 20)).ToString("0.0", CultureInfo.InvariantCulture) + " MB";

static string Sha256(string path)
{
    using FileStream stream = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
}

static Dictionary<string, string>? ReadMetadata(string path)
{
    using SafeTensorsLoader loader = new();
    loader.Load(path);
    return loader.Metadata is null ? null : new Dictionary<string, string>(loader.Metadata, StringComparer.Ordinal);
}

static string PayloadSha256(string path)
{
    using FileStream stream = File.OpenRead(path);
    Span<byte> length = stackalloc byte[8];
    stream.ReadExactly(length);
    stream.Seek(8 + BitConverter.ToInt64(length), SeekOrigin.Begin);
    return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
}

static int Distance(string a, string b)
{
    int[,] d = new int[a.Length + 1, b.Length + 1];
    for (int i = 0; i <= a.Length; i++)
    {
        d[i, 0] = i;
    }
    for (int j = 0; j <= b.Length; j++)
    {
        d[0, j] = j;
    }
    for (int i = 1; i <= a.Length; i++)
    {
        for (int j = 1; j <= b.Length; j++)
        {
            d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
        }
    }
    return d[a.Length, b.Length];
}
