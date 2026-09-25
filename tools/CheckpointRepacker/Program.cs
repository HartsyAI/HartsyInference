using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

const string HelpText = """
CheckpointRepacker - turn a model checkpoint into a self-describing .safetensors file.

  PyTorch pickles (.pt .pth .bin .th .ckpt) are converted. An existing .safetensors keeps its tensors
  byte-for-byte and only has its identity metadata rewritten. Either way the output carries the
  architecture, title, author and license SwarmUI and Hartsy read, plus a .manifest.json with hashes.

USAGE
  CheckpointRepacker <input> <output.safetensors> --model <id> [options]
  CheckpointRepacker --list-models

IDENTITY
  --model <id>          Engine model id, e.g. kokoro, whisper, dia. See --list-models. Required.
  --variant <name>      Variant, e.g. large-v3 or 1.7B-Base. Goes in the title; picks the class where
                        variants register differently (Qwen3-TTS).
  --component <name>    main (default) for the weights a user selects. For a companion file name its
                        part instead (codec, vocoder, voice, tokenizer...): companions get no
                        architecture, so they never show up as a model of their own.
  --source-repo <repo>  Upstream repo to record as provenance. Defaults to the model's known upstream.
  --meta KEY=VALUE      Add or override one metadata entry. Repeatable.
  --meta-json <file>    Add or override entries from a JSON object of strings.
  --no-metadata         Write no identity. SwarmUI then cannot classify the file; not for publishing.

KEY LAYOUT (pickles only)
  --recursive           Keep every nested state dict, prefixing keys with the dict name (bert.x ...).
  --allow-partial       Keep only the first nested state dict even though others hold tensors.
  --strip-prefix <p>    Remove a leading <p> from every key.
  --rename FROM=TO      Replace FROM with TO anywhere in a key. Repeatable. e.g. --rename .module.=.

OUTPUT
  --force               Replace the output if it already exists.
  -h, --help            Show this help.

EXAMPLES
  CheckpointRepacker kokoro-v1_0.pth kokoro-82m.safetensors --model kokoro --recursive --rename .module.=.
  CheckpointRepacker model.safetensors out/model.safetensors --model whisper --variant large-v3
  CheckpointRepacker dac.pth dac.safetensors --model dia --component codec
""";

// The tool prints its own progress; the engine's info lines would repeat it.
Logs.MinLevel = LogLevel.Warning;
string[] pickleExtensions = [".pt", ".pth", ".bin", ".th", ".ckpt"];
string[] valueOptions = ["--model", "--variant", "--component", "--source-repo", "--meta", "--meta-json", "--strip-prefix", "--rename"];
string[] flagOptions = ["--recursive", "--allow-partial", "--no-metadata", "--force", "--help", "-h", "--list-models"];

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
if (positionals.Count != 2)
{
    return Usage(positionals.Count < 2 ? "Give an input checkpoint and an output .safetensors path." : $"Too many paths: {string.Join(" ", positionals)}");
}

string input = Path.GetFullPath(positionals[0]);
string output = Path.GetFullPath(positionals[1]);
if (Directory.Exists(input))
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
if (!File.Exists(input))
{
    Console.Error.WriteLine($"Input not found: {input}");
    return 1;
}
string inputExtension = Path.GetExtension(input).ToLowerInvariant();
bool isSafeTensors = inputExtension == ".safetensors";
if (!isSafeTensors && !pickleExtensions.Contains(inputExtension))
{
    Console.Error.WriteLine($"'{Path.GetFileName(input)}' is not a supported checkpoint. Expected .safetensors or one of {string.Join(" ", pickleExtensions)}.");
    Console.Error.WriteLine(inputExtension is ".onnx" or ".gguf"
        ? "ONNX and GGUF stay in their own format; describe them with a .swarm.json sidecar (tools/repack/stamp_modelspec.py --sidecar-only)."
        : "");
    return 1;
}
if (!output.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
{
    return Usage($"The output must end in .safetensors: {positionals[1]}");
}
if (File.Exists(output) && !flags.Contains("--force"))
{
    Console.Error.WriteLine($"Output already exists: {output}");
    Console.Error.WriteLine("Add --force to replace it.");
    return 1;
}
if (isSafeTensors && (flags.Contains("--recursive") || flags.Contains("--allow-partial") || values.ContainsKey("--strip-prefix") || values.ContainsKey("--rename")))
{
    return Usage("Key layout options only apply to pickle checkpoints; a .safetensors input keeps its tensors as they are.");
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
    if (identity.VariantClassIds.Count > 0 && component == ArtifactProvenance.MainComponent
        && (variant is null || !identity.VariantClassIds.Keys.Contains(variant, StringComparer.OrdinalIgnoreCase)))
    {
        Console.Error.WriteLine($"'{modelId}' variants register as different classes; pass --variant with one of: {string.Join(", ", identity.VariantClassIds.Keys)}");
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

// Built after the load succeeds: hashing a multi-gigabyte source for provenance is wasted on a file that won't convert.
Dictionary<string, string>? BuildMetadata()
{
    Dictionary<string, string>? built = null;
    if (resolvedIdentity is not null)
    {
        Console.WriteLine($"Hashing source {Path.GetFileName(input)} for provenance...");
        string converter = isSafeTensors ? "upstream-safetensors" : "HartsyInference.PickleCheckpointRepacker";
        ArtifactProvenance provenance = ArtifactProvenance.FromSourceFile(converter, component, input,
            values.GetValueOrDefault("--source-repo")?.Last());
        built = ArtifactMetadata.ForRepack(resolvedIdentity, provenance);
        if (variant is not null)
        {
            built["hartsy.variant"] = variant;
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

Stopwatch clock = Stopwatch.StartNew();
int tensorCount;
string payloadSha;
try
{
    if (isSafeTensors)
    {
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

FileInfo sourceInfo = new(input);
FileInfo outputInfo = new(output);
string outputSha = Sha256(output);
var manifest = new
{
    schema = 1,
    conversion = isSafeTensors ? "safetensors-metadata-rewrite" : "pytorch-pickle-to-safetensors",
    converter = isSafeTensors ? "HartsyInference.SafeTensorsWriter.RewriteMetadata" : "HartsyInference.ModelAssets.PickleCheckpointRepacker",
    model = modelId,
    variant,
    component,
    source = new { file = sourceInfo.Name, bytes = sourceInfo.Length, sha256 = metadata?.GetValueOrDefault("hartsy.source_sha256") ?? Sha256(input) },
    output = new { file = outputInfo.Name, bytes = outputInfo.Length, sha256 = outputSha, payload_sha256 = payloadSha, tensor_count = tensorCount },
    recursive_flatten = flags.Contains("--recursive"),
    stripped_prefix = stripPrefix,
    renames = renames.Select(r => $"{r.From}={r.To}").ToArray(),
    dtype_cast = (string?)null,
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

static int Usage(string problem)
{
    Console.Error.WriteLine(problem);
    Console.Error.WriteLine("Run with --help for usage and examples.");
    return 2;
}

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
