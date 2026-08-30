using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

if (args.Length < 2)
{
    Console.Error.WriteLine("Usage: CheckpointRepacker <input.pt> <output.safetensors> [--recursive] [--strip-prefix PREFIX] [--meta KEY=VALUE ...] [--meta-json FILE]");
    return 2;
}

string input = Path.GetFullPath(args[0]);
string output = Path.GetFullPath(args[1]);
bool recursive = args.Contains("--recursive", StringComparer.Ordinal);
string? stripPrefix = null;
Dictionary<string, string> metadata = new(StringComparer.Ordinal);
for (int i = 2; i < args.Length; i++)
{
    if (args[i] == "--strip-prefix" && i + 1 < args.Length)
    {
        stripPrefix = args[++i];
    }
    else if (args[i] == "--meta" && i + 1 < args.Length)
    {
        string pair = args[++i];
        int split = pair.IndexOf('=', StringComparison.Ordinal);
        if (split <= 0)
        {
            Console.Error.WriteLine($"Invalid --meta '{pair}': expected KEY=VALUE.");
            return 2;
        }
        metadata[pair[..split]] = pair[(split + 1)..];
    }
    else if (args[i] == "--meta-json" && i + 1 < args.Length)
    {
        string metaPath = Path.GetFullPath(args[++i]);
        Dictionary<string, string>? loaded = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(metaPath));
        if (loaded is null)
        {
            Console.Error.WriteLine($"Could not read metadata JSON: {metaPath}");
            return 1;
        }
        foreach (KeyValuePair<string, string> entry in loaded)
        {
            metadata[entry.Key] = entry.Value;
        }
    }
}

if (!File.Exists(input))
{
    Console.Error.WriteLine($"Input not found: {input}");
    return 1;
}

string? prefix = stripPrefix;
string? MapKey(string key) => prefix is not null && key.StartsWith(prefix, StringComparison.Ordinal)
    ? key[prefix.Length..]
    : key;

// An empty hash slot tells the repacker to fill it with the tensor-payload digest it can compute for free mid-write.
if (metadata.Count > 0 && !metadata.ContainsKey("modelspec.hash_sha256"))
{
    metadata["modelspec.hash_sha256"] = "";
}
int count = PickleCheckpointRepacker.Repack(input, output, prefix is null ? null : MapKey, recursive, metadata.Count > 0 ? metadata : null);
FileInfo sourceInfo = new(input);
FileInfo outputInfo = new(output);
var manifest = new
{
    schema = 1,
    conversion = "pytorch-pickle-to-safetensors",
    converter = "HartsyInference.ModelAssets.PickleCheckpointRepacker",
    source = new
    {
        file = sourceInfo.Name,
        bytes = sourceInfo.Length,
        sha256 = Sha256(input),
    },
    output = new
    {
        file = outputInfo.Name,
        bytes = outputInfo.Length,
        sha256 = Sha256(output),
        tensor_count = count,
    },
    recursive_flatten = recursive,
    stripped_prefix = prefix,
    dtype_cast = (string?)null,
    embedded_metadata = metadata.Count > 0 ? ReadEmbeddedMetadata(output) : null,
};
string manifestPath = Path.ChangeExtension(output, ".manifest.json");
File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
Console.WriteLine($"Wrote {outputInfo.Name}: {count} tensors, {outputInfo.Length} bytes");
Console.WriteLine($"SHA-256 {manifest.output.sha256}");
return 0;

static string Sha256(string path)
{
    using FileStream stream = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
}

// Read back rather than echoing the request, so the manifest records what the file actually carries.
static Dictionary<string, string>? ReadEmbeddedMetadata(string path)
{
    using SafeTensorsLoader loader = new();
    loader.Load(path);
    return loader.Metadata is null ? null : new Dictionary<string, string>(loader.Metadata, StringComparer.Ordinal);
}
