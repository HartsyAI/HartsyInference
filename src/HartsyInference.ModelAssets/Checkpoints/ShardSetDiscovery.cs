using System.Text.RegularExpressions;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>Finds the whole of a sharded safetensors checkpoint from any one of its shards, by the Hugging Face naming
/// convention <c>&lt;stem&gt;-NNNNN-of-MMMMM.safetensors</c>. Used when a catalog or a user hands the engine a single
/// shard path and the rest sit beside it, so no index file has to be present.</summary>
public static class ShardSetDiscovery
{
    /// <summary>Largest shard count accepted. A real checkpoint is tens of shards; this bounds the probe on a crafted name.</summary>
    private const int MaxShards = 1000;

    private static readonly Regex ShardName = new(@"^(?<stem>.+)-(?<index>\d{5})-of-(?<total>\d{5})\.safetensors$",
        RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);

    /// <summary>The shard files that make up <paramref name="path"/>, in index order. A path that is not a shard
    /// name is returned on its own, so callers can pass a single-file checkpoint through unchanged.</summary>
    /// <exception cref="FileNotFoundException">The name is a shard but one of its siblings is missing from disk.</exception>
    public static IReadOnlyList<string> Resolve(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        Match match = ShardName.Match(Path.GetFileName(path));
        if (!match.Success)
        {
            return [path];
        }

        string directory = Path.GetDirectoryName(path) ?? string.Empty;
        string stem = match.Groups["stem"].Value;
        int total = int.Parse(match.Groups["total"].Value, System.Globalization.CultureInfo.InvariantCulture);
        if (total < 1 || total > MaxShards)
        {
            throw new InvalidDataException($"'{Path.GetFileName(path)}' declares {total} shards; expected 1 to {MaxShards}.");
        }
        string[] shards = new string[total];
        for (int i = 0; i < total; i++)
        {
            string name = $"{stem}-{i + 1:D5}-of-{total:D5}.safetensors";
            shards[i] = Path.Combine(directory, name);
            if (!File.Exists(shards[i]))
            {
                throw new FileNotFoundException(
                    $"Shard {i + 1} of {total} of '{stem}' is missing; expected '{name}' beside '{Path.GetFileName(path)}'.",
                    shards[i]);
            }
        }
        return shards;
    }
}
