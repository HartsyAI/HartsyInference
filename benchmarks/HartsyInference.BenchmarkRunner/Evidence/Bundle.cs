using System.IO.Compression;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner.Evidence;
/// <summary>Data-only evidence bundles with bounded expansion, portable paths, and a complete hash inventory.</summary>
public static class Bundle
{
    public const long MaximumBytes = 256L * 1024 * 1024;
    public static void Export(string root, string destination)
    {
        ValidationReport report = Validator.Validate(root);
        if (!report.Valid)
            throw new InvalidDataException(string.Join("; ", report.Errors));
        CampaignRecord campaign = BenchJson.Read(Path.Combine(root, "campaign.json"), BenchJson.Default.CampaignRecord);
        SortedDictionary<string, string> inventory = new(StringComparer.Ordinal)
        {
            ["campaign.json"] = Hashes.FileHash(Path.Combine(root, "campaign.json"))
        };
        foreach (Measurement measurement in campaign.Sessions.SelectMany(s => s.Measurements))
            inventory.Add(measurement.Output, measurement.OutputSha256);
        long total = inventory.Keys.Sum(p => new FileInfo(Hashes.SafePath(root, p)).Length);
        if (total > MaximumBytes)
            throw new InvalidDataException("Mandatory evidence exceeds 256 MiB; do not omit trials.");
        BenchJson.Write(Path.Combine(root, "checksums.json"), inventory, BenchJson.Default.SortedDictionaryStringString);
        string temporary = destination + ".tmp";
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(destination))!);
        try
        {
            using (FileStream file = new(temporary, FileMode.CreateNew))
            using (ZipArchive zip = new(file, ZipArchiveMode.Create))
                foreach (string relative in inventory.Keys.Append("checksums.json").Order(StringComparer.Ordinal))
                {
                    ZipArchiveEntry entry = zip.CreateEntry(relative, CompressionLevel.Optimal);
                    entry.LastWriteTime = new DateTimeOffset(2000, 1, 1, 0, 0, 0, TimeSpan.Zero);
                    using Stream output = entry.Open();
                    using FileStream input = File.OpenRead(Hashes.SafePath(root, relative));
                    input.CopyTo(output);
                }

            if (new FileInfo(temporary).Length > MaximumBytes)
                throw new InvalidDataException("Compressed evidence exceeds 256 MiB.");
            File.Move(temporary, destination, false);
        }
        finally
        {
            if (File.Exists(temporary))
                File.Delete(temporary);
        }
    }

    public static void Extract(string bundle, string destination)
    {
        if (new FileInfo(bundle).Length > MaximumBytes)
            throw new InvalidDataException("Oversized bundle.");
        if (Directory.Exists(destination) && Directory.EnumerateFileSystemEntries(destination).Any())
            throw new IOException("Extraction destination must be empty.");
        Directory.CreateDirectory(destination);
        using ZipArchive zip = ZipFile.OpenRead(bundle);
        if (zip.Entries.Count is < 2 or > 2000)
            throw new InvalidDataException("Invalid bundle entry count.");
        HashSet<string> names = new(StringComparer.OrdinalIgnoreCase);
        long total = 0;
        foreach (ZipArchiveEntry entry in zip.Entries)
        {
            string relative = entry.FullName;
            string file = Hashes.SafePath(destination, relative);
            int unixType = (entry.ExternalAttributes >> 16) & 0xF000;
            if (!names.Add(relative) || unixType is not (0 or 0x8000) || (entry.ExternalAttributes & (int)FileAttributes.ReparsePoint) != 0
                || entry.Length < 0 || entry.Length > 8 * 1024 * 1024 || (total += entry.Length) > MaximumBytes || !(
                relative is "campaign.json" or "checksums.json" || relative.StartsWith("sessions/", StringComparison.Ordinal) && (relative
                .EndsWith(".txt", StringComparison.Ordinal) || relative.EndsWith(".png", StringComparison.Ordinal))))
                throw new InvalidDataException("Unsafe or excessive bundle entry.");
            Directory.CreateDirectory(Path.GetDirectoryName(file)!);
            using Stream input = entry.Open();
            using FileStream output = new(file, FileMode.CreateNew);
            byte[] buffer = new byte[65536];
            long copied = 0;
            int read;
            while ((read = input.Read(buffer)) > 0)
            {
                copied += read;
                if (copied > entry.Length)
                    throw new InvalidDataException("Archive expands beyond declared size.");
                output.Write(buffer, 0, read);
            }

            if (copied != entry.Length)
                throw new InvalidDataException("Truncated archive entry.");
        }

        SortedDictionary<string, string> inventory = BenchJson.Read(Hashes.SafePath(destination, "checksums.json"), BenchJson.Default
            .SortedDictionaryStringString);
        if (!names.SetEquals(inventory.Keys.Append("checksums.json")))
            throw new InvalidDataException("Incomplete artifact inventory.");
        foreach (KeyValuePair<string, string> item in inventory)
            if (!Hashes.IsHash(item.Value) || Hashes.FileHash(Hashes.SafePath(destination, item.Key)) != item.Value)
                throw new InvalidDataException("Artifact checksum mismatch.");
        CampaignRecord campaign = BenchJson.Read(Hashes.SafePath(destination, "campaign.json"), BenchJson.Default.CampaignRecord);
        if (!inventory.Keys.ToHashSet(StringComparer.Ordinal).SetEquals(campaign.Sessions.SelectMany(s => s.Measurements).Select(m => m
            .Output).Append("campaign.json")))
            throw new InvalidDataException("Unreferenced artifact in bundle.");
    }
}
