using System.Security.Cryptography;
using System.Text;

namespace HartsyInference.BenchmarkRunner.Serialization;
/// <summary>Artifact identities and portable bounded path handling.</summary>
public static class Hashes
{
    public static string FileHash(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    public static string Text(string text) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text)));
    public static bool IsHash(string text) => text.Length == 64 && text.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f');
    public static string SafePath(string root, string relative)
    {
        if (string.IsNullOrWhiteSpace(relative) || relative.Contains('\\') || relative.Contains(':') || relative.StartsWith('/') || relative
            .Split('/').Any(p => p is ".." or "." or ""))
            throw new InvalidDataException("Unsafe artifact path.");
        string full = Path.GetFullPath(Path.Combine(root, relative));
        string current = full;
        while (current.Length >= Path.GetFullPath(root).Length)
        {
            if ((File.Exists(current) || Directory.Exists(current)) && (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                throw new InvalidDataException("Linked artifact path.");
            current = Path.GetDirectoryName(current) ?? "";
        }

        return full;
    }
}
