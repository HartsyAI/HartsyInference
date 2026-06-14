using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Tests.Common;

/// <summary>Image helpers shared across tests. Wraps ImagePostProcessor.SaveBmp with directory-aware path resolution into TestPaths.OutputDir.</summary>
public static class TestImage
{
    /// <summary>Saves an RGB byte array as BMP into TestPaths.OutputDir, ensuring the directory exists. Returns the full path written.</summary>
    public static string SaveBmp(string fileName, byte[] rgb, int width, int height)
    {
        string dir = TestPaths.OutputDir;
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, EnsureExtension(fileName, ".bmp"));
        ImagePostProcessor.SaveBmp(path, rgb, width, height);
        return path;
    }

    /// <summary>Saves an RGB byte array as BMP into a date-stamped subfolder of TestPaths.OutputDir (yyyy-MM-dd). Mirrors OutputSettings layout.</summary>
    public static string SaveBmpDated(string fileName, byte[] rgb, int width, int height)
    {
        string dir = Path.Combine(TestPaths.OutputDir, DateTime.Now.ToString("yyyy-MM-dd"));
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, EnsureExtension(fileName, ".bmp"));
        ImagePostProcessor.SaveBmp(path, rgb, width, height);
        return path;
    }

    /// <summary>Saves an RGB byte array as BMP at an absolute path, ensuring its directory exists.</summary>
    public static string SaveBmpAt(string absolutePath, byte[] rgb, int width, int height)
    {
        string? dir = Path.GetDirectoryName(absolutePath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);
        ImagePostProcessor.SaveBmp(absolutePath, rgb, width, height);
        return absolutePath;
    }

    private static string EnsureExtension(string fileName, string ext) =>
        fileName.EndsWith(ext, StringComparison.OrdinalIgnoreCase) ? fileName : fileName + ext;
}
