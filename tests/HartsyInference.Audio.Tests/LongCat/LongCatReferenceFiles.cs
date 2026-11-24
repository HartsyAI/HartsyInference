using System.Runtime.InteropServices;
using Xunit;

namespace HartsyInference.Audio.Tests.LongCat;

/// <summary>The F32 dumps of tests/python-reference/longcat_audio, read from the directory in <see cref="DirectoryVariable"/>.</summary>
internal static class LongCatReferenceFiles
{
    public const string DirectoryVariable = "HARTSY_LONGCAT_AUDIO_DIR";

    public static string? Missing(params string[] names)
    {
        string? dir = Environment.GetEnvironmentVariable(DirectoryVariable);
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
            return $"SKIPPED: {DirectoryVariable} is unset or not a directory; the reference comparison did not run.";
        foreach (string name in names)
            if (!File.Exists(Path.Combine(dir, name))) return $"SKIPPED: {name} is not in {dir}; the reference comparison did not run.";
        return null;
    }

    public static float[] Read(string file)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(Environment.GetEnvironmentVariable(DirectoryVariable)!, file));
        float[] values = new float[bytes.Length / 4];
        MemoryMarshal.Cast<byte, float>(bytes).CopyTo(values);
        return values;
    }

    public static string FullPath(string file) => Path.Combine(Environment.GetEnvironmentVariable(DirectoryVariable)!, file);

    public static int MetaInt(string key)
    {
        string text = File.ReadAllText(FullPath("meta.json"));
        using System.Text.Json.JsonDocument doc = System.Text.Json.JsonDocument.Parse(text);
        return doc.RootElement.GetProperty(key).GetInt32();
    }

    public static bool MetaBool(string key)
    {
        using System.Text.Json.JsonDocument doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(FullPath("meta.json")));
        return doc.RootElement.GetProperty(key).GetBoolean();
    }

    public static double RelRms(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double sumSq = 0, errSq = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            sumSq += (double)expected[i] * expected[i];
            errSq += (double)(actual[i] - expected[i]) * (actual[i] - expected[i]);
        }
        return Math.Sqrt(errSq / Math.Max(sumSq, 1e-30));
    }
}

/// <summary>A fact skipped, never passed, when a reference dump or a weight file it needs is absent.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class LongCatReferenceFactAttribute : FactAttribute
{
    public LongCatReferenceFactAttribute(params string[] files)
    {
        Skip = LongCatReferenceFiles.Missing(files);
    }
}

/// <summary><see cref="LongCatReferenceFactAttribute"/> that also needs the file named by an environment variable (model weights).</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class LongCatWeightsFactAttribute : FactAttribute
{
    public LongCatWeightsFactAttribute(string weightsVariable, params string[] files)
    {
        string? weights = Environment.GetEnvironmentVariable(weightsVariable);
        Skip = string.IsNullOrWhiteSpace(weights) || !File.Exists(weights)
            ? $"SKIPPED: {weightsVariable} is unset or not an existing file; the real-weight comparison did not run."
            : LongCatReferenceFiles.Missing(files);
    }
}
