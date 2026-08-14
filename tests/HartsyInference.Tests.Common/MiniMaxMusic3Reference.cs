using System.Text.Json;

namespace HartsyInference.Tests.Common;

/// <summary>The MiniMax Music 3 reference dump consumed by the Audio and Diffusion parity tests: flat F32
/// <c>.bin</c> payloads plus a <c>meta.json</c> holding each one's shape.</summary>
/// <remarks>Regenerate with <c>tests/python-reference/dump_minimax_music3_reference.py</c> under a venv carrying
/// diffusers @ dafe3733 (PR #14456). The <c>.bin</c> files are gitignored by the repo-wide <c>*.bin</c> rule, so
/// every caller must treat a null load as "skip". Lives here because both <c>HartsyInference.Audio.Tests</c>
/// (prompt, depth decoder, vocoder) and <c>HartsyInference.Diffusion.Tests</c> (condition encoder, DiT) read it.</remarks>
public sealed class MiniMaxMusic3Reference
{
    private readonly string _directory;
    private readonly JsonElement _meta;

    private MiniMaxMusic3Reference(string directory, JsonElement meta)
    {
        _directory = directory;
        _meta = meta;
    }

    /// <summary>Locates the dump directory, honouring the <c>MINIMAX_MUSIC3_REFERENCE_DIR</c> override, or null
    /// when it has not been generated on this machine.</summary>
    public static string? FixtureDirectory()
    {
        string? env = Environment.GetEnvironmentVariable("MINIMAX_MUSIC3_REFERENCE_DIR");
        if (!string.IsNullOrWhiteSpace(env))
        {
            return Directory.Exists(env) ? env : null;
        }
        string? dir = AppContext.BaseDirectory;
        for (int up = 0; up < 8 && dir is not null; up++, dir = Path.GetDirectoryName(dir))
        {
            string candidate = Path.Combine(dir, "tests", "python-reference", "minimax_music3_reference");
            if (File.Exists(Path.Combine(candidate, "meta.json")))
            {
                return candidate;
            }
        }
        return null;
    }

    /// <summary>Loads the dump's metadata, or returns null when it is absent.</summary>
    public static MiniMaxMusic3Reference? TryLoad()
    {
        string? directory = FixtureDirectory();
        if (directory is null)
        {
            return null;   // tier-lint: guarded
        }
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "meta.json")));
        return new MiniMaxMusic3Reference(directory, document.RootElement.Clone());
    }

    /// <summary>True when <paramref name="name"/> was written by the dump run present on this machine — the three
    /// stages (<c>components</c>/<c>flow</c>/<c>ar</c>) are generated separately and any subset may be missing.</summary>
    public bool Has(string name) => _meta.TryGetProperty(name, out _) && File.Exists(Path.Combine(_directory, $"{name}.bin"));

    /// <summary>The dumped shape of <paramref name="name"/>.</summary>
    public int[] Shape(string name) => [.. _meta.GetProperty(name).EnumerateArray().Select(dimension => dimension.GetInt32())];

    /// <summary>The flat F32 payload of <paramref name="name"/>.</summary>
    public float[] Read(string name)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(_directory, $"{name}.bin"));
        float[] values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * 4);
        return values;
    }

    /// <summary>The flat I32 payload of <paramref name="name"/>.</summary>
    public int[] ReadInt32(string name)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(_directory, $"{name}.bin"));
        int[] values = new int[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * 4);
        return values;
    }

    /// <summary>A scalar or array entry from <c>meta.json</c> that is not a tensor shape.</summary>
    public JsonElement Meta(string name) => _meta.GetProperty(name);

    /// <summary>Mean absolute error and worst element-wise difference against the dump.</summary>
    public static (double MeanAbsError, double MaxAbsError, double Correlation) Compare(
        ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        int count = Math.Min(actual.Length, expected.Length);
        double sumAbs = 0d, maxAbs = 0d, dot = 0d, normA = 0d, normB = 0d;
        for (int i = 0; i < count; i++)
        {
            double difference = Math.Abs(actual[i] - (double)expected[i]);
            sumAbs += difference;
            maxAbs = Math.Max(maxAbs, difference);
            dot += actual[i] * (double)expected[i];
            normA += actual[i] * (double)actual[i];
            normB += expected[i] * (double)expected[i];
        }
        return (sumAbs / Math.Max(1, count), maxAbs, dot / (Math.Sqrt(normA * normB) + 1e-12));
    }
}
