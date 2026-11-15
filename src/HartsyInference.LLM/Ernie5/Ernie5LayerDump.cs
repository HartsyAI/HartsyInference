namespace HartsyInference.LLM.Ernie5;

/// <summary>Test-only per-layer dump for the offline reference comparison (tools/ernie5-reference). When the environment variable
/// <c>ERNIE_DUMP_LAYERS</c> names a directory, <see cref="Ernie5Model.ForwardStream"/> writes the residual stream after each decoder layer as
/// <c>layer{LL}_pos{startPos}_t{tokens}_hc{hc}_h{hidden}.f32</c>: little-endian float32, row-major <c>[tokens, hc, hidden]</c>. Unset, it writes
/// nothing and costs one environment read per forward call.</summary>
internal static class Ernie5LayerDump
{
    /// <summary>The environment variable that names the dump directory.</summary>
    internal const string EnvVar = "ERNIE_DUMP_LAYERS";

    /// <summary>The dump directory from <see cref="EnvVar"/>, or null when the dump is off.</summary>
    internal static string? DirectoryFromEnvironment()
    {
        string? dir = Environment.GetEnvironmentVariable(EnvVar);
        return string.IsNullOrWhiteSpace(dir) ? null : dir;
    }

    /// <summary>Writes one layer's stream (<c>tokens * hc * hidden</c> values) into <paramref name="directory"/>.</summary>
    internal static void Write(string directory, int layer, int startPos, int tokens, int hc, int hidden, ReadOnlySpan<float> stream)
    {
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, $"layer{layer:D2}_pos{startPos}_t{tokens}_hc{hc}_h{hidden}.f32");
        using FileStream file = File.Create(path);
        file.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(stream));
    }
}
