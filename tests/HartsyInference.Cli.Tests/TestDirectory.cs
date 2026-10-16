namespace HartsyInference.Cli.Tests;

/// <summary>A fresh temporary folder for a test's input files, deleted with everything in it on dispose.</summary>
internal sealed class TestDirectory : IDisposable
{
    private readonly string _path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"hartsy-cli-tests-{Guid.NewGuid():N}");

    internal TestDirectory() => Directory.CreateDirectory(_path);

    internal string PathOf(string name) => System.IO.Path.GetFullPath(System.IO.Path.Combine(_path, name));

    internal void WriteBytes(string name, byte[] value) => File.WriteAllBytes(PathOf(name), value);

    internal string WriteText(string name, string value)
    {
        string path = PathOf(name);
        File.WriteAllText(path, value);
        return path;
    }

    public void Dispose() => Directory.Delete(_path, recursive: true);
}
