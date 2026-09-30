namespace HartsyInference.PhoneGateway.Tests.Support;

/// <summary>A secret file under the temp directory with the given content and, off Windows, the given Unix mode (0600
/// unless told otherwise); deleted on dispose.</summary>
internal sealed class TempSecret : IDisposable
{
    public TempSecret(string content, UnixFileMode mode = UnixFileMode.UserRead | UnixFileMode.UserWrite)
    {
        Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "hartsy-secret-" + Guid.NewGuid().ToString("N"));
        File.WriteAllText(Path, content);
        if (!OperatingSystem.IsWindows())
        {
            File.SetUnixFileMode(Path, mode);
        }
    }

    public string Path { get; }

    public void Dispose() => File.Delete(Path);
}
