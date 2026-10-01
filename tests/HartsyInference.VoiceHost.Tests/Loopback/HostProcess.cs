using System.Collections.Concurrent;
using System.Diagnostics;
using HartsyInference.Tests.Common;

namespace HartsyInference.VoiceHost.Tests.Loopback;

/// <summary>The voice host as its own process (<c>HartsyInference.VoiceHost.TestHost</c>: the shipping composition with a
/// scripted language model), so a test can SIGKILL it. It inherits this process's environment, CUDA_VISIBLE_DEVICES
/// included, and its output is kept for the test log.</summary>
internal sealed class HostProcess : IDisposable
{
    private const string AssemblyName = "HartsyInference.VoiceHost.TestHost";

    private readonly Process _process;
    private readonly ConcurrentQueue<string> _output = new();

    private HostProcess(Process process) => _process = process;

    public int Id => _process.Id;

    public bool HasExited => _process.HasExited;

    public int ExitCode => _process.ExitCode;

    /// <summary>Everything the host wrote so far.</summary>
    public string[] Output => [.. _output];

    public static HostProcess Start(string configPath, string reply)
    {
        string dll = Locate();
        ProcessStartInfo info = new(Environment.ProcessPath is { } dotnet && Path.GetFileNameWithoutExtension(dotnet) == "dotnet" ? dotnet : "dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        info.ArgumentList.Add("exec");
        info.ArgumentList.Add(dll);
        info.ArgumentList.Add("--config");
        info.ArgumentList.Add(configPath);
        info.ArgumentList.Add("--reply");
        info.ArgumentList.Add(reply);
        Process process = new() { StartInfo = info };
        HostProcess host = new(process);
        process.OutputDataReceived += (_, line) => host.Keep(line.Data);
        process.ErrorDataReceived += (_, line) => host.Keep(line.Data);
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        return host;
    }

    /// <summary>SIGKILL: the process dies with no chance to close its socket or end its calls.</summary>
    public void Kill() => _process.Kill(entireProcessTree: false);

    /// <summary>SIGTERM, then waits for the graceful stop; true when it exited within <paramref name="timeoutMs"/>.</summary>
    public bool Terminate(int timeoutMs)
    {
        using Process kill = Process.Start(new ProcessStartInfo("kill") { ArgumentList = { "-TERM", Id.ToString(System.Globalization.CultureInfo.InvariantCulture) }, UseShellExecute = false })!;
        kill.WaitForExit(5_000);
        return _process.WaitForExit(timeoutMs);
    }

    public void Dispose()
    {
        if (!_process.HasExited)
        {
            _process.Kill(entireProcessTree: false);
            _process.WaitForExit(10_000);
        }
        _process.Dispose();
    }

    private void Keep(string? line)
    {
        if (line is not null)
        {
            _output.Enqueue(line);
        }
    }

    /// <summary>The test host's own build output when it is there (it carries the host's runtime configuration), else the
    /// copy beside the tests.</summary>
    private static string Locate()
    {
        // .../bin/<configuration>/<framework>/
        DirectoryInfo testBin = new(AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar));
        string configuration = testBin.Parent?.Name ?? "Release";
        string own = Path.Combine(RepoRoot.Path, "tests", AssemblyName, "bin", configuration, testBin.Name, AssemblyName + ".dll");
        string beside = Path.Combine(AppContext.BaseDirectory, AssemblyName + ".dll");
        foreach (string candidate in new[] { own, beside })
        {
            if (File.Exists(candidate) && File.Exists(Path.ChangeExtension(candidate, ".runtimeconfig.json")))
            {
                return candidate;
            }
        }
        throw new FileNotFoundException($"Build {AssemblyName} first: neither {own} nor {beside} has its runtime configuration.");
    }
}
