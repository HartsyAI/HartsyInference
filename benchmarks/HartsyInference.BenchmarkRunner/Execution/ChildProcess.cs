using System.Diagnostics;
using System.Reflection;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Argument-safe child execution with process-tree termination and bounded log capture.</summary>
public static class ChildProcess
{
    public static async Task<int> RunAsync(string[] arguments, string log, TimeSpan timeout, CancellationToken cancel)
    {
        string executable = Assembly.GetEntryAssembly() == typeof(ChildProcess).Assembly ? Environment.ProcessPath! : Path.Combine(
            AppContext.BaseDirectory, OperatingSystem.IsWindows() ? "hartsy-bench.exe" : "hartsy-bench");
        ProcessStartInfo start = new(executable)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        if (Path.GetFileNameWithoutExtension(executable).Equals("dotnet", StringComparison.OrdinalIgnoreCase))
            start.ArgumentList.Add(Assembly.GetExecutingAssembly().Location);
        foreach (string argument in arguments)
            start.ArgumentList.Add(argument);
        // Worker output is local only. Exported evidence contains structured errors, not environment or arbitrary logs.
        using Process process = Process.Start(start) ?? throw new IOException("Could not start worker.");
        Task<string> output = DrainAsync(process.StandardOutput);
        Task<string> error = DrainAsync(process.StandardError);
        using CancellationTokenSource deadline = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        deadline.CancelAfter(timeout);
        int code;
        try
        {
            await process.WaitForExitAsync(deadline.Token);
            code = process.ExitCode;
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited)
                process.Kill(entireProcessTree: true);
            await process.WaitForExitAsync(CancellationToken.None);
            code = cancel.IsCancellationRequested ? 130 : 124;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(log))!);
        await File.WriteAllTextAsync(log, await output + Environment.NewLine + await error, CancellationToken.None);
        return code;
    }

    private static async Task<string> DrainAsync(StreamReader reader)
    {
        char[] buffer = new char[4096];
        System.Text.StringBuilder text = new();
        int count;
        while ((count = await reader.ReadAsync(buffer)) > 0)
            if (text.Length < 65536)
                text.Append(buffer, 0, Math.Min(count, 65536 - text.Length));
        return text.ToString();
    }
}
