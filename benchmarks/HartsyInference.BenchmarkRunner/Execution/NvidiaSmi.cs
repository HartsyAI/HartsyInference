using System.Diagnostics;
using HartsyInference.Core.Logging;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>nvidia-smi transport for attestation and sampling. NVML is deliberately not used: its process-list
/// entry point is struct-size versioned and its throttle reasons are header constants, while nvidia-smi names
/// both as CSV columns that can be read without a toolkit installed.</summary>
public static class NvidiaSmi
{
    private const string Executable = "nvidia-smi";

    /// <summary>Runs one bounded query. Null means nvidia-smi is absent, failed, or exceeded the deadline;
    /// every caller treats that as unattested rather than as an error.</summary>
    public static string[]? Query(string[] arguments, TimeSpan timeout)
    {
        try
        {
            using Process? process = Start(arguments);
            if (process is null)
                return null;
            Task<string> output = process.StandardOutput.ReadToEndAsync();
            if (!process.WaitForExit((int)timeout.TotalMilliseconds))
            {
                Kill(process);
                return null;
            }

            return process.ExitCode == 0
                ? output.GetAwaiter().GetResult().Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                : null;
        }
        catch (Exception error)when (error is not OutOfMemoryException)
        {
            return null;
        }
    }

    /// <summary>Starts a long-lived query without waiting; the caller owns draining and termination.</summary>
    public static Process? Start(string[] arguments)
    {
        ProcessStartInfo start = new(Executable)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        foreach (string argument in arguments)
            start.ArgumentList.Add(argument);
        try
        {
            return Process.Start(start);
        }
        catch (Exception error)when (error is not OutOfMemoryException)
        {
            return null;
        }
    }

    /// <summary>Terminates by process handle. Never by name pattern: a pattern kill on this host has repeatedly
    /// matched the wrong process.</summary>
    public static void Kill(Process process)
    {
        try
        {
            if (!process.HasExited)
                process.Kill(entireProcessTree: true);
        }
        catch (Exception error)when (error is not OutOfMemoryException)
        {
            Logs.Debug("Telemetry sampler had already exited: " + error.GetType().Name);
        }
        finally
        {
            // Unconditional: a Kill that throws must not skip confirming the child actually left, or the
            // caller disposes the stream while the reader is still blocked on a live process.
            try
            {
                process.WaitForExit(5000);
            }
            catch (Exception error)when (error is not OutOfMemoryException)
            {
                Logs.Debug("Telemetry sampler could not be waited on: " + error.GetType().Name);
            }
        }
    }

    /// <summary>Bare lowercase hex for either spelling. <c>cuDeviceGetUuid</c> yields 16 raw bytes while
    /// nvidia-smi prints <c>GPU-8-4-4-4-12</c>, so the two are only comparable after normalization.</summary>
    public static string NormalizeUuid(string value) => value.Trim().Replace("GPU-", "", StringComparison.OrdinalIgnoreCase)
        .Replace("-", "", StringComparison.Ordinal).ToLowerInvariant();

    /// <summary>Splits one CSV line from <c>--format=csv,noheader,nounits</c>.</summary>
    public static string[] Fields(string line) => line.Split(',', StringSplitOptions.TrimEntries);
}
