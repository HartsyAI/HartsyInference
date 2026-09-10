using System.Diagnostics;
using System.Text.RegularExpressions;
using HartsyInference.BenchmarkRunner.Evidence;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Explicit contributor command: stages a Release in a fork and opens a data-only pull request.</summary>
public static class SubmitCommand
{
    public static async Task RunAsync(string root, string bundle, string repository, string fork, CancellationToken cancel)
    {
        if (!Regex.IsMatch(fork, "^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$") || fork == "HartsyAI/HartsyInference")
            throw new ArgumentException("Provide your own existing public fork as owner/repository.");
        string extracted = Path.Combine(Path.GetTempPath(), "hartsy-submit-" + Guid.NewGuid().ToString("N"));
        Bundle.Extract(bundle, extracted);
        if (!Validator.Validate(extracted).Valid || Hashes.FileHash(Path.Combine(root, "campaign.json")) != Hashes.FileHash(Path.Combine(
            extracted, "campaign.json")))
            throw new InvalidDataException("Bundle does not match campaign.");
        string id = Hashes.FileHash(Path.Combine(root, "campaign.json"));
        string branch = "codex/benchmark-" + id[..12], tag = "benchmark-" + id;
        string worktree = Path.Combine(Path.GetTempPath(), "hartsy-pr-" + Guid.NewGuid().ToString("N"));
        await CommandAsync(repository, "gh", ["auth", "status"], cancel);
        await CommandAsync(repository, "git", ["fetch", "https://github.com/HartsyAI/HartsyInference.git", "main"], cancel);
        await CommandAsync(repository, "git", ["worktree", "add", "-b", branch, worktree, "FETCH_HEAD"], cancel);
        string upload = Path.Combine(extracted, id + ".zip");
        File.Copy(bundle, upload);
        await CommandAsync(repository, "gh", ["release", "create", tag, upload, "--repo", fork, "--title", "Benchmark evidence " + id[..12],
            "--notes", "Immutable community benchmark staging evidence."], cancel);
        Submissions.Create(root, bundle, Path.Combine(worktree, "benchmarks", "submissions"),
            $"https://github.com/{fork}/releases/download/{tag}/{id}.zip");
        await CommandAsync(worktree, "git", ["add", "--", "benchmarks/submissions/" + id], cancel);
        await CommandAsync(worktree, "git", ["commit", "-m", "bench: submit campaign " + id[..12]], cancel);
        await CommandAsync(worktree, "git", ["push", "https://github.com/" + fork + ".git", "HEAD:refs/heads/" + branch], cancel);
        string body = Path.Combine(extracted, "pr.md");
        await File.WriteAllTextAsync(body, $"Adds campaign {id}.\n\nThe complete evidence is staged in the linked fork Release. "
            + "Maintainers must validate the exact PR head, inspect outputs, and mirror the bundle before merging.\n", cancel);
        await CommandAsync(worktree, "gh", ["pr", "create", "--repo", "HartsyAI/HartsyInference", "--base", "main", "--head", fork.Split(
            '/')[0] + ":" + branch, "--title", "bench: community campaign " + id[..12], "--body-file", body], cancel);
        Console.WriteLine("Submission worktree retained at " + worktree);
    }

    private static async Task CommandAsync(string directory, string executable, string[] args, CancellationToken cancel)
    {
        ProcessStartInfo info = new(executable)
        {
            WorkingDirectory = directory,
            UseShellExecute = false
        };
        foreach (string argument in args)
            info.ArgumentList.Add(argument);
        using Process process = Process.Start(info) ?? throw new IOException("Could not start " + executable);
        try
        {
            await process.WaitForExitAsync(cancel);
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited)
                process.Kill(true);
            throw;
        }

        if (process.ExitCode != 0)
            throw new IOException(executable + " failed; inspect its output. Staged data was retained for recovery.");
    }
}
