using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests;

/// <summary>The ratchet that makes removing environment variables a one-way door: a file not on the allowlist may not read the environment, and a file on it may only read it FEWER times than last commit.</summary>
/// <remarks>The engine accumulated ~195 environment variables read from ~111 files, in six mutually inconsistent
/// value grammars, with nine names documented as working that nothing read. That did not happen through one bad
/// decision — it happened because nothing stopped the next raw read, and a doc listing them was always one commit
/// out of date.
/// <para>This runs BEFORE the migration rather than after it on purpose: with the ratchet in place every later phase
/// is provably monotonic, so a migration cannot quietly reintroduce what it just removed. Deleting the last entry is
/// the definition of done for the config rebuild.</para>
/// <para>Deliberately a source scan rather than reflection over the compiled assemblies: the point is to catch the
/// literal call being written, including in code paths that never execute on this machine.</para></remarks>
public sealed class EnvReadAllowlistTests
{
    private readonly ITestOutputHelper _output;

    public EnvReadAllowlistTests(ITestOutputHelper output) => _output = output;

    /// <summary>Every way the codebase reaches the environment. <c>EnvSwitch</c> is included because it is a wrapper, not an exemption — the goal is zero environment reads, not tidier ones.</summary>
    private static readonly Regex EnvRead = new(
        @"Environment\.GetEnvironmentVariable\s*\(|EnvSwitch\.(IsEnabled|GetInt|GetFloat|GetLong)\s*\(",
        RegexOptions.Compiled);

    private static string AllowlistPath =>
        Path.Combine(RepoRoot.Path, "tests", "HartsyInference.Core.Tests", "env-read-allowlist.txt");

    /// <summary>Counts environment reads per source file under <c>src/</c>, ignoring build output.</summary>
    private static Dictionary<string, int> ScanSource()
    {
        Dictionary<string, int> found = new(StringComparer.Ordinal);
        string src = Path.Combine(RepoRoot.Path, "src");
        foreach (string file in Directory.EnumerateFiles(src, "*.cs", SearchOption.AllDirectories))
        {
            string rel = Path.GetRelativePath(RepoRoot.Path, file).Replace('\\', '/');
            if (rel.Contains("/obj/", StringComparison.Ordinal) || rel.Contains("/bin/", StringComparison.Ordinal))
            {
                continue;
            }
            int n = EnvRead.Matches(File.ReadAllText(file)).Count;
            if (n > 0)
            {
                found[rel] = n;
            }
        }
        return found;
    }

    private static Dictionary<string, int> ReadAllowlist()
    {
        Dictionary<string, int> allowed = new(StringComparer.Ordinal);
        foreach (string line in File.ReadAllLines(AllowlistPath))
        {
            string trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith('#'))
            {
                continue;
            }
            int sep = trimmed.LastIndexOf(' ');
            allowed[trimmed[..sep]] = int.Parse(trimmed[(sep + 1)..]);
        }
        return allowed;
    }

    /// <summary>A file that reads the environment and is not on the allowlist fails the build.</summary>
    [Fact]
    public void NoNewFileReadsTheEnvironment()
    {
        Dictionary<string, int> found = ScanSource();
        Dictionary<string, int> allowed = ReadAllowlist();

        List<string> offenders = [.. found.Keys.Where(f => !allowed.ContainsKey(f)).Order()];
        if (offenders.Count > 0)
        {
            StringBuilder sb = new();
            sb.AppendLine("These files read the environment but are not on the allowlist:");
            foreach (string f in offenders) sb.AppendLine($"  {f} ({found[f]} read(s))");
            sb.AppendLine();
            sb.AppendLine("The engine is moving to typed configuration — see docs/ENV_VARS.md. Take the value as a");
            sb.AppendLine("parameter or put it on a typed options/policy record instead of reading it here.");
            Assert.Fail(sb.ToString());
        }
    }

    /// <summary>An allowlisted file may only ever read the environment FEWER times. This is the ratchet.</summary>
    [Fact]
    public void AllowlistedFilesOnlyShrink()
    {
        Dictionary<string, int> found = ScanSource();
        Dictionary<string, int> allowed = ReadAllowlist();

        List<string> grew = [.. allowed.Keys
            .Where(f => found.GetValueOrDefault(f) > allowed[f])
            .Select(f => $"  {f}: allowlist says {allowed[f]}, found {found[f]}")
            .Order()];
        if (grew.Count > 0)
        {
            Assert.Fail("These files gained environment reads:\n" + string.Join("\n", grew)
                + "\n\nThe allowlist shrinks only. Add the setting to a typed options record instead.");
        }
    }

    /// <summary>Entries that have reached zero must be deleted, so the allowlist cannot quietly hold budget for reads nobody is making.</summary>
    /// <remarks>Without this, a migrated file keeps its line and a later commit can re-add reads up to the stale
    /// number without either other test noticing — the ratchet would have slack in it.</remarks>
    [Fact]
    public void AllowlistHasNoStaleEntries()
    {
        Dictionary<string, int> found = ScanSource();
        Dictionary<string, int> allowed = ReadAllowlist();

        List<string> stale = [.. allowed.Keys
            .Where(f => found.GetValueOrDefault(f) < allowed[f])
            .Select(f => $"  {f}: allowlist says {allowed[f]}, found {found.GetValueOrDefault(f)}")
            .Order()];
        if (stale.Count > 0)
        {
            Assert.Fail("These allowlist entries are stale — lower the number, or delete the line if it is now 0:\n"
                + string.Join("\n", stale));
        }
    }

    /// <summary>Reports remaining reads, so the shrink is visible in the test log across the migration.</summary>
    [Fact]
    public void ReportRemainingEnvironmentReads()
    {
        Dictionary<string, int> found = ScanSource();
        int total = found.Values.Sum();
        _output.WriteLine($"{total} environment read(s) remain across {found.Count} file(s).");
        foreach ((string file, int n) in found.OrderByDescending(kv => kv.Value).Take(15))
        {
            _output.WriteLine($"  {n,3}  {file}");
        }
        Assert.True(total >= 0);
    }
}
