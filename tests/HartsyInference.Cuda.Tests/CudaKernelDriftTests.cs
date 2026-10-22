using System.Diagnostics;
using System.Text.RegularExpressions;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Guards the CUDA kernel "single source of truth" contract: <c>Kernels/&lt;domain&gt;/*.cu</c> is the
/// source, <c>Ptx/*.ptx</c> is a checked-in BUILD ARTIFACT of it. Nothing enforced that — the gap
/// <see cref="HartsyInference.Vulkan.Tests"/>' shader drift test closed for SPIR-V was left open here, and it bit
/// once already (commit <c>418333a7</c> shipped 8 kernels whose PTX had drifted months from their source).
///
/// <para>The failure is silent by construction: <c>dotnet build</c> never compiles a <c>.cu</c> — MSBuild only
/// copies the committed PTX — and there is no nvrtc fallback in the C# runtime. So an edited kernel whose PTX was
/// not regenerated and committed simply keeps running the old code, with nothing logged at any layer. This test
/// rebuilds every domain from current source and byte-diffs the result.</para>
///
/// <para>Each kernel is rebuilt with the compiler its PTX shipped from: the nvrtc helper, or the CUDA 13.0.88 nvcc
/// (<c>HARTSY_NVCC</c>, PATH, or the workstation's pinned install) for a domain's <c>NVCC_BUILT</c> list. The two stamp the same toolchain line yet
/// emit different PTX, so a kernel whose compiler is missing is reported UNVERIFIED instead of compared. A build that
/// errors out skips the test, and a kernel built by a different CUDA release than its committed PTX is reported
/// instead of compared, because a byte-compare is only meaningful against the toolchain that produced it. All of
/// these go to <see cref="ITestOutputHelper"/> so an inconclusive run does not read as a pass.</para>
/// <para>An arch variant (<c>Ptx/&lt;kernel&gt;.sm&lt;NNN&gt;.ptx</c>) is rebuilt with <c>--arch sm_&lt;NNN&gt;a</c> and compared
/// too. The plain rebuild compiles the other branch of its <c>#if</c>, which is how a source revert of the sm_120a e2m1
/// cvt went unseen while the shipped variant had been patched by hand. A PTX the assembler refuses fails the test
/// rather than reading as a missing toolchain.</para></summary>
[Collection("CudaSerial")]
public sealed class CudaKernelDriftTests
{
    private const string UnverifiedPrefix = "UNVERIFIED ";
    private readonly ITestOutputHelper _out;
    public CudaKernelDriftTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void CommittedPtx_MatchesFreshRebuildFromSource()
    {
        string? kernelsDir = FindKernelsDir();
        if (kernelsDir is null)
        {
            _out.WriteLine("SKIPPED: could not locate src/HartsyInference.Cuda/Kernels relative to the test binary.");
            return;
        }
        string ptxDir = Path.GetFullPath(Path.Combine(kernelsDir, "..", "Ptx"));

        string[] domains = Directory.GetDirectories(kernelsDir)
            .Where(d => File.Exists(Path.Combine(d, "build.sh")))
            .OrderBy(d => d, StringComparer.Ordinal)
            .ToArray();
        if (domains.Length == 0)
        {
            _out.WriteLine($"SKIPPED: no domain build.sh found under {kernelsDir}.");
            return;
        }

        // build.sh --no-install writes <domain>/<kernel>.ptx (gitignored intermediates). Remember which already
        // existed so the test leaves the tree as it found it.
        HashSet<string> preExisting = new(domains.SelectMany(d => Directory.GetFiles(d, "*.ptx")), StringComparer.Ordinal);
        List<string> rebuilt = new();
        List<string> unverified = new();
        List<string> refused = new();
        List<string> cleanup = new();
        List<(string Fresh, string Committed)> variants = new();

        try
        {
            foreach (string domain in domains)
            {
                (int exit, string stdout, string stderr) = RunBuild(domain);
                if (exit != 0 && AssemblerRefusal(stderr) is { } refusal)
                {
                    refused.Add($"{Path.GetFileName(domain)}: {refusal}");
                    cleanup.AddRange(Directory.GetFiles(domain, "*.ptx"));
                    continue;
                }
                if (exit != 0)
                {
                    _out.WriteLine($"SKIPPED: {Path.GetFileName(domain)}/build.sh exited {exit} — no usable toolchain " +
                        "(needs the committed Kernels/nvrtc_compile helper plus a COMPLETE header set; " +
                        "see Kernels/README.md). Inconclusive, not a pass.");
                    _out.WriteLine("stdout:\n" + stdout);
                    _out.WriteLine("stderr:\n" + stderr);
                    AssertNoneRefused(refused);
                    return;
                }
                unverified.AddRange(stdout.Split('\n')
                    .Where(l => l.StartsWith(UnverifiedPrefix, StringComparison.Ordinal))
                    .Select(l => l[UnverifiedPrefix.Length..].Trim()));
                rebuilt.AddRange(Directory.GetFiles(domain, "*.ptx").Where(f => !VariantName.IsMatch(Path.GetFileName(f))));
            }

            foreach (string committed in Directory.GetFiles(ptxDir, "*.ptx").OrderBy(x => x, StringComparer.Ordinal))
            {
                Match variant = VariantName.Match(Path.GetFileName(committed));
                if (!variant.Success) continue;
                string kernel = variant.Groups["kernel"].Value, sm = variant.Groups["sm"].Value;
                string? owner = domains.FirstOrDefault(d => File.Exists(Path.Combine(d, kernel + ".cu")));
                if (owner is null)
                {
                    refused.Add($"{Path.GetFileName(committed)}: no Kernels/<domain>/{kernel}.cu builds it");
                    continue;
                }
                // The kernel filter also rebuilds the baseline <kernel>.ptx; cleanup covers it.
                (int exit, string stdout, string stderr) = RunBuild(owner, "--arch", $"sm_{sm}a", kernel);
                if (exit != 0 && AssemblerRefusal(stderr) is { } refusal)
                {
                    refused.Add($"{Path.GetFileName(committed)}: {refusal}");
                    cleanup.Add(Path.Combine(owner, $"{kernel}.sm{sm}.ptx"));
                    continue;
                }
                if (exit != 0)
                {
                    _out.WriteLine($"SKIPPED: {Path.GetFileName(owner)}/build.sh --arch sm_{sm}a exited {exit} — this toolchain " +
                        "cannot build that arch. Inconclusive, not a pass.");
                    _out.WriteLine("stderr:\n" + stderr);
                    AssertNoneRefused(refused);
                    return;
                }
                unverified.AddRange(stdout.Split('\n')
                    .Where(l => l.StartsWith(UnverifiedPrefix, StringComparison.Ordinal))
                    .Select(l => l[UnverifiedPrefix.Length..].Trim()));
                string fresh = Path.Combine(owner, $"{kernel}.sm{sm}.ptx");
                rebuilt.Add(fresh);
                variants.Add((fresh, committed));
            }

            // An UNVERIFIED kernel was not rebuilt, so any intermediate beside its .cu predates this run.
            HashSet<string> unverifiedNames = new(unverified.Select(KernelOf), StringComparer.Ordinal);
            List<string> missing = new(), differing = new(), otherRelease = new();
            int compared = 0;
            IEnumerable<(string Fresh, string Committed)> pairs = rebuilt
                .Where(f => !VariantName.IsMatch(Path.GetFileName(f)))
                .Select(f => (f, Path.Combine(ptxDir, Path.GetFileName(f))))
                .Concat(variants)
                .OrderBy(p => p.Item1, StringComparer.Ordinal);
            foreach ((string fresh, string committed) in pairs)
            {
                string name = Path.GetFileNameWithoutExtension(fresh);
                if (unverifiedNames.Contains(name)) continue;
                if (!File.Exists(committed)) { missing.Add(name); continue; }

                // nvcc and nvrtc of one release stamp the same line, so this catches another release, not the other
                // frontend; the domain lists decide the frontend.
                string? freshTool = ToolchainLine(fresh), committedTool = ToolchainLine(committed);
                if (freshTool is not null && committedTool is not null && freshTool != committedTool)
                {
                    otherRelease.Add($"{name} (\"{freshTool}\", committed \"{committedTool}\")");
                    continue;
                }

                compared++;
                if (!SameKernel(fresh, committed)) differing.Add(name);
            }

            if (unverified.Count > 0)
                _out.WriteLine("UNVERIFIED, not compared (set HARTSY_NVCC to the CUDA 13.0.88 nvcc to check them):\n  " +
                    string.Join("\n  ", unverified));
            if (otherRelease.Count > 0)
                _out.WriteLine("SKIPPED, built by another CUDA release than the committed PTX, so a byte-compare is meaningless " +
                    "(pin nvidia-cuda-nvcc==13.0.88 AND nvidia-nvvm==13.0.*; nvvm decides the ISA): " +
                    string.Join(", ", otherRelease));
            if (missing.Count > 0)
                _out.WriteLine("Built from source but NOT present in Ptx/ (new kernel never installed/committed): " + string.Join(", ", missing));
            if (differing.Count > 0)
                _out.WriteLine("Byte-differs from a fresh rebuild — the shipped kernel is NOT this source: " + string.Join(", ", differing));
            if (refused.Count > 0)
                _out.WriteLine("REFUSED by the assembler or unbuildable from source: " + string.Join("; ", refused));

            _out.WriteLine($"Compared {compared} rebuilt kernels ({variants.Count} arch variants) across {domains.Length} domains against {ptxDir}.");
            AssertNoneRefused(refused);
            if (compared == 0 && missing.Count == 0)
            {
                _out.WriteLine("SKIPPED: no rebuilt kernel could be compared — every one was UNVERIFIED or built by another " +
                    "CUDA release. Inconclusive, not a pass.");
                return;
            }

            Assert.True(missing.Count == 0 && differing.Count == 0,
                $"Committed Ptx/ has drifted from Kernels/ source: {missing.Count} missing, {differing.Count} changed. " +
                "Run the domain's build.sh and commit the regenerated Ptx/*.ptx alongside the .cu.");
        }
        finally
        {
            foreach (string f in rebuilt.Concat(cleanup).Distinct().Where(f => !preExisting.Contains(f)))
            {
                try { File.Delete(f); } catch { /* best-effort cleanup of a gitignored intermediate */ }
            }
        }
    }

    /// <summary>Whether two PTX files are the same kernel. PTX is text, so trailing whitespace carries no meaning
    /// and is not drift — <c>h3_vsa.ptx</c> ships with exactly one extra blank line at EOF and is otherwise
    /// byte-identical to a fresh rebuild (same 4 entry points, same body). Everything before the trailing
    /// whitespace must still match exactly.</summary>
    private static bool SameKernel(string a, string b)
        => string.Equals(File.ReadAllText(a).TrimEnd(), File.ReadAllText(b).TrimEnd(), StringComparison.Ordinal);

    /// <summary>The kernel an UNVERIFIED line names: build.sh writes <c>UNVERIFIED &lt;kernel&gt;: &lt;reason&gt;</c>.</summary>
    private static string KernelOf(string unverifiedLine)
    {
        int end = unverifiedLine.IndexOfAny([':', ' ']);
        return end > 0 ? unverifiedLine[..end] : unverifiedLine;
    }

    /// <summary>The compiler-identity line NVVM stamps into every PTX it emits, e.g.
    /// <c>// Cuda compilation tools, release 13.0, V13.0.88</c>.</summary>
    private static string? ToolchainLine(string ptxPath)
    {
        foreach (string line in File.ReadLines(ptxPath).Take(12))
        {
            if (line.StartsWith("// Cuda compilation tools", StringComparison.Ordinal)) return line.Trim();
        }
        return null;
    }

    /// <summary><c>Ptx/&lt;kernel&gt;.sm&lt;NNN&gt;.ptx</c>, an arch variant build_common.sh emits for <c>--arch sm_&lt;NNN&gt;a</c>.</summary>
    private static readonly Regex VariantName = new(@"^(?<kernel>.+)\.sm(?<sm>\d+)\.ptx$", RegexOptions.CultureInvariant);

    /// <summary>The assembler's refusal build_common.sh prints, or null when the build failed for another reason.</summary>
    private static string? AssemblerRefusal(string stderr)
        => stderr.Split('\n').FirstOrDefault(l => l.Contains("does not assemble", StringComparison.Ordinal))?.Trim();

    private static (int Exit, string Stdout, string Stderr) RunBuild(string domainDir, params string[] extra)
    {
        ProcessStartInfo psi = new()
        {
            FileName = "bash",
            ArgumentList = { "build.sh", "--no-install" },
            WorkingDirectory = domainDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (string arg in extra) psi.ArgumentList.Add(arg);
        // build.sh probes for its own toolchain; forward only what an operator may have overridden.
        foreach (string key in new[] { "CUDA_LIB", "CUDA_INC", "HARTSY_NVCC", "LD_LIBRARY_PATH" })
        {
            if (Environment.GetEnvironmentVariable(key) is string v) psi.Environment[key] = v;
        }
        using Process proc = Process.Start(psi)!;
        string stdout = proc.StandardOutput.ReadToEnd();
        string stderr = proc.StandardError.ReadToEnd();
        proc.WaitForExit();
        return (proc.ExitCode, stdout, stderr);
    }

    /// <summary>A refusal already seen stays a failure even when a later build makes the run inconclusive.</summary>
    private static void AssertNoneRefused(List<string> refused)
        => Assert.True(refused.Count == 0, "A committed kernel cannot be rebuilt from its source: " + string.Join("; ", refused));

    /// <summary>Walks up from the test binary looking for the repo root (marked by <c>HartsyInference.sln</c>),
    /// then returns <c>src/HartsyInference.Cuda/Kernels</c> under it.</summary>
    private static string? FindKernelsDir()
    {
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "HartsyInference.sln")))
            {
                string kernels = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Kernels");
                return Directory.Exists(kernels) ? kernels : null;
            }
            dir = dir.Parent;
        }
        return null;
    }
}
