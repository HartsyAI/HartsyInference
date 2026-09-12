using System.Diagnostics;
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
/// <para>Skips rather than fails when the local toolchain cannot reproduce the committed artifacts — no compiler,
/// a build that errors out, or a different compiler version — because a byte-compare is only meaningful against the
/// toolchain that produced them. All three are reported through <see cref="ITestOutputHelper"/> so an inconclusive
/// run does not read as a pass.</para></summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed class CudaKernelDriftTests
{
    private readonly ITestOutputHelper _out;
    public CudaKernelDriftTests(ITestOutputHelper output) => _out = output;

    /// <summary>Kernels deliberately left at their tuned register allocation. Per <c>Kernels/README.md</c> the
    /// sources are current and only codegen differs; both are LLM-decode hot paths whose throughput was measured
    /// against llama.cpp, so they are regenerated only alongside a perf run.</summary>
    private static readonly HashSet<string> TunedAllowlist = new(StringComparer.Ordinal)
    {
        "lm_f32",
        "mul_mat_vec_q6k_q8_1",
    };

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

        try
        {
            foreach (string domain in domains)
            {
                (int exit, string stdout, string stderr) = RunBuild(domain);
                if (exit != 0)
                {
                    _out.WriteLine($"SKIPPED: {Path.GetFileName(domain)}/build.sh exited {exit} — no usable toolchain " +
                        "(needs nvcc on PATH, or the committed Kernels/nvrtc_compile helper plus a COMPLETE header set; " +
                        "see Kernels/README.md). Inconclusive, not a pass.");
                    _out.WriteLine("stdout:\n" + stdout);
                    _out.WriteLine("stderr:\n" + stderr);
                    return;
                }
                rebuilt.AddRange(Directory.GetFiles(domain, "*.ptx"));
            }

            List<string> missing = new(), differing = new(), skippedTuned = new();
            foreach (string fresh in rebuilt.OrderBy(x => x, StringComparer.Ordinal))
            {
                string name = Path.GetFileNameWithoutExtension(fresh);
                string committed = Path.Combine(ptxDir, name + ".ptx");
                if (!File.Exists(committed)) { missing.Add(name); continue; }

                // A byte-compare across compiler versions is meaningless, so verify provenance first. nvcc and
                // nvrtc at the same NVVM version emit identical PTX (verified: msda.ptx matches byte-for-byte
                // through either), but a different CUDA release does not.
                string? freshTool = ToolchainLine(fresh), committedTool = ToolchainLine(committed);
                if (freshTool is not null && committedTool is not null && freshTool != committedTool)
                {
                    _out.WriteLine($"SKIPPED at {name}: local toolchain emits \"{freshTool}\" but the committed PTX " +
                        $"was built by \"{committedTool}\". A byte-compare across compiler versions is meaningless. " +
                        "Pin nvidia-cuda-nvcc==13.0.88 AND nvidia-nvvm==13.0.* (nvvm decides the ISA). Inconclusive.");
                    return;
                }

                if (TunedAllowlist.Contains(name)) { skippedTuned.Add(name); continue; }
                if (!SameKernel(fresh, committed)) differing.Add(name);
            }

            if (skippedTuned.Count > 0)
                _out.WriteLine("Allowlisted (hand-tuned register allocation, regenerate only with a perf run): " + string.Join(", ", skippedTuned));
            if (missing.Count > 0)
                _out.WriteLine("Built from source but NOT present in Ptx/ (new kernel never installed/committed): " + string.Join(", ", missing));
            if (differing.Count > 0)
                _out.WriteLine("Byte-differs from a fresh rebuild — the shipped kernel is NOT this source: " + string.Join(", ", differing));

            _out.WriteLine($"Checked {rebuilt.Count} rebuilt kernels across {domains.Length} domains against {ptxDir}.");

            Assert.True(missing.Count == 0 && differing.Count == 0,
                $"Committed Ptx/ has drifted from Kernels/ source: {missing.Count} missing, {differing.Count} changed. " +
                "Run the domain's build.sh and commit the regenerated Ptx/*.ptx alongside the .cu.");
        }
        finally
        {
            foreach (string f in rebuilt.Where(f => !preExisting.Contains(f)))
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

    private static (int Exit, string Stdout, string Stderr) RunBuild(string domainDir)
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
        // build.sh probes for its own toolchain; forward only what an operator may have overridden.
        foreach (string key in new[] { "CUDA_LIB", "CUDA_INC", "LD_LIBRARY_PATH" })
        {
            if (Environment.GetEnvironmentVariable(key) is string v) psi.Environment[key] = v;
        }
        using Process proc = Process.Start(psi)!;
        string stdout = proc.StandardOutput.ReadToEnd();
        string stderr = proc.StandardError.ReadToEnd();
        proc.WaitForExit();
        return (proc.ExitCode, stdout, stderr);
    }

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
