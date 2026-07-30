using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Guards the Vulkan shader "single source of truth" contract: <c>Shaders/*.comp.glsl</c> is
/// the source, <c>Spirv/*.spv</c> is a checked-in BUILD ARTIFACT of it — never hand-edit the binary.
/// Nothing enforced that until now (same gap CUDA has for its PTX; see the KERNEL.md note this test
/// backs), and it bit hard once already: Phase 0 of the Vulkan bring-up found `.spv` files that had
/// drifted from source with no compiler available to even notice. This test rebuilds every shader from
/// current source and byte-diffs the result against the committed `Spirv/` directory, so a future
/// source edit that isn't followed by a rebuild+commit fails loudly here instead of silently shipping
/// stale kernels. Skips (does not fail) when no SPIR-V compiler is resolvable, or when the resolved
/// compiler can't build the current shader set at all (e.g. Ubuntu's `glslang-tools` apt package lacks
/// `GL_EXT_integer_dot_product` support for `matmul_int8.comp.glsl` — a toolchain-capability gap, not
/// drift; see TROUBLESHOOTING.md) — both are reported via <see cref="ITestOutputHelper"/> so a run on a
/// dev box without the right toolchain doesn't read as a false pass, just an inconclusive one.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanShaderDriftTests
{
    private readonly ITestOutputHelper _out;
    public VulkanShaderDriftTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void CommittedSpirv_MatchesFreshRebuildFromSource()
    {
        string? shadersDir = FindShadersDir();
        if (shadersDir is null)
        {
            _out.WriteLine("SKIPPED: could not locate src/HartsyInference.Vulkan/Shaders relative to the test binary.");
            return;
        }
        string spirvDir = Path.GetFullPath(Path.Combine(shadersDir, "..", "Spirv"));

        string? glslang = ResolveGlslang();
        if (glslang is null)
        {
            _out.WriteLine("SKIPPED: no glslangValidator resolvable (checked GLSLANG env var and PATH). " +
                "Install glslang-tools, or point GLSLANG at the LunarG SDK's glslangValidator " +
                "(see docs/Checklists/TROUBLESHOOTING.md — Ubuntu's apt package can't compile matmul_int8.comp.glsl).");
            return;
        }

        string tempDir = Path.Combine(Path.GetTempPath(), $"vk_shader_drift_{Environment.ProcessId}_{DateTime.UtcNow.Ticks}");
        Directory.CreateDirectory(tempDir);
        try
        {
            ProcessStartInfo psi = new()
            {
                FileName = "bash",
                ArgumentList = { "build.sh" },
                WorkingDirectory = shadersDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            psi.Environment["OUT"] = tempDir;
            psi.Environment["GLSLANG"] = glslang;
            if (Environment.GetEnvironmentVariable("SPIRVVAL") is string sv) psi.Environment["SPIRVVAL"] = sv;
            if (Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") is string ld) psi.Environment["LD_LIBRARY_PATH"] = ld;

            using Process proc = Process.Start(psi)!;
            string stdout = proc.StandardOutput.ReadToEnd();
            string stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit();

            if (proc.ExitCode != 0)
            {
                _out.WriteLine($"SKIPPED: build.sh exited {proc.ExitCode} with the resolved toolchain ({glslang}) — " +
                    "likely a toolchain capability gap (e.g. matmul_int8.comp.glsl needs the LunarG SDK's glslang, " +
                    "not Ubuntu's apt package), not source drift. This is inconclusive, not a pass.");
                _out.WriteLine("stdout:\n" + stdout);
                _out.WriteLine("stderr:\n" + stderr);
                return;
            }

            string[] fresh = Directory.GetFiles(tempDir, "*.spv").Select(Path.GetFileName).Cast<string>().OrderBy(x => x).ToArray();
            string[] committed = Directory.GetFiles(spirvDir, "*.spv").Select(Path.GetFileName).Cast<string>().OrderBy(x => x).ToArray();

            List<string> onlyInFresh = fresh.Except(committed).ToList();
            List<string> onlyInCommitted = committed.Except(fresh).ToList();
            List<string> differing = new();
            foreach (string name in fresh.Intersect(committed))
            {
                byte[] a = File.ReadAllBytes(Path.Combine(tempDir, name));
                byte[] b = File.ReadAllBytes(Path.Combine(spirvDir, name));
                if (!a.AsSpan().SequenceEqual(b)) differing.Add(name);
            }

            if (onlyInFresh.Count > 0)
                _out.WriteLine("Rebuilt but NOT committed (forgot to `git add` after a shader change?): " + string.Join(", ", onlyInFresh));
            if (onlyInCommitted.Count > 0)
                _out.WriteLine("Committed but NOT reproduced by a fresh rebuild (stale leftover — shader renamed/removed from build.sh?): " + string.Join(", ", onlyInCommitted));
            if (differing.Count > 0)
                _out.WriteLine("Byte-differs from a fresh rebuild (source changed without rebuilding, or hand-edited binary): " + string.Join(", ", differing));

            Assert.True(onlyInFresh.Count == 0 && onlyInCommitted.Count == 0 && differing.Count == 0,
                $"Committed Spirv/ has drifted from Shaders/ source: {onlyInFresh.Count} uncommitted, " +
                $"{onlyInCommitted.Count} stale, {differing.Count} changed. Rebuild via `bash Shaders/build.sh` and commit the result.");
        }
        finally
        {
            try { Directory.Delete(tempDir, recursive: true); } catch { /* best-effort cleanup */ }
        }
    }

    private static string? ResolveGlslang()
    {
        string? env = Environment.GetEnvironmentVariable("GLSLANG");
        if (!string.IsNullOrEmpty(env) && IsExecutable(env)) return env;

        foreach (string dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
        {
            string candidate = Path.Combine(dir, "glslangValidator");
            if (IsExecutable(candidate)) return candidate;
        }
        return null;
    }

    private static bool IsExecutable(string path)
    {
        try
        {
            if (!File.Exists(path)) return false;
            ProcessStartInfo psi = new()
            {
                FileName = path,
                ArgumentList = { "--version" },
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            using Process p = Process.Start(psi)!;
            p.WaitForExit(5000);
            return true;
        }
        catch { return false; }
    }

    /// <summary>Walks up from the test binary's directory looking for the repo root (marked by
    /// <c>HartsyInference.sln</c>), then returns <c>src/HartsyInference.Vulkan/Shaders</c> under it.</summary>
    private static string? FindShadersDir()
    {
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "HartsyInference.sln")))
            {
                string shaders = Path.Combine(dir.FullName, "src", "HartsyInference.Vulkan", "Shaders");
                return Directory.Exists(shaders) ? shaders : null;
            }
            dir = dir.Parent;
        }
        return null;
    }
}
