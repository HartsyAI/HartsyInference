using System.Text.RegularExpressions;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Arch-specific PTX is chosen for the device's exact compute capability and for nothing else. The unit rows need no GPU; the last test loads a real kernel set with a variant present and checks the card picked it.</summary>
[Collection("CudaSerial")]
public sealed class PtxVariantResolutionTests
{
    private readonly ITestOutputHelper _output;
    public PtxVariantResolutionTests(ITestOutputHelper output) => _output = output;

    [Theory]
    // Baseline sm_80, Turing variant sm_75, tuned variants sm_89 and sm_120a. Each card gets the highest target it can JIT,
    // where a tuned variant above the Turing floor runs only on its own capability.
    [InlineData(120, "foo.sm120a.ptx")]  // the arch-specific build for its own capability beats the forward-compatible sm_89
    [InlineData(121, "foo.ptx")]         // sm_89 is tuned, so it does not run here; sm_120a is exact-only; the baseline is the floor
    [InlineData(90, "foo.ptx")]          // Hopper: the sm_89 tuned build is not run on another capability
    [InlineData(89, "foo.sm89.ptx")]     // the card the variant was built for
    [InlineData(86, "foo.ptx")]          // sm_89 is not this card's; the sm_75 floor build is lower than the baseline
    [InlineData(75, "foo.sm75.ptx")]     // Turing: the baseline sm_80 is too new, so the sm_75 build is the only one it can run
    [InlineData(70, "foo.ptx")]          // older than every variant: the baseline comes back and the gate reports it skipped
    [InlineData(0, "foo.ptx")]           // no device known
    public void HighestBuildTheDeviceCanRunIsChosen(int sm, string expectedFile)
    {
        string dir = Directory.CreateTempSubdirectory("ptx-resolve-").FullName;
        try
        {
            WritePtx(dir, "foo.ptx", 80);
            WritePtx(dir, "foo.sm75.ptx", 75);
            WritePtx(dir, "foo.sm89.ptx", 89);
            WritePtx(dir, "foo.sm120a.ptx", 120, archSpecific: true);
            Assert.Equal(Path.Combine(dir, expectedFile), CudaKernels.PtxPath(dir, "foo", sm));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ArchSpecificBuildIsNeverRunOnAnotherCapability()
    {
        string dir = Directory.CreateTempSubdirectory("ptx-resolve-").FullName;
        try
        {
            WritePtx(dir, "foo.ptx", 80);
            WritePtx(dir, "foo.sm120a.ptx", 120, archSpecific: true);
            Assert.Equal(Path.Combine(dir, "foo.ptx"), CudaKernels.PtxPath(dir, "foo", 89));
            Assert.Equal(Path.Combine(dir, "foo.ptx"), CudaKernels.PtxPath(dir, "foo", 121));
            Assert.Equal(Path.Combine(dir, "foo.sm120a.ptx"), CudaKernels.PtxPath(dir, "foo", 120));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void VariantNamedByItsHeaderNotItsFileName()
    {
        string dir = Directory.CreateTempSubdirectory("ptx-resolve-").FullName;
        try
        {
            WritePtx(dir, "foo.ptx", 80);
            WritePtx(dir, "foo.sm89.ptx", 84);   // the name says 89, the header says 84: the header decides which card it is built for
            Assert.Equal(Path.Combine(dir, "foo.sm89.ptx"), CudaKernels.PtxPath(dir, "foo", 84));
            Assert.Equal(Path.Combine(dir, "foo.ptx"), CudaKernels.PtxPath(dir, "foo", 89));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static void WritePtx(string dir, string file, int sm, bool archSpecific = false)
    {
        string target = $"sm_{sm}{(archSpecific ? "a" : "")}";
        File.WriteAllLines(Path.Combine(dir, file), ["//", $".version 9.0", $".target {target}", ".address_size 64"]);
    }

    [Fact]
    public void WithoutAVariantTheBaselineIsUsedEvenForThatCard()
    {
        string dir = Directory.CreateTempSubdirectory("ptx-resolve-").FullName;
        try
        {
            File.WriteAllText(Path.Combine(dir, "foo.ptx"), "// baseline");
            Assert.Equal(Path.Combine(dir, "foo.ptx"), CudaKernels.PtxPath(dir, "foo", 89));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>A copy of the shipped PTX with the baseline block_quant duplicated under this card's suffix, its <c>.target</c> set to this card: the kernel set must report it as the loaded variant, and only it. The body is the baseline, so nothing else about the run changes.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void TheRunningCardLoadsItsOwnVariant()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string shipped = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(shipped))
            shipped = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaContext context = new CudaContext(0);
        // The copy's body is the sm_80 baseline, so only a card that runs sm_80 code can load it.
        if (context.Sm < CudaArch.Ampere) { _output.WriteLine($"SKIPPED: SM {context.Sm} cannot run the sm_80 body of the copy"); return; }
        string dir = Directory.CreateTempSubdirectory("ptx-variant-").FullName;
        try
        {
            foreach (string file in Directory.EnumerateFiles(shipped, "*.ptx"))
                File.Copy(file, Path.Combine(dir, Path.GetFileName(file)));
            string baselineText = File.ReadAllText(Path.Combine(dir, "block_quant.ptx"));
            File.WriteAllText(Path.Combine(dir, $"block_quant.sm{context.Sm}.ptx"),
                Regex.Replace(baselineText, @"^\.target .*$", $".target sm_{context.Sm}", RegexOptions.Multiline));
            using CudaKernels kernels = new CudaKernels(dir, context);
            Assert.Equal(context.Sm, kernels.Sm);
            Assert.True(kernels.HasBlockQuantKernels);
            // The copied block_quant is the variant under test. Other shipped variants for this card (int8_attn_v2 on sm_89)
            // also load, and that is correct, so the assertion is on block_quant alone.
            Assert.Contains("block_quant", kernels.ArchVariantsLoaded);
            _output.WriteLine($"SM {context.Sm} loaded variants: {string.Join(", ", kernels.ArchVariantsLoaded)}");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
