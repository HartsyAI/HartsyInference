using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Arch-specific PTX is chosen for the device's exact compute capability and for nothing else. The unit rows need no GPU; the last test loads a real kernel set with a variant present and checks the card picked it.</summary>
public sealed class PtxVariantResolutionTests
{
    private readonly ITestOutputHelper _output;
    public PtxVariantResolutionTests(ITestOutputHelper output) => _output = output;

    [Theory]
    // Baseline sm_80, variants sm_75 and sm_89, and an arch-specific sm_120a. Each card gets the highest target it can JIT.
    [InlineData(120, "foo.sm120a.ptx")]  // the arch-specific build for its own capability beats the forward-compatible sm_89
    [InlineData(121, "foo.sm89.ptx")]    // sm_120a is exact-only, so the highest forward-compatible build the card can run
    [InlineData(89, "foo.sm89.ptx")]     // the card the variant was built for
    [InlineData(86, "foo.ptx")]          // the sm_89 variant is too new, the sm_75 one is lower: the baseline is higher still
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
            WritePtx(dir, "foo.sm89.ptx", 84);   // the name says 89 (too new for an 86), the header says 84 (runnable)
            Assert.Equal(Path.Combine(dir, "foo.sm89.ptx"), CudaKernels.PtxPath(dir, "foo", 86));
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

    /// <summary>A copy of the shipped PTX with the baseline block_quant duplicated under this card's suffix: the kernel set must report it as the loaded variant, and only it. The duplicate is bit-identical to the baseline, so nothing else about the run changes.</summary>
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
            File.Copy(Path.Combine(dir, "block_quant.ptx"), Path.Combine(dir, $"block_quant.sm{context.Sm}.ptx"), overwrite: true);
            using CudaKernels kernels = new CudaKernels(dir, context);
            Assert.Equal(context.Sm, kernels.Sm);
            Assert.True(kernels.HasBlockQuantKernels);
            Assert.Equal(["block_quant"], kernels.ArchVariantsLoaded);
            _output.WriteLine($"SM {context.Sm} loaded variants: {string.Join(", ", kernels.ArchVariantsLoaded)}");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
