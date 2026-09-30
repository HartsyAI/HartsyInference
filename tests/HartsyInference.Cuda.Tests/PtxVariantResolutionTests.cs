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
    [InlineData(89, "foo.sm89.ptx")]   // the card the variant was built for
    [InlineData(86, "foo.ptx")]        // another card: the baseline, never a foreign variant
    [InlineData(120, "foo.ptx")]
    [InlineData(0, "foo.ptx")]         // no device known
    public void VariantIsChosenOnlyForItsExactComputeCapability(int sm, string expectedFile)
    {
        string dir = Directory.CreateTempSubdirectory("ptx-resolve-").FullName;
        try
        {
            File.WriteAllText(Path.Combine(dir, "foo.ptx"), "// baseline");
            File.WriteAllText(Path.Combine(dir, "foo.sm89.ptx"), "// sm_89a");
            Assert.Equal(Path.Combine(dir, expectedFile), CudaKernels.PtxPath(dir, "foo", sm));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
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
    [Fact]
    public void TheRunningCardLoadsItsOwnVariant()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string shipped = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(shipped))
            shipped = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaContext context = new CudaContext(0);
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
