using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;
using Checks = HartsyInference.Core.Tests.Backends.BlockSparseAttentionDeviceChecks;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Block-sparse attention on CUDA against the CPU reference and against LongCat-Video's kernel
/// (<see cref="BlockSparseAttentionDeviceChecks"/>).</summary>
[Trait("Category", "GpuIntegration")]
public sealed class BlockSparseAttentionGpuTests(ITestOutputHelper log)
{
    [CudaTheory]
    [MemberData(nameof(Checks.ReferenceCases), MemberType = typeof(Checks))]
    public void MatchesTheReferenceKernelsVectors(string name)
    {
        using CudaBackend cuda = Open();
        Checks.MatchesTheReferenceKernelsVectors(cuda, log, name);
    }

    [CudaTheory]
    [MemberData(nameof(Checks.ReferenceCases), MemberType = typeof(Checks))]
    public void MatchesTheCpuReferenceInEveryDtype(string name)
    {
        using CudaBackend cuda = Open();
        Checks.MatchesTheCpuReferenceInEveryDtype(cuda, log, name);
    }

    [CudaTheory]
    [MemberData(nameof(Checks.RandomShapes), MemberType = typeof(Checks))]
    public void RandomShapesMatchTheCpuReference(int seed, int batch, int heads, int dim, int[] queryGrid, int[] keyGrid,
        int[] queryBlock, int[] keyBlock, double sparsity, float scale)
    {
        using CudaBackend cuda = Open();
        Checks.MatchesTheCpuReferenceOnRandomShapes(cuda, log, seed, batch, heads, dim, queryGrid, keyGrid, queryBlock, keyBlock,
            sparsity, scale);
    }

    [CudaFact]
    public void SelectsAsTheReferenceDoesAmongManyKeyBlocks()
    {
        using CudaBackend cuda = Open();
        Checks.SelectsAsTheReferenceDoesAmongManyKeyBlocks(cuda, log);
    }

    [CudaTheory]
    [MemberData(nameof(Checks.SpecialCases), MemberType = typeof(Checks))]
    public void SpecialScoresMatchTheCpuReference(string name)
    {
        using CudaBackend cuda = Open();
        Checks.SpecialScoresMatchTheCpuReference(cuda, log, name);
    }

    [CudaFact]
    public void ResidentOperandsAreReadAndNothingIsSyncedUntilAResultIs()
    {
        using CudaBackend cuda = Open();
        Checks.ResidentOperandsAreReadAndNothingIsSyncedUntilAResultIs(cuda);
    }

    /// <summary>12152 key blocks fill the selection's 48 KB of shared memory: 4 bytes a block beside 544 static ones.</summary>
    [CudaFact]
    public void TheMostKeyBlocksTheSelectionHoldsStillRun()
    {
        using CudaBackend cuda = Open();
        Checks.TheMostKeyBlocksStillRun(cuda, log, 12152);
    }

    [CudaFact]
    public void AnUncompiledHeadDimAndTooManyKeyBlocksAreRefusedBeforeAnyWrite()
    {
        using CudaBackend cuda = Open();
        Checks.Refused(cuda, new() { QueryGrid = new(4, 4, 4), KeyGrid = new(4, 4, 4), Sparsity = 0.0 }, 24);
        Checks.Refused(cuda, new() { QueryGrid = new(4, 4, 4), KeyGrid = new(4, 4, 4 * 12153), Sparsity = 0.9 }, 16);
    }

    private static CudaBackend Open()
    {
        string beside = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(beside)) return new CudaBackend(0, beside);
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return new CudaBackend(0, candidate);
        }
        throw new DirectoryNotFoundException("No CUDA PTX directory beside the test assembly or in the repository.");
    }

    private sealed class CudaFactAttribute : FactAttribute
    {
        public CudaFactAttribute()
        {
            if (!CudaContext.IsAvailable()) Skip = "SKIPPED: CUDA unavailable.";
        }
    }

    private sealed class CudaTheoryAttribute : TheoryAttribute
    {
        public CudaTheoryAttribute()
        {
            if (!CudaContext.IsAvailable()) Skip = "SKIPPED: CUDA unavailable.";
        }
    }
}
